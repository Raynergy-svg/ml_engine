"""TensorFlow Unified Engine for Buddy.

This module provides a TensorFlow-based engine that is compatible with Buddy's
prediction interface, allowing Buddy to use TensorFlow TFT models instead of PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Lazy TensorFlow import to avoid startup overhead
_tf = None
_keras = None
_custom_objects = None


def _ensure_tensorflow():
    """Lazy load TensorFlow and register custom objects."""
    global _tf, _keras, _custom_objects
    if _tf is None:
        import tensorflow as tf
        _tf = tf
        _keras = tf.keras
        
        # Import custom model classes for deserialization
        _custom_objects = {}
        try:
            from tensorflow_models import (
                TFTemporalFusionTransformer,
                TFTCNPredictor,
                TFTransformerPredictor,
                TFAttentiveLSTM,
                TFStockPredictor,
                TFEnsemblePredictor,
            )
            _custom_objects.update({
                'TFTemporalFusionTransformer': TFTemporalFusionTransformer,
                'TFTCNPredictor': TFTCNPredictor,
                'TFTransformerPredictor': TFTransformerPredictor,
                'TFAttentiveLSTM': TFAttentiveLSTM,
                'TFStockPredictor': TFStockPredictor,
                'TFEnsemblePredictor': TFEnsemblePredictor,
            })
        except ImportError as e:
            print(f"⚠ Could not import some TensorFlow models: {e}")
            
        # Import custom loss functions (optional).
        # NOTE: `tensorflow_engine.py` may intentionally raise SystemExit in some
        # configurations; never let that break inference/REPL startup.
        try:
            from tensorflow_engine import BinaryFocalLoss
            _custom_objects['BinaryFocalLoss'] = BinaryFocalLoss
        except BaseException:
            pass
            
    return _tf, _keras


@dataclass
class TFUnifiedPrediction:
    """Prediction output compatible with Buddy's expected format."""
    prediction: np.ndarray
    uncertainty: float
    trend: Optional[np.ndarray] = None
    risk: Optional[np.ndarray] = None
    state_probs: Optional[np.ndarray] = None
    direction: Optional[np.ndarray] = None


class _SavedModelWrapper:
    """Wrapper to make SavedModel callable like Keras model."""
    
    def __init__(self, saved_model):
        self.saved_model = saved_model
        # Get the default signature
        self.infer = saved_model.signatures.get('serving_default')
        if self.infer is None:
            # Fallback: try to get any signature
            sig_keys = list(saved_model.signatures.keys())
            if sig_keys:
                self.infer = saved_model.signatures[sig_keys[0]]
            else:
                # Try direct call
                self.infer = None
                
    def __call__(self, inputs, training=False):
        """Run inference."""
        import tensorflow as tf
        inputs = tf.cast(inputs, tf.float32)
        
        if self.infer is not None:
            # Use signature
            result = self.infer(inputs)
            # Convert to list if dict
            if isinstance(result, dict):
                return list(result.values())
            return result
        else:
            # Direct call
            return self.saved_model(inputs, training=training)
            
    def predict(self, inputs):
        """Keras-compatible predict method."""
        return self(inputs, training=False)


def _rebuild_and_load_weights(tf, weights_path):
    """Rebuild TFT model architecture and load weights from H5 file."""
    from tensorflow_models import TFTemporalFusionTransformer
    
    # Build fresh model with same architecture as training
    # NOTE: Training used hidden_size=32 (verified from saved weight shapes)
    model = TFTemporalFusionTransformer(
        input_size=128,
        hidden_size=32,  # Must match training! (was 64 incorrectly)
        num_heads=4,
        dropout=0.35,
        state_classes=3,
        multi_task=True,
    )
    
    # Build model with dummy input
    dummy_input = tf.zeros((1, 120, 128), dtype=tf.float32)
    _ = model(dummy_input)
    
    # Load weights from H5 file (saved by training with save_weights())
    # This preserves weight-to-variable mapping from the training session
    model.load_weights(str(weights_path))
    return model


def _normalize_weight_name(name):
    """Normalize weight name for matching (remove :0, replace / with _, lowercase)."""
    return name.replace(':', '_').replace('/', '_').lower().rstrip('_0123456789')


def _calculate_pattern_score(var_name, saved_name):
    """Calculate similarity score between variable and saved weight names."""
    saved_lower = saved_name.lower()
    score = 0
    
    # Key pattern matching with weights
    patterns = [
        ('direction', 10), ('price', 10), ('trend', 10), ('risk', 10), ('state', 10),
        ('encoder_lstm', 8), ('encoder_vsn', 8), ('decoder', 8),
        ('attention', 7), ('gated', 6), ('projection', 6),
        ('dense', 3), ('layer_norm', 3), ('lstm', 5),
        ('kernel', 2), ('bias', 2), ('gamma', 2), ('beta', 2),
    ]
    
    for pattern, weight in patterns:
        if pattern in var_name and pattern in saved_lower:
            score += weight
    
    # Exact substring matching
    var_parts = var_name.replace(':', '/').replace('_', '/').split('/')
    for part in var_parts:
        if len(part) > 2 and part in saved_lower:
            score += 2
    
    return score


def _match_weights_by_name(model_vars, saved_weights):
    """Match model variables to saved weights by normalized name."""
    matched = 0
    used_saved = set()
    
    for var in model_vars:
        var_norm = _normalize_weight_name(var.name)
        
        for saved_name, saved_weight in saved_weights.items():
            if saved_name in used_saved:
                continue
            saved_norm = _normalize_weight_name(saved_name)
            
            if saved_norm == var_norm and saved_weight.shape == tuple(var.shape):
                # Cast to match dtype for Metal compatibility
                weight_to_assign = saved_weight.astype(var.dtype.as_numpy_dtype) if hasattr(saved_weight, 'astype') else saved_weight
                var.assign(weight_to_assign)
                used_saved.add(saved_name)
                matched += 1
                break
    
    return matched, used_saved


def _match_weights_by_shape_and_pattern(model_vars, saved_weights, used_saved):
    """Match remaining variables by shape and name pattern similarity."""
    remaining_vars = [v for v in model_vars if _normalize_weight_name(v.name) not in 
                      [_normalize_weight_name(s) for s in used_saved]]
    remaining_saved = {k: v for k, v in saved_weights.items() if k not in used_saved}
    
    # Group remaining by shape
    shape_to_saved = {}
    for name, weight in remaining_saved.items():
        shape_key = tuple(weight.shape)
        if shape_key not in shape_to_saved:
            shape_to_saved[shape_key] = []
        shape_to_saved[shape_key].append((name, weight))
    
    matched = 0
    for var in remaining_vars:
        var_shape = tuple(var.shape)
        var_name = var.name.lower()
        
        if var_shape not in shape_to_saved or not shape_to_saved[var_shape]:
            continue
        
        best_match, best_score = _find_best_weight_match(var_name, shape_to_saved[var_shape])
        
        if best_match is not None and best_score > 0:
            # Cast to match dtype for Metal compatibility
            weight_to_assign = best_match[1].astype(var.dtype.as_numpy_dtype) if hasattr(best_match[1], 'astype') else best_match[1]
            var.assign(weight_to_assign)
            shape_to_saved[var_shape] = [
                (n, w) for n, w in shape_to_saved[var_shape] if n != best_match[0]
            ]
            matched += 1

    return matched


def _find_best_weight_match(var_name, candidates):
    """Find the best matching weight from candidates based on name similarity."""
    best_match = None
    best_score = -1
    
    for saved_name, saved_weight in candidates:
        score = _calculate_pattern_score(var_name, saved_name)
        if score > best_score:
            best_score = score
            best_match = (saved_name, saved_weight)
    
    return best_match, best_score


def _load_from_npz(tf, npz_path):
    """Load weights from NPZ file saved during training.
    
    The NPZ file contains weights with variable names (colons replaced with underscores).
    We rebuild the model and match weights by comparing normalized names.
    """
    import numpy as np
    from tensorflow_models import TFTemporalFusionTransformer
    
    # Load saved weights
    weights_data = np.load(str(npz_path))
    saved_weights = {k: weights_data[k] for k in weights_data.files}
    
    print(f"  NPZ contains {len(saved_weights)} weight arrays")
    
    # Build fresh model with same architecture as training
    model = TFTemporalFusionTransformer(
        input_size=128,
        hidden_size=32,
        num_heads=4,
        dropout=0.35,
        state_classes=3,
        multi_task=True,
    )
    
    # Build model with dummy input
    dummy_input = tf.zeros((1, 120, 128), dtype=tf.float32)
    _ = model(dummy_input)
    
    # Get model variables
    model_vars = model.trainable_variables + model.non_trainable_variables
    
    # First try: exact name matching
    matched, used_saved = _match_weights_by_name(model_vars, saved_weights)
    
    # Second try: shape + pattern matching for remaining
    matched += _match_weights_by_shape_and_pattern(model_vars, saved_weights, used_saved)
    
    print(f"  Matched {matched}/{len(model_vars)} weight tensors from NPZ")
    
    if matched < len(model_vars) * 0.5:
        print("  ⚠ Warning: Less than 50% weights matched. Model may not work correctly.")
    
    return model


def _load_weights_from_npz(model, npz_path):
    """Load weights from NPZ file by matching layer names and shapes."""
    import numpy as np
    
    weights_data = np.load(str(npz_path))
    
    # Create mapping from saved weight name patterns to model weights
    saved_weights = {k: weights_data[k] for k in weights_data.files}
    
    # Get all trainable variables from model
    model_vars = model.trainable_variables + model.non_trainable_variables
    
    matched = 0
    unmatched = []
    
    for var in model_vars:
        var_name = var.name
        var_shape = tuple(var.shape)
        
        # Try to find matching saved weight by name pattern and shape
        best_match = None
        
        for saved_name, saved_weight in saved_weights.items():
            if saved_weight.shape == var_shape:
                # Check if names have similar patterns
                var_parts = var_name.lower().replace(':', '/').split('/')
                saved_parts = saved_name.lower().split('/')
                
                # Look for key matches (direction_head, price, encoder_lstm, etc.)
                common_keys = set(var_parts) & set(saved_parts)
                if common_keys or any(p in saved_name.lower() for p in var_parts[-3:]):
                    best_match = saved_name
                    break
        
        if best_match is not None:
            # Cast to match dtype for Metal compatibility
            weight_to_assign = saved_weights[best_match]
            if hasattr(weight_to_assign, 'astype'):
                weight_to_assign = weight_to_assign.astype(var.dtype.as_numpy_dtype)
            var.assign(weight_to_assign)
            del saved_weights[best_match]  # Remove to avoid reuse
            matched += 1
        else:
            unmatched.append(f"{var_name} {var_shape}")
    
    print(f"  Loaded {matched}/{len(model_vars)} weights")
    if unmatched:
        print(f"  ⚠ {len(unmatched)} unmatched variables")
    
    return model


def _extract_keras_config(keras_path):
    """Extract model configuration from .keras archive."""
    import zipfile
    import json
    
    with zipfile.ZipFile(str(keras_path), 'r') as zf:
        config_data = json.loads(zf.read('config.json'))
    
    # Default values
    input_size = 128
    hidden_size = 64
    state_classes = 3
    
    # Override with saved config if available
    saved_config = config_data.get('config', {})
    if 'input_size' in saved_config:
        input_size = saved_config['input_size']
    if 'hidden_size' in saved_config:
        hidden_size = saved_config['hidden_size']
    if 'state_classes' in saved_config:
        state_classes = saved_config['state_classes']
    
    return input_size, hidden_size, state_classes


def _calculate_keras_weight_score(var_name, saved_name):
    """Calculate similarity score for keras archive weight matching."""
    saved_lower = saved_name.lower()
    score = 0
    
    # Key patterns to match
    patterns = [
        'direction', 'price', 'trend', 'risk', 'state',
        'encoder', 'decoder', 'lstm', 'attention',
        'dense', 'kernel', 'bias', 'projection'
    ]
    for pattern in patterns:
        if pattern in var_name and pattern in saved_lower:
            score += 10
    
    # Position-based matching (vars/0, vars/1)
    if 'vars/0' in saved_lower and ('kernel' in var_name or '/0' in var_name):
        score += 5
    if 'vars/1' in saved_lower and ('bias' in var_name or '/1' in var_name):
        score += 5
    
    return score


def _match_keras_weights_to_vars(model_vars, weights_dict):
    """Match model variables to weights from keras archive by shape and pattern."""
    # Create shape->weights mapping
    shape_to_saved = {}
    for name, weight in weights_dict.items():
        shape_key = tuple(weight.shape)
        if shape_key not in shape_to_saved:
            shape_to_saved[shape_key] = []
        shape_to_saved[shape_key].append((name, weight))
    
    matched = 0
    for var in model_vars:
        var_shape = tuple(var.shape)
        var_name = var.name.lower()
        
        if var_shape not in shape_to_saved:
            continue
        
        # Find best match by score
        best_match = None
        best_score = 0
        
        for saved_name, saved_weight in shape_to_saved[var_shape]:
            score = _calculate_keras_weight_score(var_name, saved_name)
            if score > best_score:
                best_score = score
                best_match = (saved_name, saved_weight)
        
        if best_match is not None:
            # Cast to match dtype for Metal compatibility
            weight_to_assign = best_match[1].astype(var.dtype.as_numpy_dtype) if hasattr(best_match[1], 'astype') else best_match[1]
            var.assign(weight_to_assign)
            shape_to_saved[var_shape] = [
                (n, w) for n, w in shape_to_saved[var_shape]
                if n != best_match[0]
            ]
            matched += 1

    return matched


def _read_h5_weights(weights_file):
    """Read weights from H5 file into a dictionary."""
    import h5py
    import numpy as np
    
    weights_dict = {}
    
    def read_weight(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.shape != ():
            try:
                weights_dict[name] = np.array(obj)
            except (ValueError, OSError):
                pass
    
    with h5py.File(str(weights_file), 'r') as f:
        f.visititems(read_weight)
    
    return weights_dict


def _load_from_keras_archive(tf, keras_path):
    """Extract weights from .keras archive and load into fresh model."""
    import zipfile
    import tempfile
    
    from tensorflow_models import TFTemporalFusionTransformer
    
    # Extract configuration
    input_size, hidden_size, state_classes = _extract_keras_config(keras_path)
    
    print(f"  Rebuilding TFT with: input_size={input_size}, hidden_size={hidden_size}, state_classes={state_classes}")
    
    # Build fresh model
    model = TFTemporalFusionTransformer(
        input_size=input_size,
        hidden_size=hidden_size,
        num_heads=4,
        dropout=0.35,
        state_classes=state_classes,
        multi_task=True,
    )
    
    # Build model with dummy input
    dummy_input = tf.zeros((1, 120, input_size), dtype=tf.float32)
    _ = model(dummy_input)
    
    # Extract and load weights
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(str(keras_path), 'r') as zf:
            zf.extract('model.weights.h5', tmpdir)
            weights_file = Path(tmpdir) / 'model.weights.h5'
            
            # Load weights from H5 file
            weights_dict = _read_h5_weights(weights_file)
            
            # Match weights to model variables
            model_vars = model.trainable_variables + model.non_trainable_variables
            matched = _match_keras_weights_to_vars(model_vars, weights_dict)
            
            print(f"  Matched {matched}/{len(model_vars)} weight tensors")
    
    return model


def _try_load_savedmodel(tf, saved_model_path):
    """Try loading from TensorFlow SavedModel format."""
    if not saved_model_path.exists() or not (saved_model_path / 'saved_model.pb').exists():
        return None
    
    print(f"🧠 Loading TensorFlow SavedModel from: {saved_model_path}")
    try:
        model = tf.saved_model.load(str(saved_model_path))
        model = _SavedModelWrapper(model)
        print("✓ Loaded from SavedModel format")
        return model
    except Exception as e:
        print(f"⚠ SavedModel load failed: {e}")
        return None


def _try_load_weights_h5(tf, weights_path, format_name):
    """Try loading from H5 weights file."""
    if not weights_path.exists():
        return None
    
    print(f"🧠 Loading weights from: {weights_path}")
    try:
        model = _rebuild_and_load_weights(tf, weights_path)
        print(f"✓ Loaded from {format_name}")
        return model
    except Exception as e:
        print(f"⚠ {format_name} load failed: {e}")
        return None


def _try_load_npz(tf, weights_npz_path):
    """Try loading from NPZ weights file."""
    if not weights_npz_path.exists():
        return None
    
    print(f"🧠 Loading weights from NPZ: {weights_npz_path}")
    try:
        model = _load_from_npz(tf, weights_npz_path)
        print("✓ Loaded from NPZ weights file")
        return model
    except Exception as e:
        print(f"⚠ NPZ weights load failed: {e}")
        return None


def _try_load_keras_archive(tf, checkpoint_path):
    """Try loading from .keras archive."""
    if not checkpoint_path.exists() or not str(checkpoint_path).endswith('.keras'):
        return None
    
    print(f"🧠 Extracting weights from: {checkpoint_path}")
    try:
        model = _load_from_keras_archive(tf, checkpoint_path)
        print("✓ Loaded from .keras archive")
        return model
    except Exception as e:
        print(f"⚠ .keras archive extraction failed: {e}")
        return None


def _infer_state_classes(model):
    """Try to infer state_classes from model output shape."""
    try:
        if hasattr(model, 'outputs'):
            for output in model.outputs:
                if 'state' in output.name.lower():
                    return output.shape[-1]
    except Exception:
        pass
    return 3  # default


class TensorFlowUnifiedEngine:
    """TensorFlow engine wrapper compatible with Buddy's UnifiedNeuralEngine interface.
    
    This allows Buddy to use TensorFlow TFT models trained via train_visual.py.
    """
    
    def __init__(self, model, config: Dict[str, Any] = None):
        """Initialize with a loaded Keras model.
        
        Args:
            model: Loaded Keras model
            config: Optional configuration dict
        """
        self.model = model
        self.config = config or {}
        self.state_classes = config.get('state_classes', 3) if config else 3
        
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = None,  # Ignored for TF, but kept for API compatibility
    ) -> "TensorFlowUnifiedEngine":
        """Load a TensorFlow model from SavedModel or .keras checkpoint.
        
        Args:
            checkpoint_path: Path to .keras file or directory containing SavedModel
            device: Ignored (TensorFlow handles device placement automatically)
            
        Returns:
            TensorFlowUnifiedEngine instance
        """
        tf, _ = _ensure_tensorflow()
        
        checkpoint_path = Path(checkpoint_path)
        checkpoint_dir = checkpoint_path.parent
        
        # Define checkpoint paths
        saved_model_path = checkpoint_dir / 'tf_model_savedmodel'
        weights_h5_path = checkpoint_dir / 'tf_model_weights.weights.h5'
        weights_h5_old_path = checkpoint_dir / 'tf_model_weights.h5'
        weights_npz_path = checkpoint_dir / 'tf_model_weights.npz'
        
        # Try loading strategies in order of preference
        model = _try_load_savedmodel(tf, saved_model_path)
        if model is None:
            model = _try_load_weights_h5(tf, weights_h5_path, "Keras 3 weights file")
        if model is None:
            model = _try_load_weights_h5(tf, weights_h5_old_path, "legacy H5 weights file")
        if model is None:
            model = _try_load_npz(tf, weights_npz_path)
        if model is None:
            model = _try_load_keras_archive(tf, checkpoint_path)
        
        if model is None:
            raise RuntimeError(
                f"Failed to load TensorFlow model.\n"
                f"Tried:\n"
                f"  1. SavedModel at: {saved_model_path}\n"
                f"  2. Keras 3 weights at: {weights_h5_path}\n"
                f"  3. NPZ weights at: {weights_npz_path}\n"
                f"  4. .keras archive at: {checkpoint_path}\n"
                f"\nTo create compatible checkpoints, run training:\n"
                f"  python3 train_visual.py --framework tensorflow --model tft --multi-task --data-dir trained_data/data"
            )
        
        state_classes = _infer_state_classes(model)
        config = {
            'checkpoint_path': str(checkpoint_path),
            'state_classes': state_classes,
        }
        
        print(f"✓ TensorFlow TFT model loaded (state_classes={state_classes})")
        return cls(model, config)
    
    def _convert_input_to_numpy(self, x):
        """Convert input tensor to numpy array with correct dtype."""
        if hasattr(x, 'numpy'):  # torch tensor
            x_np = x.numpy()
        elif hasattr(x, 'detach'):  # torch tensor with grad
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        return x_np.astype(np.float32)
    
    def _run_model_inference(self, x_np):
        """Run model inference and return outputs."""
        tf, _ = _ensure_tensorflow()
        
        model_inputs = x_np
        try:
            input_names = list(getattr(self.model, "input_names", []) or [])
            if len(input_names) == 1 and input_names[0] == "features":
                model_inputs = {"features": x_np}
        except Exception:
            model_inputs = x_np
        
        # Prefer direct call for single-batch inference to avoid overhead/hanging
        # associated with model.predict() in some environments.
        try:
            outputs = self.model(model_inputs, training=False)
        except Exception:
            # Fallback to predict() if direct call fails (e.g. for some SavedModels)
            if hasattr(self.model, 'predict'):
                outputs = self.model.predict(model_inputs, verbose=0)
            else:
                raise
        
        if isinstance(outputs, (list, tuple)):
            return [o.numpy() if hasattr(o, 'numpy') else o for o in outputs]
        return outputs
    
    def _parse_dict_outputs(self, outputs, result):
        """Parse dict-style model outputs."""
        tf, _ = _ensure_tensorflow()
        
        if 'price' in outputs:
            result['prediction'] = outputs['price']
            pred_arr = np.asarray(outputs['price'])
            result['uncertainty'] = float(np.std(pred_arr)) if pred_arr.size > 1 else 0.1
        if 'trend' in outputs:
            result['trend'] = outputs['trend']
        if 'direction' in outputs:
            result['direction'] = outputs['direction']
        if 'risk' in outputs:
            result['risk'] = outputs['risk']
        if 'state_logits' in outputs:
            # Temperature scaling: lower temp = sharper/more confident probabilities
            # Use temp=0.5 to make model more decisive when it has a clear winner
            temperature = 0.5
            result['state_probs'] = tf.nn.softmax(outputs['state_logits'] / temperature, axis=-1).numpy()
        elif 'state' in outputs:
            temperature = 0.5
            result['state_probs'] = tf.nn.softmax(outputs['state'] / temperature, axis=-1).numpy()
    
    def _parse_list_outputs_standard(self, outputs, result):
        """Parse list-style outputs with standard TFT ordering."""
        tf, _ = _ensure_tensorflow()
        
        result['prediction'] = outputs[0]
        result['uncertainty'] = float(np.std(outputs[0])) if outputs[0].size > 1 else 0.1
        result['trend'] = outputs[1]
        result['direction'] = outputs[2]
        result['risk'] = outputs[3]
        # Temperature scaling: lower temp = sharper/more confident probabilities
        temperature = 0.5
        result['state_probs'] = tf.nn.softmax(outputs[4] / temperature, axis=-1).numpy()
    
    def _parse_list_outputs_by_name(self, outputs, result):
        """Parse list-style outputs by matching output names."""
        tf, _ = _ensure_tensorflow()
        
        output_names = []
        if hasattr(self.model, 'outputs') and self.model.outputs:
            output_names = [o.name.split('/')[0].lower() for o in self.model.outputs]
        
        for i, out in enumerate(outputs):
            name = output_names[i] if i < len(output_names) else f'output_{i}'
            self._assign_output_by_name(name, out, result, i, tf)
    
    def _assign_output_by_name(self, name, out, result, index, tf):
        """Assign a single output to the result dict based on name."""
        if 'price' in name or index == 0:
            result['prediction'] = out
            result['uncertainty'] = float(np.std(out)) if out.size > 1 else 0.1
        elif 'trend' in name:
            result['trend'] = out
        elif 'direction' in name:
            result['direction'] = out
        elif 'risk' in name:
            result['risk'] = out
        elif 'state' in name:
            # Temperature scaling: lower temp = sharper/more confident probabilities
            temperature = 0.5
            result['state_probs'] = tf.nn.softmax(out / temperature, axis=-1).numpy()
    
    def predict(self, x) -> Dict[str, Any]:
        """Run prediction on input tensor.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, features)
               Can be numpy array or torch tensor (will be converted)
               
        Returns:
            Dict with prediction outputs matching Buddy's expected format.
        """
        x_np = self._convert_input_to_numpy(x)
        outputs = self._run_model_inference(x_np)
        
        result = {
            'prediction': None,
            'uncertainty': 0.0,
            'trend': None,
            'risk': None,
            'state_probs': None,
            'direction': None,
        }
        
        if isinstance(outputs, dict):
            self._parse_dict_outputs(outputs, result)
        elif isinstance(outputs, (list, tuple)) and len(outputs) >= 5:
            self._parse_list_outputs_standard(outputs, result)
        elif isinstance(outputs, (list, tuple)):
            self._parse_list_outputs_by_name(outputs, result)
        else:
            result['prediction'] = outputs
            result['uncertainty'] = float(np.std(outputs)) if outputs.size > 1 else 0.1
            
        return result
    
    def get_confidence_level(self, result: Dict[str, Any]) -> str:
        """Determine confidence level from prediction results.
        
        Args:
            result: Prediction result dict
            
        Returns:
            'HIGH', 'MEDIUM', or 'LOW'
        """
        # Use state probabilities for confidence
        state_probs = result.get('state_probs')
        if state_probs is not None:
            max_prob = float(np.max(state_probs))
            if max_prob > 0.7:
                return 'HIGH'
            elif max_prob > 0.5:
                return 'MEDIUM'
            else:
                return 'LOW'
        
        # Fallback to direction confidence
        direction = result.get('direction')
        if direction is not None:
            dir_val = float(np.mean(direction))
            # Direction close to 0.5 = uncertain, close to 0 or 1 = confident
            dir_confidence = abs(dir_val - 0.5) * 2  # 0 to 1
            if dir_confidence > 0.7:
                return 'HIGH'
            elif dir_confidence > 0.4:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        return 'MEDIUM'  # Default
    
    def get_signal(self, result: Dict[str, Any], last_close: float = None) -> str:
        """Determine trading signal from prediction results.
        
        Args:
            result: Prediction result dict
            last_close: Last known close price
            
        Returns:
            'BUY', 'SELL', or 'HOLD'
        """
        # Primary signal from direction head
        direction = result.get('direction')
        if direction is not None:
            dir_val = float(np.mean(direction))
            if dir_val > 0.6:
                return 'BUY'
            elif dir_val < 0.4:
                return 'SELL'
        
        # Fallback to price prediction vs last close
        prediction = result.get('prediction')
        if prediction is not None and last_close is not None:
            pred_val = float(np.mean(prediction))
            if pred_val > last_close * 1.001:  # >0.1% up
                return 'BUY'
            elif pred_val < last_close * 0.999:  # >0.1% down
                return 'SELL'
        
        # Fallback to trend
        trend = result.get('trend')
        if trend is not None:
            trend_val = float(np.mean(trend))
            if trend_val > 0.1:
                return 'BUY'
            elif trend_val < -0.1:
                return 'SELL'
                
        return 'HOLD'


def select_latest_tf_checkpoint(model_dir: Path) -> Optional[Path]:
    """Find the newest TensorFlow checkpoint in a directory.
    
    Args:
        model_dir: Directory to search
        
    Returns:
        Path to newest .keras file, or None
    """
    if not model_dir.exists() or not model_dir.is_dir():
        return None
        
    candidates = list(model_dir.glob("*.keras"))
    if not candidates:
        return None
        
    # Sort by modification time, newest first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def get_default_tf_checkpoint() -> str:
    """Get the default TensorFlow checkpoint path.
    
    Returns:
        Path to default TensorFlow checkpoint
    """
    return "trained_data/checkpoints/tensorflow/tf_model_best.keras"


def is_tensorflow_checkpoint(path: str) -> bool:
    """Check if a path is a TensorFlow checkpoint.
    
    Args:
        path: Path to check
        
    Returns:
        True if path ends with .keras or .h5
    """
    path_lower = str(path).lower()
    return path_lower.endswith('.keras') or path_lower.endswith('.h5')


def is_pytorch_checkpoint(path: str) -> bool:
    """Check if a path is a PyTorch checkpoint.
    
    Args:
        path: Path to check
        
    Returns:
        True if path ends with .pth or .pt
    """
    path_lower = str(path).lower()
    return path_lower.endswith('.pth') or path_lower.endswith('.pt')
# — Raynergy-svg —
