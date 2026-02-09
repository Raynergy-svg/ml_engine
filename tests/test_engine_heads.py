"""
Integration tests for engine head modules.

Validates:
1. All 5 engine heads importable from src.models.heads
2. Backward-compatible shim modules at repo root re-export correctly
3. TF-conditional instantiation and forward pass
4. get_config() round-trip for serialization
5. Input validation (hidden_size > 0, dropout in [0,1), etc.)
"""

from __future__ import annotations

import pytest

# Skip entire module if TensorFlow is not available
tf = pytest.importorskip("tensorflow", reason="TensorFlow required for engine head tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_input():
    """Create a dummy 3D tensor (batch=2, seq_len=10, features=16)."""
    import numpy as np
    return tf.constant(np.random.randn(2, 10, 16).astype("float32"))


# ---------------------------------------------------------------------------
# Test: Package-level imports
# ---------------------------------------------------------------------------

class TestPackageImports:
    """Verify all 5 heads are importable from the package __init__."""

    def test_import_ml_engine_head(self):
        """MLEngineHead importable from src.models.heads."""
        from src.models.heads import MLEngineHead
        assert MLEngineHead is not None

    def test_import_mr_engine_head(self):
        """MREngineHead importable from src.models.heads."""
        from src.models.heads import MREngineHead
        assert MREngineHead is not None

    def test_import_ms_engine_head(self):
        """MSEngineHead importable from src.models.heads."""
        from src.models.heads import MSEngineHead
        assert MSEngineHead is not None

    def test_import_mt_engine_head(self):
        """MTEngineHead importable from src.models.heads."""
        from src.models.heads import MTEngineHead
        assert MTEngineHead is not None

    def test_import_mx_engine_head(self):
        """MXEngineHead importable from src.models.heads."""
        from src.models.heads import MXEngineHead
        assert MXEngineHead is not None

    def test_all_exports(self):
        """__all__ should list exactly 5 engine heads."""
        from src.models.heads import __all__
        expected = {'MLEngineHead', 'MREngineHead', 'MSEngineHead',
                    'MTEngineHead', 'MXEngineHead'}
        assert set(__all__) == expected


# ---------------------------------------------------------------------------
# Test: Backward-compatible shim imports
# ---------------------------------------------------------------------------

class TestShimImports:
    """Verify repo-root shim modules re-export the correct classes."""

    def test_ml_head_shim(self):
        """ml_head_engine.py shim re-exports MLEngineHead."""
        from ml_head_engine import MLEngineHead as ShimML
        from src.models.heads.ml_head_engine import MLEngineHead as PkgML
        assert ShimML is PkgML, "Shim should re-export the same class object"

    def test_mr_engine_shim(self):
        """mr_engine.py shim re-exports MREngineHead."""
        from mr_engine import MREngineHead as ShimMR
        from src.models.heads.mr_engine import MREngineHead as PkgMR
        assert ShimMR is PkgMR

    def test_ms_head_shim(self):
        """ms_head_engine.py shim re-exports MSEngineHead."""
        from ms_head_engine import MSEngineHead as ShimMS
        from src.models.heads.ms_head_engine import MSEngineHead as PkgMS
        assert ShimMS is PkgMS

    def test_mt_engine_shim(self):
        """mt_engine.py shim re-exports MTEngineHead."""
        from mt_engine import MTEngineHead as ShimMT
        from src.models.heads.mt_engine import MTEngineHead as PkgMT
        assert ShimMT is PkgMT

    def test_mx_head_shim(self):
        """mx_head_engine.py shim re-exports MXEngineHead."""
        from mx_head_engine import MXEngineHead as ShimMX
        from src.models.heads.mx_head_engine import MXEngineHead as PkgMX
        assert ShimMX is PkgMX


# ---------------------------------------------------------------------------
# Test: MLEngineHead instantiation and forward pass
# ---------------------------------------------------------------------------

class TestMLEngineHead:
    """Test MLEngineHead LSTM head."""

    def test_default_instantiation(self):
        """Can create with default parameters."""
        from src.models.heads import MLEngineHead
        head = MLEngineHead()
        assert head.hidden_size == 64
        assert head.num_layers == 2
        assert head.dropout == pytest.approx(0.1)
        assert head.projection_dim is None

    def test_custom_params(self):
        """Can create with custom parameters."""
        from src.models.heads import MLEngineHead
        head = MLEngineHead(hidden_size=128, num_layers=3, dropout=0.2, projection_dim=32)
        assert head.hidden_size == 128
        assert head.num_layers == 3
        assert head.projection_dim == 32

    def test_forward_pass(self, dummy_input):
        """Forward pass produces output of correct shape."""
        from src.models.heads import MLEngineHead
        head = MLEngineHead(hidden_size=32, num_layers=1)
        output = head(dummy_input, training=False)
        # Should be (batch, hidden_size) since last LSTM returns_sequences=False
        assert output.shape == (2, 32), f"Expected (2, 32), got {output.shape}"

    def test_forward_pass_with_projection(self, dummy_input):
        """Forward pass with projection produces projected output."""
        from src.models.heads import MLEngineHead
        head = MLEngineHead(hidden_size=32, num_layers=1, projection_dim=8)
        output = head(dummy_input, training=False)
        assert output.shape == (2, 8), f"Expected (2, 8), got {output.shape}"

    def test_get_config_round_trip(self):
        """get_config returns serializable dict with all parameters."""
        from src.models.heads import MLEngineHead
        head = MLEngineHead(hidden_size=128, num_layers=3, dropout=0.2, projection_dim=64)
        config = head.get_config()

        assert config['hidden_size'] == 128
        assert config['num_layers'] == 3
        assert config['dropout'] == pytest.approx(0.2)
        assert config['projection_dim'] == 64

    def test_invalid_hidden_size_raises(self):
        """hidden_size <= 0 should raise ValueError."""
        from src.models.heads import MLEngineHead
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            MLEngineHead(hidden_size=0)

    def test_invalid_num_layers_raises(self):
        """num_layers <= 0 should raise ValueError."""
        from src.models.heads import MLEngineHead
        with pytest.raises(ValueError, match="num_layers must be positive"):
            MLEngineHead(num_layers=0)

    def test_invalid_dropout_raises(self):
        """dropout >= 1.0 should raise ValueError."""
        from src.models.heads import MLEngineHead
        with pytest.raises(ValueError, match="dropout must be in"):
            MLEngineHead(dropout=1.0)


# ---------------------------------------------------------------------------
# Test: MREngineHead instantiation and forward pass
# ---------------------------------------------------------------------------

class TestMREngineHead:
    """Test MREngineHead LSTM head."""

    def test_default_instantiation(self):
        """Can create MREngineHead with defaults."""
        from src.models.heads import MREngineHead
        head = MREngineHead()
        assert head.hidden_size == 64
        assert head.name == "mr_head"

    def test_forward_pass(self, dummy_input):
        """MREngineHead forward pass works."""
        from src.models.heads import MREngineHead
        head = MREngineHead(hidden_size=32, num_layers=1)
        output = head(dummy_input, training=False)
        assert output.shape == (2, 32)

    def test_get_config(self):
        """get_config returns expected keys."""
        from src.models.heads import MREngineHead
        head = MREngineHead(hidden_size=48)
        config = head.get_config()
        assert config['hidden_size'] == 48


# ---------------------------------------------------------------------------
# Test: All heads have consistent interface
# ---------------------------------------------------------------------------

class TestConsistentInterface:
    """Verify all 5 heads share the same constructor signature and behavior."""

    @pytest.fixture(params=[
        'MLEngineHead', 'MREngineHead', 'MSEngineHead',
        'MTEngineHead', 'MXEngineHead',
    ])
    def head_class(self, request):
        """Parametrized fixture returning each head class."""
        import src.models.heads as heads_pkg
        return getattr(heads_pkg, request.param)

    def test_all_accept_hidden_size(self, head_class):
        """All heads accept hidden_size kwarg."""
        head = head_class(hidden_size=32, num_layers=1)
        assert head.hidden_size == 32

    def test_all_have_get_config(self, head_class):
        """All heads implement get_config()."""
        head = head_class(hidden_size=32, num_layers=1)
        config = head.get_config()
        assert isinstance(config, dict)
        assert 'hidden_size' in config

    def test_all_produce_output(self, head_class, dummy_input):
        """All heads produce tensor output from forward pass."""
        head = head_class(hidden_size=32, num_layers=1)
        output = head(dummy_input, training=False)
        assert output.shape[0] == 2  # batch dimension preserved

    def test_all_reject_invalid_hidden_size(self, head_class):
        """All heads raise ValueError for invalid hidden_size."""
        with pytest.raises(ValueError):
            head_class(hidden_size=-1)

    def test_all_are_keras_models(self, head_class):
        """All heads are subclasses of tf.keras.Model."""
        head = head_class(hidden_size=32, num_layers=1)
        assert isinstance(head, tf.keras.Model)
