"""Custom exceptions for training module.

This module defines custom exception classes for the training pipeline,
enabling precise error handling and diagnostic logging.
"""

from typing import Dict, List, Any, Optional


class WeightMismatchError(Exception):
    """Raised when weight loading produces shape or name mismatches.
    
    This exception is raised during model weight transfer when:
    - Weight shapes don't match between source and target
    - Weight names cannot be matched by pattern
    - Critical weights are missing from the checkpoint
    
    Attributes:
        message: Human-readable error description
        mismatches: Dictionary containing detailed mismatch information
            - 'shape_mismatches': List of (layer_name, expected_shape, actual_shape)
            - 'name_mismatches': List of (model_var, closest_saved_var)
            - 'missing_weights': List of weight names not found in checkpoint
    """
    
    def __init__(
        self,
        message: str,
        mismatches: Optional[Dict[str, List[Any]]] = None
    ) -> None:
        """Initialize WeightMismatchError.
        
        Args:
            message: Human-readable error description
            mismatches: Optional dictionary containing detailed mismatch info
        """
        super().__init__(message)
        self.mismatches = mismatches or {}
        
    def __str__(self) -> str:
        """Return detailed error message including mismatch information."""
        base_msg = super().__str__()
        if not self.mismatches:
            return base_msg
            
        details = []
        if 'shape_mismatches' in self.mismatches:
            shape_errors = self.mismatches['shape_mismatches']
            if shape_errors:
                shape_str = ", ".join(f"{n}:{e}!={a}" for n, e, a in shape_errors[:3])
                details.append(f"Shape mismatches ({len(shape_errors)}): {shape_str}")
        if 'name_mismatches' in self.mismatches:
            name_errors = self.mismatches['name_mismatches']
            if name_errors:
                details.append(f"Name mismatches ({len(name_errors)})")
        if 'missing_weights' in self.mismatches:
            missing = self.mismatches['missing_weights']
            if missing:
                details.append(f"Missing weights ({len(missing)}): {missing[:3]}")
        elif 'unmatched_vars' in self.mismatches:
            unmatched = self.mismatches['unmatched_vars']
            if unmatched:
                details.append(f"Unmatched vars ({len(unmatched)}): {unmatched[:3]}")
                
        if details:
            return f"{base_msg}\n  Details: {'; '.join(details)}"
        return base_msg


class CheckpointValidationError(Exception):
    """Raised when checkpoint validation fails.
    
    This exception is raised during checkpoint loading when:
    - Checkpoint file is corrupted or unreadable
    - Checkpoint format is incompatible with current model
    - Required metadata is missing
    """
    pass
