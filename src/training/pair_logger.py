"""
Per-pair training logger for structured JSON logging.

Enables targeted debugging and historical trend analysis for individual pairs.
"""
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional


class PairTrainingLogger:
    """Structured JSON logger for pair-specific training events."""
    
    def __init__(self, pair_name: str, log_dir: str = "trained_data/logs"):
        self.pair_name = pair_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_path = self.log_dir / f"{pair_name.lower()}_training.jsonl"
        self.logger = logging.getLogger(f"train.{pair_name}")
        
        # Setup file handler for JSON lines
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_path)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_event(self, event: str, **kwargs: Any) -> None:
        """Log a structured event as JSON."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "pair": self.pair_name,
            "event": event,
            **kwargs
        }
        self.logger.info(json.dumps(record))
    
    def log_fold_complete(
        self,
        fold: int,
        train_acc: float,
        val_acc: float,
        bootstrap_ci_lower: Optional[float] = None,
        cv_std: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """Log fold completion with standard metrics."""
        self.log_event(
            "fold_complete",
            fold=fold,
            train_acc=train_acc,
            val_acc=val_acc,
            bootstrap_ci_lower=bootstrap_ci_lower,
            cv_std=cv_std,
            **kwargs
        )
    
    def log_training_complete(
        self,
        final_train_acc: float,
        final_val_acc: float,
        deployment_approved: bool,
        critical_failures: int,
        **kwargs: Any
    ) -> None:
        """Log training completion with final metrics."""
        self.log_event(
            "training_complete",
            final_train_acc=final_train_acc,
            final_val_acc=final_val_acc,
            deployment_approved=deployment_approved,
            critical_failures=critical_failures,
            **kwargs
        )
    
    def log_error(self, error_type: str, message: str, **kwargs: Any) -> None:
        """Log an error event."""
        self.log_event(
            "error",
            error_type=error_type,
            message=message,
            **kwargs
        )


def get_pair_logger(pair_name: str, log_dir: str = "trained_data/logs") -> PairTrainingLogger:
    """Factory function to get or create a pair logger."""
    return PairTrainingLogger(pair_name, log_dir)
