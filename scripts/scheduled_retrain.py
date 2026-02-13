#!/usr/bin/env python3
"""
Scheduled Joint Model Retraining Script.

This script is designed to be run via launchd on Mon/Wed/Fri at 10 AM UTC.
It trains joint models for all scanner pairs and sends failure-only email alerts.

Usage:
    python scripts/scheduled_retrain.py
    
    # Or with custom pairs:
    python scripts/scheduled_retrain.py --pairs EUR_USD,GBP_USD,USD_JPY
    
Schedule (launchd):
    Mon/Wed/Fri at 10 AM UTC during London session
    See: scripts/com.mlengine.retrain.plist

Email Alerts:
    - Only sent on FAILURE (no spam on success)
    - Recipient: dcertan84@gmail.com
"""

import argparse
import logging
import os
import smtplib
import sys
import traceback
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Default pairs for scanner
DEFAULT_SCANNER_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY",
    "EUR_AUD", "GBP_AUD", "EUR_CHF", "GBP_CHF",
]

# Email configuration
EMAIL_RECIPIENT = "dcertan84@gmail.com"
EMAIL_SENDER = os.getenv("RETRAIN_EMAIL_SENDER", "mlengine-retrain@localhost")
SMTP_HOST = os.getenv("RETRAIN_SMTP_HOST", "localhost")
SMTP_PORT = int(os.getenv("RETRAIN_SMTP_PORT", "25"))


def send_failure_email(
    subject: str,
    body: str,
    recipient: str = EMAIL_RECIPIENT,
) -> bool:
    """Send failure notification email.
    
    Args:
        subject: Email subject
        body: Email body (plain text)
        recipient: Email recipient
        
    Returns:
        True if email sent successfully
    """
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = recipient
        msg["Subject"] = f"[ML Engine] {subject}"
        
        # Add body
        msg.attach(MIMEText(body, "plain"))
        
        # Try to send via SMTP
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.sendmail(EMAIL_SENDER, [recipient], msg.as_string())
        
        logger.info(f"Failure email sent to {recipient}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to send email: {e}")
        # Fallback: write to a file for manual review
        alert_file = LOG_DIR / f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        alert_file.write_text(f"Subject: {subject}\n\n{body}")
        logger.info(f"Alert written to {alert_file}")
        return False


def run_train_joint(
    pairs: List[str],
    granularity: str = "H1",
    candles: int = 15000,
) -> tuple[bool, str, Optional[dict]]:
    """Run joint training for specified pairs.
    
    Args:
        pairs: List of currency pairs
        granularity: Timeframe
        candles: Number of candles to fetch
        
    Returns:
        Tuple of (success, message, result_dict)
    """
    logger.info(f"Starting joint training for {len(pairs)} pairs...")
    logger.info(f"Pairs: {', '.join(pairs)}")
    logger.info(f"Granularity: {granularity}, Candles: {candles}")
    
    try:
        from src.training.buddy_training_helpers import train_joint_multi_pair_ensemble
        from rich.console import Console
        
        console = Console(file=sys.stdout, force_terminal=True)
        
        result = train_joint_multi_pair_ensemble(
            instruments=pairs,
            granularity=granularity,
            candles=candles,
            fine_tune=True,
            fine_tune_threshold=0.05,
            console=console,
        )
        
        if result.get("status") == "success":
            msg = (
                f"Joint training completed successfully\n"
                f"Instruments: {result.get('n_instruments', 0)}\n"
                f"Save directory: {result.get('joint_save_dir', 'N/A')}"
            )
            logger.info(msg)
            return True, msg, result
        else:
            error = result.get("error", "Unknown error")
            msg = f"Joint training failed: {error}"
            logger.error(msg)
            return False, msg, result
            
    except Exception as e:
        error_trace = traceback.format_exc()
        msg = f"Joint training exception: {e}\n\n{error_trace}"
        logger.error(msg)
        return False, msg, None


def validate_models(model_dir: Path) -> tuple[bool, str]:
    """Validate that required models exist after training.
    
    Args:
        model_dir: Path to joint model directory
        
    Returns:
        Tuple of (success, message)
    """
    required_models = [
        "transformer_direction.keras",
        "tcn_volatility_regime.keras",  # REQUIRED for scanner
        "ridge_confidence.pkl",
    ]
    
    optional_models = [
        "catboost_momentum.cbm",
        "xgb_momentum.pkl",
        "rf_risk.pkl",
        "meta_labeler.pkl",
    ]
    
    missing_required = []
    missing_optional = []
    
    for model in required_models:
        if not (model_dir / model).exists():
            missing_required.append(model)
    
    for model in optional_models:
        if not (model_dir / model).exists():
            missing_optional.append(model)
    
    if missing_required:
        msg = f"CRITICAL: Missing required models: {', '.join(missing_required)}"
        logger.error(msg)
        return False, msg
    
    if missing_optional:
        logger.warning(f"Missing optional models: {', '.join(missing_optional)}")
    
    msg = "All required models validated successfully"
    logger.info(msg)
    return True, msg


def main():
    """Main entry point for scheduled retraining."""
    parser = argparse.ArgumentParser(
        description="Scheduled joint model retraining for ML Engine scanner",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default=",".join(DEFAULT_SCANNER_PAIRS),
        help="Comma-separated list of pairs to train",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="H1",
        help="Timeframe (default: H1)",
    )
    parser.add_argument(
        "--candles",
        type=int,
        default=15000,
        help="Number of candles to fetch (default: 15000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be done without actually training",
    )
    args = parser.parse_args()
    
    # Parse pairs
    pairs = [p.strip().upper().replace("/", "_") for p in args.pairs.split(",")]
    
    # Expand "all" to the full list of default pairs
    if len(pairs) == 1 and pairs[0] == "ALL":
        from src.scanner.config import DEFAULT_PAIRS
        pairs = DEFAULT_PAIRS.copy()
        logger.info(f"Expanded 'all' to {len(pairs)} pairs: {', '.join(pairs)}")
    
    # Log start
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("ML ENGINE SCHEDULED JOINT RETRAINING")
    logger.info(f"Start time: {start_time.isoformat()}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("[DRY RUN] Would train the following pairs:")
        for pair in pairs:
            logger.info(f"  - {pair}")
        logger.info("[DRY RUN] Exiting without training")
        return 0
    
    # Run training
    success, message, _result = run_train_joint(
        pairs=pairs,
        granularity=args.granularity,
        candles=args.candles,
    )
    
    # Validate models
    model_dir = PROJECT_ROOT / "trained_data" / "models" / "joint"
    if success:
        valid, valid_msg = validate_models(model_dir)
        if not valid:
            success = False
            message = f"{message}\n\nValidation failed: {valid_msg}"
    
    # Log completion
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    
    logger.info("=" * 60)
    logger.info(f"RETRAINING {'COMPLETED' if success else 'FAILED'}")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("=" * 60)
    
    # Send failure email (no spam on success)
    if not success:
        email_body = f"""
ML Engine Joint Retraining Failed

Time: {start_time.isoformat()}
Duration: {duration:.1f} seconds
Pairs: {', '.join(pairs)}

Error Details:
{message}

Log file: {LOG_FILE}

---
This is an automated alert. No action required if you're already investigating.
        """.strip()
        
        send_failure_email(
            subject=f"Joint Retraining FAILED - {start_time.strftime('%Y-%m-%d')}",
            body=email_body,
        )
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
