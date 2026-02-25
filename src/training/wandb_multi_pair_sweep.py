"""Multi-Pair W&B Sweep Runner for FX Trading Correlation Groups.

This module runs W&B sweeps across multiple FX pairs organized by correlation groups.
It supports parallel sweep agents for efficient hyperparameter optimization.

Reference:
    https://docs.wandb.ai/guides/sweeps

Correlation Groups (from analysis):
    - Euro Basket: EUR/USD, EUR/JPY, EUR/GBP, EUR/AUD (0.85-0.92 correlation)
    - Yen Carry: EUR/JPY, GBP/JPY, AUD/JPY, NZD/JPY, USD/JPY (0.75-0.90)
    - Sterling Cluster: GBP/USD, GBP/JPY, GBP/AUD, EUR/GBP (0.78-0.88)
    - Aussie/Kiwi: AUD/USD, NZD/USD, AUD/JPY, NZD/JPY (0.80-0.90)
    - USD Safe-Haven: USD/JPY, USD/CHF, USD/CAD (0.70-0.85)

Usage:
    # Run sweep on all pairs in a group
    python -m src.training.wandb_multi_pair_sweep --group euro_basket

    # Run sweep on specific pairs
    python -m src.training.wandb_multi_pair_sweep --pairs EUR_USD,GBP_USD,USD_JPY

    # Run master pairs only (best from each group)
    python -m src.training.wandb_multi_pair_sweep --master-pairs

    # Run all pairs with multiple agents
    python -m src.training.wandb_multi_pair_sweep --all --agents 4

Example:
    ```python
    from src.training.wandb_multi_pair_sweep import (
        CORRELATION_GROUPS,
        create_multi_pair_sweep,
        run_sweep_for_pairs,
    )

    # Get pairs for a group
    pairs = CORRELATION_GROUPS["euro_basket"]["pairs"]

    # Create and run sweep
    sweep_id = create_multi_pair_sweep(pairs, sweep_type="tcn")
    run_sweep_for_pairs(sweep_id, pairs, count=10)
    ```
"""

from __future__ import annotations

import logging
import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Correlation Group Definitions
# ═══════════════════════════════════════════════════════════════════════════

CORRELATION_GROUPS: Dict[str, Dict[str, Any]] = {
    "euro_basket": {
        "name": "Euro Basket",
        "pairs": ["EUR_USD", "EUR_JPY", "EUR_GBP", "EUR_AUD"],
        "avg_correlation": (0.85, 0.92),
        "description": "Euro strength/weakness drives them all",
        "master_pair": "EUR_USD",
        "master_reason": "Most liquid, cleanest signal",
    },
    "yen_carry": {
        "name": "Yen Carry / Risk-On",
        "pairs": ["EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY", "USD_JPY"],
        "avg_correlation": (0.75, 0.90),
        "description": "Yen weakens on risk-on, strengthens on fear",
        "master_pair": "EUR_JPY",
        "master_reason": "70% winner - transfers best",
    },
    "sterling_cluster": {
        "name": "Sterling Cluster",
        "pairs": ["GBP_USD", "GBP_JPY", "GBP_AUD", "EUR_GBP"],
        "avg_correlation": (0.78, 0.88),
        "description": "UK data + Brexit noise, tracks USD & yen",
        "master_pair": "GBP_USD",
        "master_reason": "Highest volume, least noise",
    },
    "aussie_kiwi": {
        "name": "Aussie / Kiwi Commodity",
        "pairs": ["AUD_USD", "NZD_USD", "AUD_JPY", "NZD_JPY"],
        "avg_correlation": (0.80, 0.90),
        "description": "Commodity prices + China growth",
        "master_pair": "AUD_USD",
        "master_reason": "More volume than NZD",
    },
    "usd_safe_haven": {
        "name": "USD Safe-Haven",
        "pairs": ["USD_JPY", "USD_CHF", "USD_CAD"],
        "avg_correlation": (0.70, 0.85),
        "description": "USD strength vs risk-off (yen, franc)",
        "master_pair": "USD_JPY",
        "master_reason": "Most responsive to sentiment",
    },
    "crosses": {
        "name": "Crosses (less common)",
        "pairs": ["GBP_AUD", "EUR_AUD", "AUD_NZD"],
        "avg_correlation": (0.65, 0.80),
        "description": "Commodity + risk-on, but noisier",
        "master_pair": "AUD_JPY",
        "master_reason": "If you want yen exposure",
    },
}

# All unique pairs across all groups
ALL_PAIRS: List[str] = list(
    set(pair for group in CORRELATION_GROUPS.values() for pair in group["pairs"])
)

# Master pairs (best from each group for transfer learning)
MASTER_PAIRS: List[str] = [
    group["master_pair"] for group in CORRELATION_GROUPS.values()
]


# ═══════════════════════════════════════════════════════════════════════════
# Sweep Configuration for Multi-Pair
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MultiPairSweepConfig:
    """Configuration for multi-pair sweep.

    Attributes:
        pairs: List of trading pairs to sweep.
        sweep_type: Type of sweep configuration (tcn, lstm, ensemble, quick).
        granularity: Timeframe for data (H1, M5, etc.).
        project: W&B project name.
        count: Number of runs per pair.
        agents: Number of parallel agents.
    """

    pairs: List[str]
    sweep_type: str = "tcn"
    granularity: str = "H1"
    project: str = "ml_engine_fx"
    count: Optional[int] = None
    agents: int = 1

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.pairs:
            raise ValueError("pairs cannot be empty")
        if self.agents < 1:
            raise ValueError(f"agents must be >= 1, got {self.agents}")


def get_pair_sweep_config(
    pair: str,
    sweep_type: str = "tcn",
    granularity: str = "H1",
) -> Dict[str, Any]:
    """Get W&B sweep configuration for a specific pair.

    Args:
        pair: Trading pair (e.g., "EUR_USD").
        sweep_type: Type of sweep configuration.
        granularity: Timeframe for data.

    Returns:
        W&B sweep configuration dictionary.
    """
    from src.training.wandb_sweep_config import get_sweep_config

    # Get base sweep config
    config = get_sweep_config(sweep_type)
    spec = config.to_wandb_spec()

    # Add pair-specific fixed parameters
    spec["parameters"]["instrument"] = {"value": pair}
    spec["parameters"]["granularity"] = {"value": granularity}

    # Update name to include pair
    spec["name"] = f"{pair.lower()}_{sweep_type}_sweep"

    return spec


def create_pair_sweep(
    pair: str,
    sweep_type: str = "tcn",
    granularity: str = "H1",
    project: str = "ml_engine_fx",
    entity: Optional[str] = None,
) -> str:
    """Create a W&B sweep for a specific pair.

    Args:
        pair: Trading pair (e.g., "EUR_USD").
        sweep_type: Type of sweep configuration.
        granularity: Timeframe for data.
        project: W&B project name.
        entity: W&B entity.

    Returns:
        Sweep ID.
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb not installed. Install with: pip install wandb")

    spec = get_pair_sweep_config(pair, sweep_type, granularity)

    sweep_id = wandb.sweep(
        sweep=spec,
        entity=entity,
        project=project,
    )

    logger.info(f"Created sweep for {pair}: {sweep_id}")
    return sweep_id


def create_multi_pair_sweep(
    pairs: List[str],
    sweep_type: str = "tcn",
    granularity: str = "H1",
    project: str = "ml_engine_fx",
    entity: Optional[str] = None,
) -> Dict[str, str]:
    """Create W&B sweeps for multiple pairs.

    Args:
        pairs: List of trading pairs.
        sweep_type: Type of sweep configuration.
        granularity: Timeframe for data.
        project: W&B project name.
        entity: W&B entity.

    Returns:
        Dictionary mapping pair to sweep ID.
    """
    sweep_ids = {}

    for pair in pairs:
        try:
            sweep_id = create_pair_sweep(
                pair=pair,
                sweep_type=sweep_type,
                granularity=granularity,
                project=project,
                entity=entity,
            )
            sweep_ids[pair] = sweep_id
        except Exception as e:
            logger.error(f"Failed to create sweep for {pair}: {e}")

    return sweep_ids


# ═══════════════════════════════════════════════════════════════════════════
# Sweep Agent Runner
# ═══════════════════════════════════════════════════════════════════════════


def run_agent_for_pair(
    pair: str,
    sweep_id: str,
    count: Optional[int] = None,
    entity: Optional[str] = None,
    project: str = "ml_engine_fx",
) -> None:
    """Run a sweep agent for a specific pair.

    Args:
        pair: Trading pair (for logging).
        sweep_id: W&B sweep ID.
        count: Maximum number of runs.
        entity: W&B entity.
        project: W&B project name.
    """
    logger.info(f"Starting agent for {pair} (sweep: {sweep_id})")

    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed")
        return

    from src.training.wandb_sweep_agent import train_with_sweep_config

    def sweep_train():
        """Training function called by W&B sweep agent."""
        wandb.init()
        config = dict(wandb.config)

        try:
            # Override instrument in config
            config["instrument"] = pair
            metrics = train_with_sweep_config(config)
            wandb.log(metrics)
            logger.info(f"{pair} run completed. Accuracy: {metrics.get('val_direction_accuracy', 0):.4f}")
        except Exception as e:
            logger.error(f"{pair} training failed: {e}")
            wandb.log({"error": str(e)})
            raise
        finally:
            wandb.finish()

    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        entity=entity,
        project=project,
        count=count,
    )


def run_sweep_for_pairs(
    sweep_ids: Dict[str, str],
    count: Optional[int] = None,
    agents: int = 1,
    entity: Optional[str] = None,
    project: str = "ml_engine_fx",
    parallel: bool = False,
) -> None:
    """Run sweep agents for multiple pairs.

    Args:
        sweep_ids: Dictionary mapping pair to sweep ID.
        count: Maximum number of runs per pair.
        agents: Number of parallel agents.
        entity: W&B entity.
        project: W&B project name.
        parallel: Whether to run agents in parallel processes.
    """
    if parallel and agents > 1:
        # Run agents in parallel processes
        processes = []

        for i, (pair, sweep_id) in enumerate(sweep_ids.items()):
            # Distribute pairs across agents
            agent_id = i % agents

            p = multiprocessing.Process(
                target=run_agent_for_pair,
                args=(pair, sweep_id, count, entity, project),
                name=f"agent-{agent_id}-{pair}",
            )
            p.start()
            processes.append(p)

            # Stagger starts to avoid API rate limits
            time.sleep(2)

        # Wait for all processes to complete
        for p in processes:
            p.join()

    else:
        # Run agents sequentially
        for pair, sweep_id in sweep_ids.items():
            run_agent_for_pair(
                pair=pair,
                sweep_id=sweep_id,
                count=count,
                entity=entity,
                project=project,
            )


def run_multi_pair_sweep(
    config: MultiPairSweepConfig,
    entity: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, str]:
    """Run complete multi-pair sweep workflow.

    Args:
        config: Multi-pair sweep configuration.
        entity: W&B entity.
        dry_run: If True, only print what would be done.

    Returns:
        Dictionary mapping pair to sweep ID.
    """
    logger.info(f"Starting multi-pair sweep for {len(config.pairs)} pairs")
    logger.info(f"  Sweep type: {config.sweep_type}")
    logger.info(f"  Granularity: {config.granularity}")
    logger.info(f"  Runs per pair: {config.count or 'unlimited'}")
    logger.info(f"  Parallel agents: {config.agents}")

    if dry_run:
        logger.info("DRY RUN - no sweeps will be created")
        for pair in config.pairs:
            logger.info(f"  Would create sweep for: {pair}")
        return {}

    # Create sweeps
    sweep_ids = create_multi_pair_sweep(
        pairs=config.pairs,
        sweep_type=config.sweep_type,
        granularity=config.granularity,
        project=config.project,
        entity=entity,
    )

    # Print sweep info
    print("\n" + "=" * 60)
    print("Created Sweeps")
    print("=" * 60)
    for pair, sweep_id in sweep_ids.items():
        print(f"  {pair}: {sweep_id}")
    print("=" * 60)

    # Print agent commands
    print("\nTo run agents, use:")
    for pair, sweep_id in sweep_ids.items():
        print(f"  wandb agent {sweep_id}")

    return sweep_ids


# ═══════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════


def main():
    """CLI entry point for multi-pair sweep operations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Pair W&B Sweep Runner for FX Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Correlation Groups:
  euro_basket    EUR_USD, EUR_JPY, EUR_GBP, EUR_AUD
  yen_carry      EUR_JPY, GBP_JPY, AUD_JPY, NZD_JPY, USD_JPY
  sterling       GBP_USD, GBP_JPY, GBP_AUD, EUR_GBP
  aussie_kiwi    AUD_USD, NZD_USD, AUD_JPY, NZD_JPY
  usd_safe       USD_JPY, USD_CHF, USD_CAD
  crosses        GBP_AUD, EUR_AUD, AUD_NZD

Examples:
  # Run sweep on Euro Basket group
  python -m src.training.wandb_multi_pair_sweep --group euro_basket

  # Run sweep on specific pairs
  python -m src.training.wandb_multi_pair_sweep --pairs EUR_USD,GBP_USD,USD_JPY

  # Run master pairs only (best from each group)
  python -m src.training.wandb_multi_pair_sweep --master-pairs

  # Run all pairs
  python -m src.training.wandb_multi_pair_sweep --all

  # Dry run to see what would happen
  python -m src.training.wandb_multi_pair_sweep --group euro_basket --dry-run
        """,
    )

    # Pair selection (mutually exclusive)
    pair_group = parser.add_mutually_exclusive_group(required=True)
    pair_group.add_argument(
        "--group",
        choices=list(CORRELATION_GROUPS.keys()),
        help="Run sweep on a correlation group",
    )
    pair_group.add_argument(
        "--pairs",
        help="Comma-separated list of pairs (e.g., EUR_USD,GBP_USD)",
    )
    pair_group.add_argument(
        "--master-pairs",
        action="store_true",
        help="Run sweep on master pairs only (best from each group)",
    )
    pair_group.add_argument(
        "--all",
        action="store_true",
        help="Run sweep on all pairs",
    )

    # Sweep configuration
    parser.add_argument(
        "--type",
        choices=["tcn", "lstm", "ensemble", "quick"],
        default="tcn",
        help="Sweep configuration type (default: tcn)",
    )
    parser.add_argument(
        "--granularity",
        default="H1",
        help="Timeframe for data (default: H1)",
    )
    parser.add_argument(
        "--project",
        default="ml_engine_fx",
        help="W&B project name (default: ml_engine_fx)",
    )
    parser.add_argument(
        "--entity",
        help="W&B entity (username or team)",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Maximum number of runs per pair",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=1,
        help="Number of parallel agents (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without creating sweeps",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Determine pairs to sweep
    pairs: List[str] = []

    if args.group:
        pairs = CORRELATION_GROUPS[args.group]["pairs"]
        group_name = CORRELATION_GROUPS[args.group]["name"]
        print(f"\n📊 Running sweep on {group_name} group")
        print(f"   Pairs: {', '.join(pairs)}")

    elif args.pairs:
        pairs = [p.strip().upper() for p in args.pairs.split(",")]
        # Normalize pair format (EUR-USD -> EUR_USD)
        pairs = [p.replace("-", "_") for p in pairs]
        print(f"\n📊 Running sweep on {len(pairs)} pairs")
        print(f"   Pairs: {', '.join(pairs)}")

    elif args.master_pairs:
        pairs = MASTER_PAIRS
        print("\n📊 Running sweep on master pairs")
        print(f"   Pairs: {', '.join(pairs)}")

    elif args.all:
        pairs = ALL_PAIRS
        print(f"\n📊 Running sweep on ALL pairs ({len(pairs)} total)")
        print(f"   Pairs: {', '.join(pairs)}")

    # Create configuration
    config = MultiPairSweepConfig(
        pairs=pairs,
        sweep_type=args.type,
        granularity=args.granularity,
        project=args.project,
        count=args.count,
        agents=args.agents,
    )

    # Run sweep
    sweep_ids = run_multi_pair_sweep(
        config=config,
        entity=args.entity,
        dry_run=args.dry_run,
    )

    if sweep_ids and not args.dry_run:
        print("\n✅ Sweeps created successfully!")
        print("\nNext steps:")
        print("1. Run agents for each sweep:")
        for pair, sweep_id in sweep_ids.items():
            print(f"   wandb agent {sweep_id}")
        print("\n2. Or run multiple agents in parallel:")
        print("   # In separate terminals:")
        for i, (pair, sweep_id) in enumerate(sweep_ids.items()):
            print(f"   Terminal {i+1}: wandb agent {sweep_id}")


if __name__ == "__main__":
    main()
