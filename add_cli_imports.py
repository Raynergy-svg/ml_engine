#!/usr/bin/env python3
"""Script to add CLI imports to main.py."""

with open('main.py', 'r') as f:
    content = f.read()

old_pattern = """# Paper trade planning (extracted to src/trading/paper_trade.py)
from src.trading.paper_trade import (
    PaperTradePlan as _FxPaperTradePlan,
    PaperTradeContext,
    setup_paper_trade_context as _fx_setup_paper_trade,
    build_paper_trade_plan as _fx_build_paper_trade_plan,
    execute_paper_trade_plan as _fx_execute_paper_trade_plan,
)



# Constants"""

new_block = """# Paper trade planning (extracted to src/trading/paper_trade.py)
from src.trading.paper_trade import (
    PaperTradePlan as _FxPaperTradePlan,
    PaperTradeContext,
    setup_paper_trade_context as _fx_setup_paper_trade,
    build_paper_trade_plan as _fx_build_paper_trade_plan,
    execute_paper_trade_plan as _fx_execute_paper_trade_plan,
)

# ============================================================================
# CLI package imports (Phase 4-5 decomposition - replaces inline definitions)
# ============================================================================
from cli import (
    # Training
    train_buddy as _cli_train_buddy,
    # Inference
    buddy as _cli_buddy,
    buddy_loop as _cli_buddy_loop,
    buddy_scan as _cli_buddy_scan,
    # FX Trading
    fx_paper_trade as _cli_fx_paper_trade,
    generate_dashboard as _cli_generate_dashboard,
    # Model management
    model_status as _cli_model_status,
    suggest_improvements as _cli_suggest_improvements,
    retrain_gates as _cli_retrain_gates,
    train_rl_sizer as _cli_train_rl_sizer,
    # Monitoring/Analytics
    buddy_monitor as _cli_buddy_monitor,
    buddy_journal as _cli_buddy_journal,
    buddy_analyze as _cli_buddy_analyze,
    # Testing/Validation
    buddy_validate as _cli_buddy_validate,
    buddy_test as _cli_buddy_test,
    # Dispatch functions
    _dispatch_buddy,
    _dispatch_train_buddy,
    _normalize_command_args,
    _maybe_run_buddy_interactive_wizard,
    _maybe_launch_buddy_repl,
)
from cli.model_management import promote_model as _cli_promote_model


# Constants"""

if old_pattern in content:
    content = content.replace(old_pattern, new_block)
    with open('main.py', 'w') as f:
        f.write(content)
    print("CLI imports added successfully")
else:
    print("Pattern not found")
