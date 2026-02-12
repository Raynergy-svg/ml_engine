"""RL worker processes for macOS-safe SB3/PyTorch usage.

These workers are intentionally isolated from TensorFlow-heavy modules.
They load stable-baselines3 models in a separate process and serve requests
via multiprocessing queues.
"""

from __future__ import annotations

from typing import Any


def rl_gates_worker_main(request_queue: Any, response_queue: Any) -> None:
    """Child process: load RL gates model and serve threshold requests."""
    try:
        from src.rl.gate_threshold_env import GateThresholdRL

        optimizer = GateThresholdRL()
        loaded = optimizer.load()
        response_queue.put({'type': 'status', 'ok': bool(loaded)})
        if not loaded:
            return

        while True:
            msg = request_queue.get()
            if not isinstance(msg, dict):
                continue
            if msg.get('type') == 'shutdown':
                return
            if msg.get('type') != 'thresholds':
                continue

            req_id = msg.get('id')
            features = msg.get('features') or {}
            win_rate = float(msg.get('win_rate', 0.5))
            drawdown = float(msg.get('drawdown', 0.0))
            adjusted = optimizer.get_adjusted_thresholds(
                features=features,
                win_rate=win_rate,
                drawdown=drawdown,
            )
            response_queue.put({'type': 'thresholds', 'id': req_id, 'ok': True, 'result': adjusted})
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        try:
            response_queue.put({'type': 'status', 'ok': False, 'error': str(e)})
        except Exception:
            pass


def rl_exits_worker_main(request_queue: Any, response_queue: Any) -> None:
    """Child process: load RL exits model and serve exit-decision requests."""
    try:
        from src.rl.optimal_exit_env import OptimalExitRL

        optimizer = OptimalExitRL()
        loaded = optimizer.load()
        response_queue.put({'type': 'status', 'ok': bool(loaded)})
        if not loaded:
            return

        while True:
            msg = request_queue.get()
            if not isinstance(msg, dict):
                continue
            if msg.get('type') == 'shutdown':
                return
            if msg.get('type') != 'exit_decision':
                continue

            req_id = msg.get('id')
            action, confidence = optimizer.get_exit_decision(
                unrealized_pnl_pips=float(msg.get('unrealized_pnl_pips', 0.0)),
                bars_in_trade=int(msg.get('bars_in_trade', 0)),
                momentum=float(msg.get('momentum', 0.0)),
                atr=float(msg.get('atr', 0.001)),
            )
            response_queue.put(
                {
                    'type': 'exit_decision',
                    'id': req_id,
                    'ok': True,
                    'result': {'action': int(action), 'confidence': float(confidence)},
                }
            )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        try:
            response_queue.put({'type': 'status', 'ok': False, 'error': str(e)})
        except Exception:
            pass
