# Buddy Planner Dataset Schema

The planner model is trained to emit:
- a plan
- tool calls
- a grounded final answer from tool results

It is explicitly **not** trained to invent trading facts from weights alone.

## Format

One JSON object per line in `train.jsonl` and `valid.jsonl`.

```json
{
  "id": "planner_000001",
  "split": "train",
  "user_message": "scan h1 and auto-execute top 2 if approved",
  "chat_history": [
    {"role": "user", "content": "switch to eur/usd"},
    {"role": "assistant", "content": "loaded oanda:EUR_USD:M5:300."}
  ],
  "runtime_context": {
    "active_instrument": "EUR_USD",
    "granularity": "M5",
    "execute_mode": "dry-run",
    "knowledge_only": false
  },
  "available_tools": [
    {
      "name": "scan_market",
      "description": "Run a multi-pair scan with optional auto-execution.",
      "args_schema": {
        "pairs": {"type": "string|null", "required": false},
        "granularity": {"type": "string", "required": false},
        "top_n": {"type": "integer", "required": false},
        "auto_execute": {"type": "boolean", "required": false},
        "force": {"type": "boolean", "required": false},
        "diversified": {"type": "boolean", "required": false}
      }
    }
  ],
  "target_plan": {
    "needs_clarification": false,
    "clarification_question": null,
    "final_answer": null,
    "steps": [
      {
        "tool": "scan_market",
        "args": {
          "pairs": null,
          "granularity": "H1",
          "top_n": 2,
          "auto_execute": true,
          "force": false,
          "diversified": false
        }
      }
    ]
  },
  "tool_observations": [
    {
      "tool": "scan_market",
      "args": {
        "pairs": null,
        "granularity": "H1",
        "top_n": 2,
        "auto_execute": true,
        "force": false,
        "diversified": false
      },
      "result": {
        "count": 2,
        "tradable_count": 1,
        "approved_count": 1,
        "rows": [
          {
            "pair": "EUR_USD",
            "direction": "LONG",
            "gates_passed": true,
            "confidence": 0.81,
            "agent_total": 2.0,
            "error": null
          }
        ]
      }
    }
  ],
  "target_response": "I scanned H1 and found 1 approved setup in the top 2. EUR/USD is the strongest current setup and auto-execution was enabled for approved trades.",
  "safety_label": "tool_grounded",
  "tags": ["scan", "auto_execute", "multi_step"]
}
```

## Required fields

- `id`
- `split`
- `user_message`
- `chat_history`
- `runtime_context`
- `available_tools`
- `target_plan`
- `tool_observations`
- `target_response`
- `safety_label`
- `tags`

## Labels

- `tool_grounded`: answer comes strictly from tool outputs
- `clarification`: planner must ask for missing details
- `safe_refusal`: planner must refuse unsupported or unsafe requests

## Generation rules

- Never write target responses that introduce facts absent from `tool_observations`.
- Keep `target_plan.steps` minimal and ordered.
- Use `final_answer` only when no tools are needed.
- Keep clarification questions short and specific.
