# Existing Site tool declarations (read; no writes invoked)

## read_native_read_status

Private read-only preview of the reviewed Axiom 2.0 dashboard. No live observations or financial actions.

Read current freshness-qualified Axiom 2.0 native read status. Does not contact Robinhood or authenticate a native session.

exec tool declaration:
```ts
declare const tools: { mcp__codex_apps__axiom_2_0___interface_preview__axiom_2_0_interface_preview_read_native_read_status(args: {}): Promise<CallToolResult>; };
```

## publish_native_read_status

Private read-only preview of the reviewed Axiom 2.0 dashboard. No live observations or financial actions.

Publish a signed sanitized Axiom 2.0 native-read status for the explicitly enrolled owner/session. Disabled until native access is approved and configured. Never accepts research heartbeats or account contents.

exec tool declaration:
```ts
declare const tools: { mcp__codex_apps__axiom_2_0___interface_preview__axiom_2_0_interface_preview_publish_native_read_status(args: { payload: { [key: string]: unknown; }; signature: string; }): Promise<CallToolResult>; };
```

## read_market_data

Private read-only preview of the reviewed Axiom 2.0 dashboard. No live observations or financial actions.

Read signed native market quotes for Axiom’s enrolled symbols. Feed provenance and exchange coverage are explicit. Source age and delivery age are recomputed at read time. Default requires every quote field within 1000ms; stale data returns an explicit error. This reads the latest delivered batch, does not contact the broker or authorize trading.

exec tool declaration:
```ts
declare const tools: { mcp__codex_apps__axiom_2_0___interface_preview__axiom_2_0_interface_preview_read_market_data(args: { max_age_ms?: number; require_fresh?: boolean; }): Promise<CallToolResult>; };
```

## axiom_runtime_status

Private read-only preview of the reviewed Axiom 2.0 dashboard. No live observations or financial actions.

Read Axiom analytical worker capabilities, admitted dataset IDs and completed job count. Broker ingestion is caller-delivered; this is not a continuous market stream or a trained model.

exec tool declaration:
```ts
declare const tools: { mcp__codex_apps__axiom_2_0___interface_preview__axiom_2_0_interface_preview_axiom_runtime_status(args: {}): Promise<CallToolResult>; };
```

## axiom_research_status

Private read-only preview of the reviewed Axiom 2.0 dashboard. No live observations or financial actions.

Read recent comparative research jobs and the native Python worker heartbeat. No broker or order actions.

exec tool declaration:
```ts
declare const tools: { mcp__codex_apps__axiom_2_0___interface_preview__axiom_2_0_interface_preview_axiom_research_status(args: {}): Promise<CallToolResult>; };
```

## axiom_research_health

Private read-only preview of the reviewed Axiom 2.0 dashboard. No live observations or financial actions.

Read collector health separately from source freshness. An absent collector is unavailable.

exec tool declaration:
```ts
declare const tools: { mcp__codex_apps__axiom_2_0___interface_preview__axiom_2_0_interface_preview_axiom_research_health(args: {}): Promise<CallToolResult>; };
```

## axiom_publish_research_health

Private read-only preview of the reviewed Axiom 2.0 dashboard. No live observations or financial actions.

Versioned read-only research evidence operation. Explicit provider basis, immutable inputs and results; no trading or capital authority.

exec tool declaration:
```ts
declare const tools: { mcp__codex_apps__axiom_2_0___interface_preview__axiom_2_0_interface_preview_axiom_publish_research_health(args: { delivery: { [key: string]: unknown; }; }): Promise<CallToolResult>; };
```


