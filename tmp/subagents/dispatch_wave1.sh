#!/bin/bash
# Wave 1 Subagent Dispatcher - Orchestrator harness

ROOT=/Users/buddy/Documents/ml_engine
LOGDIR=$ROOT/tmp/subagents/logs
mkdir -p $LOGDIR

echo "=== Wave 1 Subagent Dispatch ==="
echo "Timestamp: $(date)"
echo ""

# Subagent 1: Data Harvest
echo "[DISPATCH] Harvest subagent..."
cd $ROOT
cat $ROOT/tmp/subagents/harvest.prompt | codex exec -- > $LOGDIR/harvest.log 2>&1 &
HARVEST_PID=$!
echo "  PID=$HARVEST_PID"

# Subagent 2: SOTA Sanity
echo "[DISPATCH] Sanity test subagent..."
cd $ROOT
cat $ROOT/tmp/subagents/sanity.prompt | codex exec -- > $LOGDIR/sanity.log 2>&1 &
SANITY_PID=$!
echo "  PID=$SANITY_PID"

# Subagent 3: Neural Bootstrap
echo "[DISPATCH] Neural bootstrap subagent..."
cd $ROOT
cat $ROOT/tmp/subagents/neural_bootstrap.prompt | codex exec -- > $LOGDIR/neural.log 2>&1 &
NEURAL_PID=$!
echo "  PID=$NEURAL_PID"

# Save PIDs
echo "$HARVEST_PID $SANITY_PID $NEURAL_PID" > $LOGDIR/pids.txt
echo ""
echo "All subagents dispatched. PIDs saved to $LOGDIR/pids.txt"
echo "Monitor with: tail -f $LOGDIR/*.log"
