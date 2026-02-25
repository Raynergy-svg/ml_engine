#!/bin/bash
# Run all W&B sweep agents in parallel
# Each agent runs in a separate terminal tab (macOS)

# Set your W&B entity and project
WANDB_ENTITY=${WANDB_ENTITY:-"tencylinder8310-smartdebtflow-com"}
WANDB_PROJECT=${WANDB_PROJECT:-"ml_engine_fx"}

# Sweep IDs from the created sweeps (short IDs)
SWEEP_IDS=(
    "ldjkqfhk"
    "f3t6673s"
    "m738sn9x"
    "eksbr3ub"
    "zn7e8fop"
    "rivyiff6"
    "q4ufao2c"
    "jcj2zyg6"
    "syb43ybn"
    "hjm5qoz5"
    "dh0s9idy"
    "t92nqb3a"
    "18qqka9a"
    "kgl86rn8"
    "xrt0eu0l"
)

# Number of agents to run in parallel (adjust based on your machine)
# M1 Macs can handle 4-6 efficiently
MAX_PARALLEL=${MAX_PARALLEL:-4}

echo "🚀 Starting W&B Sweep Agents"
echo "   Total sweeps: ${#SWEEP_IDS[@]}"
echo "   Parallel agents: $MAX_PARALLEL"
echo ""

# Function to run agent in background
run_agent() {
    local sweep_id=$1
    local full_sweep_id="${WANDB_ENTITY}/${WANDB_PROJECT}/${sweep_id}"
    echo "Starting agent for sweep: $full_sweep_id"
    wandb agent "$full_sweep_id" 2>&1 | while read -r line; do
        echo "[$sweep_id] $line"
    done
}

# Run agents with limited parallelism
count=0
for sweep_id in "${SWEEP_IDS[@]}"; do
    run_agent "$sweep_id" &
    ((count++))
    
    # Wait when we hit max parallel
    if ((count >= MAX_PARALLEL)); then
        wait -n  # Wait for any one job to finish
        ((count--))
    fi
done

# Wait for all remaining agents
wait

echo ""
echo "✅ All sweep agents completed"
