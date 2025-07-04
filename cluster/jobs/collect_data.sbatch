#!/bin/bash
#SBATCH --job-name=acbo_collect_data
#SBATCH --cpus-per-task=8            # 8 CPU cores for parallel collection
#SBATCH --mem=32G                    # 32GB RAM for data processing
#SBATCH --time=24:00:00              # 24 hour time limit
#SBATCH --output=/vol/bitbucket/%u/causal_bayes_opt/logs/collection/collect_data_%j.out
#SBATCH --error=/vol/bitbucket/%u/causal_bayes_opt/logs/collection/collect_data_%j.err

# Job information
echo "=================================================="
echo "ACBO Data Collection Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=================================================="

# Configuration - Edit these parameters as needed
DATASET_SIZE=${DATASET_SIZE:-"medium"}      # small/medium/large/xlarge
DIFFICULTY_LEVELS=${DIFFICULTY_LEVELS:-"all"}  # difficulty_1 difficulty_2 ... or all
OUTPUT_DIR=${OUTPUT_DIR:-"/vol/bitbucket/$USER/causal_bayes_opt/data/raw"}
WORKERS=${WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-100}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-500}  # Checkpoint every N demonstrations
MIN_ACCURACY=${MIN_ACCURACY:-0.7}
SEED=${SEED:-$SLURM_JOB_ID}  # Use job ID as seed if not provided
TRAJECTORY_LENGTH=${TRAJECTORY_LENGTH:-5}  # Default to 5 CBO iterations

# Setup environment
PROJECT_DIR="/vol/bitbucket/$USER/causal_bayes_opt"
cd "$PROJECT_DIR"

# Activate environment
echo "Activating environment..."
source activate_env.sh

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start data collection
echo "Starting data collection with configuration:"
echo "  Dataset size: $DATASET_SIZE"
echo "  Difficulty levels: $DIFFICULTY_LEVELS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Processing: Serial mode (avoiding pickle issues)"
echo "  Batch size: $BATCH_SIZE"
echo "  Checkpoint interval: $CHECKPOINT_INTERVAL"
echo "  Min accuracy: $MIN_ACCURACY"
echo "  Seed: $SEED"
echo "  Trajectory length: $TRAJECTORY_LENGTH"
echo ""

# Determine if we should use Poetry
if [ -f "${PROJECT_DIR}/pyproject.toml" ] && command -v poetry &> /dev/null; then
    echo "Using Poetry for execution"
    CMD="poetry run python scripts/core/collect_sft_dataset.py"
else
    echo "Using direct Python execution"
    CMD="python scripts/core/collect_sft_dataset.py"
fi

# Build command arguments
CMD="$CMD --size $DATASET_SIZE"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --serial"  # Use serial mode to avoid pickle issues
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --checkpoint-interval $CHECKPOINT_INTERVAL"
CMD="$CMD --min-accuracy $MIN_ACCURACY"
CMD="$CMD --seed $SEED"
CMD="$CMD --trajectory-length $TRAJECTORY_LENGTH"

# Add difficulty levels if not "all"
if [ "$DIFFICULTY_LEVELS" != "all" ]; then
    CMD="$CMD --difficulty $DIFFICULTY_LEVELS"
    CMD="$CMD --progressive"
fi

# Check for resume file
RESUME_FILE="$OUTPUT_DIR/checkpoints/latest_checkpoint.pkl"
if [ -f "$RESUME_FILE" ]; then
    echo "Found checkpoint file, resuming collection..."
    CMD="$CMD --resume $RESUME_FILE"
fi

echo "Executing: $CMD"
echo "=================================================="

# Execute collection
eval $CMD
COLLECTION_EXIT_CODE=$?

# Validate collected data if collection succeeded
if [ $COLLECTION_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Data collection completed successfully!"
    echo "Running validation..."
    
    # Use Poetry for validation if available
    if [ -f "${PROJECT_DIR}/pyproject.toml" ] && command -v poetry &> /dev/null; then
        poetry run python scripts/core/validate_dataset.py "$OUTPUT_DIR" --detailed --export-report --visualizations
    else
        python scripts/core/validate_dataset.py "$OUTPUT_DIR" --detailed --export-report --visualizations
    fi
    VALIDATION_EXIT_CODE=$?
    
    if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
        echo "✅ Data collection and validation completed successfully!"
    else
        echo "⚠️ Data collection succeeded but validation failed (exit code: $VALIDATION_EXIT_CODE)"
    fi
else
    echo "❌ Data collection failed with exit code: $COLLECTION_EXIT_CODE"
fi

# Print job statistics
echo ""
echo "=================================================="
echo "Job Statistics:"
echo "End time: $(date)"
echo "Duration: $SECONDS seconds"
echo "Exit code: $COLLECTION_EXIT_CODE"

# Check system resources
echo ""
echo "Final system status:"
echo "Memory usage: $(free -h | head -2)"
echo "CPU load: $(uptime)"

# Clean up temporary files if collection was successful
if [ $COLLECTION_EXIT_CODE -eq 0 ]; then
    echo "Cleaning up temporary files..."
    find "$OUTPUT_DIR" -name "*.tmp" -delete 2>/dev/null || true
fi

exit $COLLECTION_EXIT_CODE