#!/bin/bash
# Comprehensive ACBO Training Script (~2 hours)
# Trains BC surrogate, BC policy, GRPO with surrogate, and GRPO without surrogate

set -e  # Exit on error

echo "=============================================="
echo "COMPREHENSIVE ACBO TRAINING (~2 HOURS)"
echo "=============================================="
echo ""
echo "This will train:"
echo "  1. BC Surrogate (structure learning)"
echo "  2. BC Policy (behavioral cloning)"
echo "  3. GRPO WITH pre-trained surrogate"
echo "  4. GRPO WITHOUT surrogate (baseline)"
echo ""
echo "Estimated time: ~2 hours"
echo ""

# Create checkpoint directory
CHECKPOINT_DIR="checkpoints/comprehensive_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECKPOINT_DIR"

echo "Saving checkpoints to: $CHECKPOINT_DIR"
echo ""

# Log start time
START_TIME=$(date +%s)
echo "Start time: $(date)"
echo ""

# 1. Train BC Surrogate for structure learning
echo "=============================================="
echo "1/4: Training BC Surrogate (500 episodes, ~30 min)"
echo "=============================================="
poetry run python scripts/train_acbo_methods.py \
    --method surrogate \
    --episodes 500 \
    --batch_size 128 \
    --learning_rate 1e-3 \
    --surrogate_hidden_dim 256 \
    --surrogate_layers 6 \
    --surrogate_heads 8 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --max_demos 100 \
    --seed 42

echo ""
echo "✓ BC Surrogate training complete!"
echo ""

# 2. Train BC Policy for intervention selection
echo "=============================================="
echo "2/4: Training BC Policy (500 episodes, ~30 min)"
echo "=============================================="
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --episodes 500 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --hidden_dim 512 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --max_demos 100 \
    --seed 43

echo ""
echo "✓ BC Policy training complete!"
echo ""

# 3. Train GRPO WITH pre-trained surrogate
echo "=============================================="
echo "3/4: Training GRPO WITH Surrogate (1000 episodes, ~45 min)"
echo "=============================================="
poetry run python scripts/train_acbo_methods.py \
    --method grpo_with_surrogate \
    --episodes 1000 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --surrogate_lr 5e-4 \
    --scm_type mixed \
    --min_vars 3 \
    --max_vars 10 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --max_demos 100 \
    --seed 44

echo ""
echo "✓ GRPO with surrogate training complete!"
echo ""

# 4. Train GRPO WITHOUT surrogate (baseline)
echo "=============================================="
echo "4/4: Training GRPO WITHOUT Surrogate (1000 episodes, ~30 min)"
echo "=============================================="
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 1000 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --hidden_dim 512 \
    --scm_type mixed \
    --min_vars 3 \
    --max_vars 10 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --seed 45

echo ""
echo "✓ GRPO without surrogate training complete!"
echo ""

# Calculate training duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

# Create a summary file
echo "=============================================="
echo "Creating training summary..."
echo "=============================================="

cat > "$CHECKPOINT_DIR/training_summary.txt" << EOF
Comprehensive ACBO Training Summary
==================================
Date: $(date)
Total training time: ${HOURS}h ${MINUTES}m
Directory: $CHECKPOINT_DIR

Models Trained:
1. BC Surrogate (500 episodes)
   - Location: $CHECKPOINT_DIR/bc_surrogate_final/
   - Architecture: 256 hidden dim, 6 layers, 8 heads
   - Purpose: Structure learning / parent prediction

2. BC Policy (500 episodes) 
   - Location: $CHECKPOINT_DIR/bc_final/
   - Architecture: 512 hidden dim
   - Purpose: Behavioral cloning for intervention selection

3. GRPO WITH Surrogate (1000 episodes)
   - Location: $CHECKPOINT_DIR/unified_grpo_final/
   - Architecture: 512 hidden dim
   - Purpose: RL-based intervention with structure learning

4. GRPO WITHOUT Surrogate (1000 episodes)
   - Location: $CHECKPOINT_DIR/grpo_no_surrogate_final/
   - Architecture: 512 hidden dim
   - Purpose: RL-based intervention baseline (no structure)

Training Parameters:
- Demo path: expert_demonstrations/raw/raw_demonstrations
- Max demos used: 100
- Variable range: 3-10
- Seeds: 42 (BC surrogate), 43 (BC policy), 44 (GRPO+surrogate), 45 (GRPO-surrogate)

Next Steps:
1. Use evaluate_comprehensive.sh to evaluate all combinations
2. Compare GRPO with vs without surrogate
3. Compare with/without active learning
4. Analyze trajectory plots and learning curves
EOF

echo ""
echo "Training summary saved to: $CHECKPOINT_DIR/training_summary.txt"
echo ""

# Create comprehensive evaluation script
cat > "$CHECKPOINT_DIR/evaluate_all.sh" << 'EOF'
#!/bin/bash
# Comprehensive evaluation of all trained models

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "======================================================="
echo "COMPREHENSIVE EVALUATION OF TRAINED MODELS"
echo "======================================================="
echo ""
echo "Models directory: $SCRIPT_DIR"
echo ""

# 1. Evaluate WITHOUT active learning
echo "1. Evaluating WITHOUT active learning..."
echo "-------------------------------------------------------"
poetry run python scripts/evaluate_acbo_methods.py \
    --register_surrogate bc "$SCRIPT_DIR/bc_surrogate_final/checkpoint.pkl" \
    --register_policy grpo_with_surrogate "$SCRIPT_DIR/unified_grpo_final/checkpoint.pkl" \
    --register_policy grpo_no_surrogate "$SCRIPT_DIR/grpo_no_surrogate_final/checkpoint.pkl" \
    --register_policy bc "$SCRIPT_DIR/bc_final/checkpoint.pkl" \
    --evaluate_pairs grpo_with_surrogate bc \
    --evaluate_pairs grpo_no_surrogate bc \
    --evaluate_pairs bc bc \
    --evaluate_pairs random bc \
    --include_baselines --baseline_surrogate bc \
    --surrogate_update_strategy none \
    --n_scms 10 \
    --n_interventions 30 \
    --plot --plot_trajectories \
    --output_dir "$SCRIPT_DIR/eval_no_active_learning"

echo ""
echo "2. Evaluating WITH active learning (BIC)..."
echo "-------------------------------------------------------"
poetry run python scripts/evaluate_acbo_methods.py \
    --register_surrogate bc "$SCRIPT_DIR/bc_surrogate_final/checkpoint.pkl" \
    --register_policy grpo_with_surrogate "$SCRIPT_DIR/unified_grpo_final/checkpoint.pkl" \
    --register_policy grpo_no_surrogate "$SCRIPT_DIR/grpo_no_surrogate_final/checkpoint.pkl" \
    --register_policy bc "$SCRIPT_DIR/bc_final/checkpoint.pkl" \
    --evaluate_pairs grpo_with_surrogate bc \
    --evaluate_pairs grpo_no_surrogate bc \
    --evaluate_pairs bc bc \
    --evaluate_pairs random bc \
    --include_baselines --baseline_surrogate bc \
    --surrogate_update_strategy bic \
    --n_scms 10 \
    --n_interventions 30 \
    --plot --plot_trajectories \
    --output_dir "$SCRIPT_DIR/eval_with_active_learning"

echo ""
echo "======================================================="
echo "EVALUATION COMPLETE!"
echo "======================================================="
echo ""
echo "Results saved to:"
echo "  - $SCRIPT_DIR/eval_no_active_learning/"
echo "  - $SCRIPT_DIR/eval_with_active_learning/"
echo ""
echo "Key comparisons to look for:"
echo "  1. GRPO with surrogate vs GRPO without surrogate"
echo "  2. Effect of active learning on performance"
echo "  3. BC policy vs GRPO policies"
echo "  4. Trajectory plots showing learning dynamics"
echo ""
EOF

chmod +x "$CHECKPOINT_DIR/evaluate_all.sh"

echo "=============================================="
echo "TRAINING COMPLETE!"
echo "=============================================="
echo ""
echo "Training duration: ${HOURS}h ${MINUTES}m"
echo ""
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "Available models:"
echo "  1. BC Surrogate: $CHECKPOINT_DIR/bc_surrogate_final/"
echo "  2. BC Policy: $CHECKPOINT_DIR/bc_final/"
echo "  3. GRPO WITH Surrogate: $CHECKPOINT_DIR/unified_grpo_final/"
echo "  4. GRPO WITHOUT Surrogate: $CHECKPOINT_DIR/grpo_no_surrogate_final/"
echo ""
echo "To run comprehensive evaluation:"
echo "  $CHECKPOINT_DIR/evaluate_all.sh"
echo ""
echo "=============================================="