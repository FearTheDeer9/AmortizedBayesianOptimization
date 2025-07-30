# ACBO Training Commands

## Individual Training Commands

### 1. GRPO Training
Train GRPO policy (Note: --use_surrogate currently only initializes a surrogate, doesn't train it):
```bash
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 2000 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --scm_type mixed \
    --checkpoint_dir checkpoints/grpo_long \
    --seed 42
```

**With surrogate (experimental - not fully integrated):**
```bash
poetry run python scripts/train_acbo_methods.py \
    --method grpo \
    --episodes 2000 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --use_surrogate \
    --surrogate_lr 1e-3 \
    --surrogate_layers 4 \
    --surrogate_hidden_dim 128 \
    --scm_type mixed \
    --checkpoint_dir checkpoints/grpo_with_surrogate \
    --seed 42
```

### 2. BC Policy Training
Train policy from expert demonstrations:
```bash
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --model_type policy \
    --episodes 1500 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --hidden_dim 256 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir checkpoints/bc_policy_long \
    --seed 43
```

### 3. BC Surrogate Training
Train surrogate from demonstrations:
```bash
poetry run python scripts/train_acbo_methods.py \
    --method bc \
    --model_type surrogate \
    --episodes 1500 \
    --batch_size 32 \
    --surrogate_lr 1e-3 \
    --surrogate_layers 4 \
    --surrogate_hidden_dim 128 \
    --demo_path expert_demonstrations/raw/raw_demonstrations \
    --checkpoint_dir checkpoints/bc_surrogate_long \
    --seed 44
```

## Evaluation Commands

### Standard Evaluation
Evaluate trained policies with active learning surrogates:
```bash
poetry run python scripts/evaluate_acbo_methods.py \
    --grpo checkpoints/grpo_long/checkpoint_final.pkl \
    --bc checkpoints/bc_policy_long/checkpoint_final.pkl \
    --n_scms 20 \
    --n_interventions 30 \
    --n_obs 100 \
    --n_samples 20 \
    --use_active_learning \
    --plot \
    --output_dir evaluation_results/comprehensive
```

### With Pre-trained Surrogate
Use a pre-trained BC surrogate (Note: not fully integrated with GRPO rewards):
```bash
poetry run python scripts/evaluate_acbo_methods.py \
    --grpo checkpoints/grpo_long/checkpoint_final.pkl \
    --bc checkpoints/bc_policy_long/checkpoint_final.pkl \
    --surrogate_checkpoint checkpoints/bc_surrogate_long/checkpoint_final.pkl \
    --n_scms 20 \
    --n_interventions 30 \
    --use_active_learning \
    --plot \
    --output_dir evaluation_results/with_surrogate
```

## Training Tips

1. **Episodes**: 
   - Quick test: 100-500 episodes
   - Normal training: 1000-2000 episodes  
   - Thorough training: 3000-5000 episodes

2. **Batch Size**:
   - Larger batches (64-128) are more stable but slower
   - Smaller batches (16-32) train faster but may be noisier

3. **Learning Rates**:
   - GRPO policy: 1e-4 to 3e-4
   - Surrogates: 5e-4 to 1e-3
   - BC: 5e-5 to 1e-4

4. **Model Sizes**:
   - Policy hidden_dim: 256-512
   - Surrogate hidden_dim: 128-256
   - Surrogate layers: 4-6

## Quick Test Commands

For quick testing (5-10 minutes):
```bash
# Quick GRPO
poetry run python scripts/train_acbo_methods.py --method grpo --episodes 200

# Quick BC
poetry run python scripts/train_acbo_methods.py --method bc --episodes 200 --max_demos 10

# Quick evaluation
poetry run python scripts/evaluate_acbo_methods.py \
    --grpo checkpoints/grpo/checkpoint_final.pkl \
    --bc checkpoints/bc/checkpoint_final.pkl \
    --n_scms 4 --n_interventions 10
```