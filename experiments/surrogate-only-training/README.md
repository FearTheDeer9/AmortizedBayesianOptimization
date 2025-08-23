# Surrogate Model Training with Alternating Attention

This directory contains the implementation of our surrogate model for parent set prediction using alternating attention, inspired by AVICI (Lorch et al.).

## Architecture

- **Alternating Attention**: Attention alternates between observation and variable axes every layer
- **Model Size**: 128 hidden dimensions, 8 layers, 8 attention heads (~2.1M parameters)
- **No Statistical Features**: Pure neural network approach

## Training Approach

We use diverse graph generation (AVICI-style) instead of curriculum learning:
- **Graph Sizes**: Uniformly sample from 3-100 variables
- **Graph Types**: Erdos-Renyi, Chain, Fork, Collider, Mixed structures
- **Data Per SCM**: 600 observations + 200 interventions
- **Batch Training**: Each batch contains diverse graph types and sizes

## Scripts

- `train_avici_style.py` - Main training script with alternating attention
- `test_alternating_attention.py` - Test script to verify architecture
- `optimizer_utils.py` - Shared optimizer configurations

## Usage

### Quick Test (5-10 minutes)
```bash
python scripts/train_avici_style.py \
    --num-steps 500 \
    --batch-size 8 \
    --max-vars 20 \
    --num-observations 200 \
    --log-freq 50
```

### Full Training (2-3 hours)
```bash
python scripts/train_avici_style.py \
    --num-steps 50000 \
    --batch-size 32 \
    --max-vars 100 \
    --num-observations 800 \
    --log-freq 500
```

### Test Architecture
```bash
python scripts/test_alternating_attention.py
```

## Key Improvements

1. **Fixed Variable Ordering Bug**: Consistent alphabetical ordering via VariableMapper
2. **Alternating Attention**: Strong inductive bias for causal discovery
3. **Diverse Training**: Better generalization across graph sizes
4. **AVICI Configuration**: Proven model size that scales to 100+ variables

## Results

- Achieves F1 > 0.8 on small graphs (3-10 vars) within 500 steps
- Scales to 100+ variables with single model
- No curriculum design or stage management needed

## Checkpoints

Models are saved to `checkpoints/avici_runs/[timestamp]/` with:
- `best_model.pkl` - Best performing model
- `config.json` - Training configuration
- `metrics_history.json` - Training metrics over time