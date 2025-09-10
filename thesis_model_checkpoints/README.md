# Thesis Model Checkpoints

This directory contains the model checkpoints actually used to generate the results reported in the thesis, specifically from the experiments in `thesis_results/surrogate/different_size_performance/`.

## Checkpoints

### Policy Models

- `policy_fork_and_chain_phase_32.pkl` - Fork and chain specialized policy
  - This is the main policy used in most surrogate different_size_performance experiments
  - Referenced as `local_simple_10hr/fork_and_chain/policy_phase_30.pkl` in evaluation files (phase number differs)
  - Used in: chain, fork, star, scale_free, scale_free_continued_learning, and most 300_node evaluations
  - From: `checkpoints/fork_and_chain/policy_phase_32.pkl`

- `policy_production_12hour_phase_12.pkl` - Joint training policy from 12-hour production run
  - Used for Erdős-Rényi graph experiments
  - From: `experiments/joint-training/checkpoints/production_12hour/policy_phase_12.pkl`

### Surrogate Models

- `surrogate_avici_chain_20250903_154909.pkl` - AVICI-style surrogate for chain structures
  - Used in chain structure evaluations  
  - From: `checkpoints/avici_runs/avici_style_20250903_154909/checkpoint_step_1000.pkl`

- `surrogate_avici_fork_20250903_220822.pkl` - AVICI-style surrogate for fork structures
  - Used in fork structure evaluations
  - From: `checkpoints/avici_runs/avici_style_20250903_220822/checkpoint_step_1000.pkl`

- `surrogate_scalefree_production_phase_33.pkl` - Joint training surrogate for scale-free graphs
  - Used in scale-free structure evaluations
  - From: `imperial-vm-checkpoints/checkpoints/production_12hr_10hr_trainingy/surrogate_phase_33.pkl`

- `surrogate_scalefree_continued_phase_32.pkl` - Continued learning surrogate for scale-free graphs  
  - Used in scale-free continued learning experiments
  - From: `imperial-vm-checkpoints/checkpoints/vm_scalefree_target_12hr/surrogate_phase_32.pkl`

- `surrogate_avici_erdos_renyi_20250905_222147.pkl` - AVICI-style surrogate for Erdős-Rényi graphs
  - Used in Erdős-Rényi and 300-node evaluations
  - From: `imperial-vm-checkpoints/avici_style_20250905_222147/checkpoint_step_1000.pkl`

- `surrogate_fork_and_chain_phase_31.pkl` - Fork and chain specialized surrogate
  - Used specifically for star/collider structure evaluations
  - Referenced as `local_simple_10hr/fork_and_chain/surrogate_phase_31.pkl` in evaluation files
  - From: `checkpoints/fork_and_chain/surrogate_phase_31.pkl`

## Usage

These checkpoints can be loaded using the evaluation scripts in `experiments/evaluation/stepwise/`:

```bash
python full_evaluation.py \
    --policy-path thesis_model_checkpoints/policy_production_12hour_phase_12.pkl \
    --surrogate-path thesis_model_checkpoints/surrogate_avici_chain_20250903_154909.pkl
```

## Mapping to Thesis Results

The specific usage of each checkpoint can be traced through the JSON evaluation files in:
- `thesis_results/surrogate/different_size_performance/chain/` - Uses chain surrogate
- `thesis_results/surrogate/different_size_performance/fork/` - Uses fork surrogate  
- `thesis_results/surrogate/different_size_performance/scale_free/` - Uses scale-free production surrogate
- `thesis_results/surrogate/different_size_performance/scale_free_continued_learning/` - Uses continued learning surrogate
- `thesis_results/surrogate/different_size_performance/erdos_renyi_checkpoint1000/` - Uses Erdős-Rényi surrogate
- `thesis_results/surrogate/different_size_performance/300_node_evaluation/` - Uses various surrogates for different structures