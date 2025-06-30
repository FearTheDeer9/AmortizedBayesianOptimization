# Scripts Directory

This directory contains production and development scripts organized by functionality.

## Directory Structure

### 📁 **core/** - Production-Ready Scripts
- `erdos_renyi_scaling_experiment.py` - Main scaling experiment with Hydra/WandB
- `acbo_wandb_experiment.py` - **ACBO method comparison (frozen vs learning surrogates)**
- `collect_sft_dataset.py` - SFT dataset collection with checkpointing
- `prepare_sft_data.py` - SFT data format conversion
- `split_dataset.py` - Intelligent dataset splitting
- `validate_dataset.py` - Dataset quality validation

### 📁 **deployment/** - Cluster & Infrastructure
- `deploy_to_cluster.py` - Complete cluster deployment orchestration
- `sync_to_cluster.py` - Efficient project synchronization
- `validate_cluster_integration.py` - Cluster validation

### 📁 **benchmarks/** - Performance Analysis
- `performance_targets_validation.py` - Phase 2.2 requirements validation
- `training_speed_benchmarks.py` - Training speed analysis
- `sample_efficiency_benchmarks.py` - Sample efficiency validation
- `optimization_performance_benchmarks.py` - Target optimization benchmarks
- `structure_learning_benchmarks.py` - ⚠️ SIMULATION-ONLY structure learning benchmarks

### 📁 **development/** - Development Tools
- `dev_workflow.py` - Local development workflow
- `quick_validation.py` - Quick validation experiments
- `analyze_test_coverage.py` - Test coverage analysis

### 📁 **maintenance/** - Legacy & Cleanup
- `cleanup_deprecated.py` - JAX migration cleanup utilities

## Quick Start

**For ACBO comparison (frozen vs learning surrogates):**
```bash
poetry run python scripts/core/acbo_wandb_experiment.py --config-path ../config --config-name acbo_comparison_config.yaml
```

**For scaling experiments:**
```bash
poetry run python scripts/core/erdos_renyi_scaling_experiment.py --config-path ../config --config-name config.yaml
```

**For development workflow:**
```bash
poetry run python scripts/development/dev_workflow.py --mode quick-test
```

## Scripts

### Data Collection & Processing

#### `collect_sft_dataset.py`
Collects expert demonstrations for SFT training.

```bash
# Collect medium dataset (10K demonstrations)
python scripts/collect_sft_dataset.py --size medium --output-dir sft_data

# Collect with curriculum progression
python scripts/collect_sft_dataset.py --size large --difficulty difficulty_1 difficulty_2 --progressive

# Resume interrupted collection
python scripts/collect_sft_dataset.py --resume checkpoints/latest_checkpoint.pkl
```

**Features:**
- Configurable dataset sizes (small: 1K, medium: 10K, large: 100K, xlarge: 500K)
- Curriculum-aware difficulty progression
- Parallel processing with intelligent batch sizing
- Resumable collection with checkpointing
- Memory-aware resource management

#### `validate_dataset.py`
Validates dataset quality and generates reports.

```bash
# Basic validation
python scripts/validate_dataset.py sft_datasets/medium_dataset

# Detailed validation with visualizations
python scripts/validate_dataset.py sft_datasets/large_dataset --detailed --export-report --visualizations
```

**Features:**
- Data integrity and format validation
- Quality metrics (accuracy distributions, corruption detection)
- Distribution analysis and diversity metrics
- Visualization generation
- Actionable recommendations

#### `prepare_sft_data.py`
Converts demonstrations to SFT training format.

```bash
# Convert to JAX-compatible format
python scripts/prepare_sft_data.py sft_datasets/raw_data --output sft_datasets/training_ready

# Convert with curriculum awareness
python scripts/prepare_sft_data.py sft_datasets/raw_data --output sft_datasets/curriculum --format hdf5 --curriculum
```

**Features:**
- JAX-compatible [N, d, 3] format conversion
- Curriculum-aware difficulty classification
- Multiple output formats (pickle, HDF5, numpy)
- Memory-efficient processing

#### `split_dataset.py`
Creates stratified train/val/test splits.

```bash
# Basic 70/20/10 split
python scripts/split_dataset.py sft_datasets/training_ready --splits 0.7 0.2 0.1

# Curriculum-aware stratified split
python scripts/split_dataset.py sft_datasets/training_ready --stratify difficulty accuracy --curriculum
```

**Features:**
- Stratified splitting on multiple criteria
- Curriculum-aware strategies
- Balance verification and quality metrics
- Reproducible splits with seed control

### Cluster Deployment

#### `sync_to_cluster.py`
Syncs project code and data to cluster.

```bash
# Sync code only
python scripts/sync_to_cluster.py --user your_username --sync-code

# Sync specific data directory
python scripts/sync_to_cluster.py --user your_username --sync-data data/raw

# Full sync
python scripts/sync_to_cluster.py --user your_username --full-sync
```

**Features:**
- Incremental sync with rsync
- Bandwidth optimization
- Selective sync of code vs data
- Resume capability for interrupted transfers

#### `deploy_to_cluster.py`
Master deployment orchestrator.

```bash
# Setup deployment environment
python scripts/deploy_to_cluster.py --user your_username --setup

# Run complete pipeline
python scripts/deploy_to_cluster.py --user your_username --run-pipeline

# Monitor pipeline progress
python scripts/deploy_to_cluster.py --user your_username --monitor --follow
```

**Features:**
- One-command deployment setup
- Job submission and monitoring
- Dependency chain management
- Error recovery and retry logic

### Local Development

#### `dev_workflow.py`
Local development and testing workflow.

```bash
# Run quick integration test
python scripts/dev_workflow.py --quick-test

# Collect development dataset
python scripts/dev_workflow.py --collect-dev-data

# Test training pipeline
python scripts/dev_workflow.py --test-training
```

**Features:**
- Small-scale testing and validation
- Quick iteration and debugging
- Integration validation
- Configuration testing

## Cluster Directory Structure

The `cluster/` directory contains:

- `scripts/setup_env.sh`: Environment setup for cluster
- `jobs/collect_data.sbatch`: Data collection job template
- `jobs/train_surrogate.sbatch`: Surrogate training job template  
- `jobs/train_acquisition.sbatch`: Acquisition training job template

## Usage Workflow

### 1. Local Development
```bash
# Test your setup locally
python scripts/dev_workflow.py --quick-test

# Collect small development dataset
python scripts/dev_workflow.py --collect-dev-data

# Test processing pipeline
python scripts/dev_workflow.py --test-training
```

### 2. Cluster Deployment
```bash
# Setup cluster environment (one-time)
python scripts/deploy_to_cluster.py --user your_username --setup

# Run complete training pipeline
python scripts/deploy_to_cluster.py --user your_username --run-pipeline --dataset-size large

# Monitor progress
python scripts/deploy_to_cluster.py --user your_username --monitor --follow
```

### 3. Manual Cluster Usage
```bash
# Sync code and data
python scripts/sync_to_cluster.py --user your_username --full-sync

# SSH to cluster and run individual steps
ssh your_username@gpucluster2.doc.ic.ac.uk
cd /vol/bitbucket/your_username/causal_bayes_opt
source activate_env.sh

# Submit individual jobs
sbatch cluster/jobs/collect_data.sbatch
sbatch cluster/jobs/train_surrogate.sbatch
sbatch cluster/jobs/train_acquisition.sbatch
```

## Configuration

### Environment Variables
Key environment variables for cluster jobs:

- `DATASET_SIZE`: small/medium/large/xlarge
- `DIFFICULTY_LEVELS`: space-separated difficulty levels
- `EPOCHS`: training epochs
- `BATCH_SIZE`: training batch size
- `LEARNING_RATE`: learning rate

### Resource Allocation
Default resource requests:

- **Data Collection**: 1 GPU, 8 CPU cores, 32GB RAM, 24h
- **Surrogate Training**: 1 A100, 16 CPU cores, 64GB RAM, 48h  
- **Acquisition Training**: 1 A100, 16 CPU cores, 64GB RAM, 72h

## Monitoring and Debugging

### Job Monitoring
```bash
# Check job status
squeue -u $USER

# Check job details
scontrol show job JOB_ID

# View job logs
tail -f /vol/bitbucket/$USER/causal_bayes_opt/logs/training/train_surrogate_JOB_ID.out
```

### Common Issues

1. **Connection Issues**: Ensure SSH keys are set up and you're on Imperial network/VPN
2. **Disk Space**: Check available space with `df -h /vol/bitbucket/$USER`
3. **GPU Memory**: Monitor with `nvidia-smi` during training
4. **Dependency Errors**: Re-run environment setup if imports fail

## Best Practices

1. **Test Locally First**: Always run `dev_workflow.py --quick-test` before cluster deployment
2. **Small Scale Testing**: Use small datasets for initial validation
3. **Monitor Resources**: Check GPU utilization and memory usage
4. **Checkpoint Regularly**: Enable checkpointing for long-running jobs
5. **Archive Results**: Sync results back to local machine for analysis

## Support

For issues with:
- **Scripts**: Check logs and error messages
- **Cluster Access**: Contact Imperial Computing Support
- **ACBO System**: Review project documentation in `docs/`