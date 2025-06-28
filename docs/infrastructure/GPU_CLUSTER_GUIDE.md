# GPU Cluster Data Collection Guide

**Objective**: Set up and validate comprehensive SFT data collection on Imperial's GPU cluster

## 🎯 Quick Start

### Prerequisites
- Imperial College HPC account
- SSH access to `login.hpc.ic.ac.uk`
- Local development environment working (confirmed with `python scripts/dev_workflow.py --quick-test`)

### Step-by-Step Process

#### 1. 🔧 **Initial Setup & Validation**

```bash
# Show validation steps overview
python scripts/validate_cluster_integration.py --user YOUR_USERNAME --show-steps

# Run comprehensive validation (recommended)
python scripts/validate_cluster_integration.py --user YOUR_USERNAME --step all
```

#### 2. 🚀 **Quick Validation (If you prefer manual steps)**

```bash
# A. Sync project to cluster
python scripts/sync_to_cluster.py --user YOUR_USERNAME --sync-code

# B. Test environment on cluster  
ssh YOUR_USERNAME@login.hpc.ic.ac.uk 'cd /vol/bitbucket/YOUR_USERNAME/causal_bayes_opt && python scripts/dev_workflow.py --quick-test'

# C. Submit small test job
ssh YOUR_USERNAME@login.hpc.ic.ac.uk 'cd /vol/bitbucket/YOUR_USERNAME/causal_bayes_opt && sbatch --export=DATASET_SIZE=small,BATCH_SIZE=5 cluster/jobs/collect_data.sbatch'

# D. Monitor job
ssh YOUR_USERNAME@login.hpc.ic.ac.uk 'squeue -u YOUR_USERNAME'
```

#### 3. 📊 **Production Data Collection**

Once validation passes, start comprehensive collection:

```bash
# Medium dataset (10K demonstrations) - Recommended first production run
ssh YOUR_USERNAME@login.hpc.ic.ac.uk 'cd /vol/bitbucket/YOUR_USERNAME/causal_bayes_opt && sbatch --export=DATASET_SIZE=medium cluster/jobs/collect_data.sbatch'

# Large dataset (100K demonstrations) - Full production
ssh YOUR_USERNAME@login.hpc.ic.ac.uk 'cd /vol/bitbucket/YOUR_USERNAME/causal_bayes_opt && sbatch --export=DATASET_SIZE=large cluster/jobs/collect_data.sbatch'

# Extra large dataset (500K demonstrations) - Maximum scale
ssh YOUR_USERNAME@login.hpc.ic.ac.uk 'cd /vol/bitbucket/YOUR_USERNAME/causal_bayes_opt && sbatch --export=DATASET_SIZE=xlarge cluster/jobs/collect_data.sbatch'
```

## 📋 Detailed Configuration

### Dataset Sizes & Estimated Times

| Size | Demonstrations | Estimated Time | Recommended Use |
|------|----------------|----------------|-----------------|
| `small` | 1,000 | 30 min | Testing, validation |
| `medium` | 10,000 | 3-6 hours | Development, initial training |
| `large` | 100,000 | 1-2 days | Production training |
| `xlarge` | 500,000 | 3-5 days | Maximum scale research |

### Slurm Job Configuration

The `cluster/jobs/collect_data.sbatch` script includes:

- **Partition**: `training` (for taught students)
- **Resources**: 1 GPU, 8 CPUs, 32GB RAM
- **Time Limit**: 24 hours (adjust for larger datasets)
- **Processing**: Serial mode (avoiding pickle issues)
- **Auto-validation**: Runs validation after collection

### Environment Variables

Customize collection with environment variables:

```bash
# Example: Medium dataset with custom settings
sbatch --export=DATASET_SIZE=medium,BATCH_SIZE=20,MIN_ACCURACY=0.8 cluster/jobs/collect_data.sbatch

# Example: Progressive difficulty training
sbatch --export=DATASET_SIZE=large,DIFFICULTY_LEVELS="difficulty_1 difficulty_2" cluster/jobs/collect_data.sbatch
```

Available variables:
- `DATASET_SIZE`: small/medium/large/xlarge
- `BATCH_SIZE`: Demonstrations per batch (default: 100)
- `MIN_ACCURACY`: Minimum PARENT_SCALE accuracy (default: 0.7)
- `DIFFICULTY_LEVELS`: Specific difficulties or "all"
- `OUTPUT_DIR`: Custom output directory

## 🔍 Monitoring & Troubleshooting

### Monitor Job Progress

```bash
# Check job status
squeue -u YOUR_USERNAME

# Watch job output in real-time
tail -f /vol/bitbucket/YOUR_USERNAME/causal_bayes_opt/logs/collection/collect_data_JOBID.out

# Check for errors
tail -f /vol/bitbucket/YOUR_USERNAME/causal_bayes_opt/logs/collection/collect_data_JOBID.err
```

### Common Issues & Solutions

#### 1. **Environment Issues**
```bash
# Problem: Module not found or import errors
# Solution: Re-run environment setup
ssh YOUR_USERNAME@login.hpc.ic.ac.uk 'cd /vol/bitbucket/YOUR_USERNAME/causal_bayes_opt && source cluster/scripts/setup_env.sh'
```

#### 2. **Job Failed to Start**
```bash
# Problem: Job pending or failed to submit
# Solution: Check partition and QoS settings
sinfo  # Check available partitions
squeue -u YOUR_USERNAME  # Check job status
```

#### 3. **Out of Memory Errors**
```bash
# Problem: Job killed due to memory
# Solution: Reduce batch size
sbatch --export=DATASET_SIZE=medium,BATCH_SIZE=50 cluster/jobs/collect_data.sbatch
```

#### 4. **Validation Failures**
```bash
# Problem: Data validation fails after collection
# Solution: Check specific validation output
ssh YOUR_USERNAME@login.hpc.ic.ac.uk 'cd /vol/bitbucket/YOUR_USERNAME/causal_bayes_opt && python scripts/validate_dataset.py data/raw --detailed'
```

## 📊 Post-Collection Workflow

### 1. **Download Results**
```bash
# Sync collected data back to local machine
python scripts/sync_to_cluster.py --user YOUR_USERNAME --sync-data data/raw

# Or use rsync directly
rsync -avz YOUR_USERNAME@login.hpc.ic.ac.uk:/vol/bitbucket/YOUR_USERNAME/causal_bayes_opt/data/raw/ ./cluster_data/
```

### 2. **Prepare for Training**
```bash
# Convert to training format
python scripts/prepare_sft_data.py cluster_data/raw --output ./training_data

# Split dataset
python scripts/split_dataset.py training_data --output ./training_splits
```

### 3. **Start Training**
```bash
# Submit training jobs (if desired on cluster)
ssh YOUR_USERNAME@login.hpc.ic.ac.uk 'cd /vol/bitbucket/YOUR_USERNAME/causal_bayes_opt && sbatch cluster/jobs/train_surrogate.sbatch'
```

## 🎯 Production Recommendations

### For Development Phase
1. Start with `DATASET_SIZE=small` for validation
2. Run `DATASET_SIZE=medium` for initial training experiments  
3. Monitor resource usage and adjust batch sizes

### For Production Training
1. Use `DATASET_SIZE=large` for main training datasets
2. Run multiple jobs with different difficulty levels for curriculum learning
3. Set up automated monitoring and recovery

### Resource Optimization
- **Serial Mode**: Currently using serial processing to avoid pickle issues
- **Batch Size**: Start with 100, reduce if memory issues occur
- **Checkpointing**: Automatic resume capability for interrupted jobs
- **Validation**: Built-in quality checks and dataset validation

## 🚨 Important Notes

1. **Serial Processing**: Updated to use serial mode due to SCM pickle issues. Parallel processing fix planned for future.

2. **Resource Limits**: Jobs limited to 24 hours. For larger datasets, consider splitting into multiple jobs.

3. **Storage**: Ensure sufficient space in `/vol/bitbucket/YOUR_USERNAME/` for dataset storage.

4. **Queue Priority**: Using `training` partition - jobs may have lower priority during peak times.

5. **Data Persistence**: Collected data persists in cluster storage - remember to clean up when no longer needed.

---

## 📞 Need Help?

1. **Validation Issues**: Run `python scripts/validate_cluster_integration.py --user YOUR_USERNAME --step environment`
2. **Job Issues**: Check logs in `/vol/bitbucket/YOUR_USERNAME/causal_bayes_opt/logs/collection/`
3. **Import Errors**: Verify environment with `python scripts/dev_workflow.py --quick-test`
4. **Storage Issues**: Check disk usage with `du -sh /vol/bitbucket/YOUR_USERNAME/causal_bayes_opt/`