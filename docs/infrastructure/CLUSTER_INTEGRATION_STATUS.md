# GPU Cluster Integration Status & Production Roadmap

**Created**: 2025-06-23  
**Last Updated**: 2025-06-23  
**Status**: Environment Setup In Progress  
**User**: hl1824@ic.ac.uk  

## 🎯 **Executive Summary**

We have successfully implemented a comprehensive SFT dataset collection infrastructure for the ACBO project. Local development is fully operational, cluster deployment scripts are ready, and we've resolved critical parallel processing issues. The final step is completing the Imperial HPC cluster environment setup and launching production data collection.

---

## ✅ **What We Have Accomplished**

### **1. Local Development Infrastructure (Complete)**

#### **Core Collection Scripts** ✅
- **`scripts/collect_sft_dataset.py`** - Production-ready SFT dataset collection
  - Supports 4 dataset sizes: small (1K), medium (10K), large (100K), xlarge (500K)
  - Curriculum-aware difficulty progression
  - Automatic checkpointing and resumable collection
  - Memory-aware batch sizing
  - Comprehensive error handling and logging

- **`scripts/validate_dataset.py`** - Data quality validation and analysis
  - Integrity checks and corruption detection
  - Quality metrics (accuracy distributions, diversity analysis)
  - Visualization generation and JSON reporting
  - Actionable recommendations for dataset improvement

- **`scripts/prepare_sft_data.py`** - JAX-compatible format conversion
  - Converts PARENT_SCALE demonstrations to [N, d, 3] format
  - Extracts state-action pairs for behavioral cloning
  - Preserves metadata and curriculum information
  - Supports multiple output formats (pickle, HDF5, numpy)

- **`scripts/split_dataset.py`** - Stratified dataset splitting
  - Multi-criteria stratification (difficulty, graph_type, accuracy, node_size)
  - Curriculum-aware splitting strategies
  - Balance verification and reproducible splits
  - Quality metrics for each split

#### **Development Workflow** ✅
- **`scripts/dev_workflow.py`** - Streamlined local testing
  - Quick integration tests (`--quick-test`)
  - Development data collection (`--collect-dev-data`)
  - Configuration validation (`--test-config`) 
  - Cleanup utilities (`--clean`)
  - **Updated to use serial mode by default for reliability**

#### **Critical Bug Fixes** ✅
- **Parallel Processing Pickle Issue (Fixed)**
  - **Root Cause**: SCM mechanisms contained nested functions that couldn't be pickled for multiprocessing
  - **Solution**: Implemented automatic fallback from parallel to serial processing
  - **Error Detection**: Smart detection of pickle-related failures with graceful degradation
  - **Reliability**: Permanent switch to serial mode after first failure to avoid repeated errors
  - **Status**: All local tests pass, data collection works reliably

### **2. Cluster Infrastructure (Ready for Deployment)**

#### **Deployment Scripts** ✅
- **`scripts/deploy_to_cluster.py`** - Master deployment orchestrator
  - One-command setup and pipeline execution
  - Job submission and monitoring
  - Resource optimization and error recovery
  - Dependency chain management

- **`scripts/sync_to_cluster.py`** - Project synchronization
  - Incremental sync with rsync for speed
  - Selective sync of code vs data
  - Compression and bandwidth optimization
  - Resume capability for interrupted transfers
  - **Updated hostname to `login.hpc.ic.ac.uk`**

#### **Slurm Job Templates** ✅
- **`cluster/jobs/collect_data.sbatch`** - Data collection job
  - Configured for Imperial's training partition
  - 1 GPU, 8 CPUs, 32GB RAM, 24-hour limit
  - **Serial processing mode (avoiding pickle issues)**
  - Automatic validation after collection
  - Configurable via environment variables
  - Comprehensive logging and error reporting

- **`cluster/jobs/train_surrogate.sbatch`** - Surrogate model training
- **`cluster/jobs/train_acquisition.sbatch`** - GRPO acquisition training

#### **Environment Setup** ✅
- **`cluster/scripts/setup_env.sh`** - Complete environment configuration
  - JAX GPU support setup
  - Python environment with all dependencies
  - Module loading for Imperial's cluster
  - Environment variable configuration

#### **Validation Tools** ✅
- **`scripts/validate_cluster_integration.py`** - Step-by-step cluster validation
  - Project synchronization validation
  - Environment setup verification
  - PARENT_SCALE integration testing
  - Slurm job submission testing
  - Full pipeline validation
  - **Comprehensive error handling and user guidance**

#### **Documentation** ✅
- **`GPU_CLUSTER_GUIDE.md`** - Operational guide with troubleshooting
- **`IMPERIAL_CLUSTER_SETUP.md`** - Imperial-specific setup instructions
- **`SFT_DATASET_COLLECTION_PLAN.md`** - Technical implementation details

### **3. Testing & Validation** ✅

#### **Local Validation** ✅
- ✅ Environment checks pass (JAX, dependencies)
- ✅ Import validation successful
- ✅ PARENT_SCALE integration working
- ✅ Expert collection functional (serial mode)
- ✅ Data processing pipeline operational

#### **Integration Testing** ✅
- ✅ End-to-end local workflow tested
- ✅ Collection → Validation → Processing → Splitting pipeline working
- ✅ Error handling and recovery mechanisms validated
- ✅ Checkpointing and resume functionality confirmed

---

## 🔄 **Current Status: Environment Setup In Progress**

### **SSH Connection Resolution** ✅ → 🔄
- ✅ **Issue Identified**: Required Imperial VPN for off-campus access
- ✅ **Solution Applied**: User configured VPN connection  
- ✅ **SSH Access**: Successfully connecting to `login.hpc.ic.ac.uk`
- 🔄 **Current**: Resolving remaining VPN/environment configuration issues

### **Next Immediate Steps**
1. **Complete VPN/SSH setup** (user working on this)
2. **Transfer project to cluster** 
3. **Validate cluster environment**
4. **Submit test collection job**

---

## 🎯 **What We Need To Do Next**

### **Phase 1: Complete Cluster Environment Setup** 
**Status**: Ready to execute once SSH issues resolved  
**Estimated Time**: 30 minutes  

#### **1.1 Project Transfer**
```bash
# From local machine
cd /Users/harellidar/Documents/Imperial/Individual_Project/causal_bayes_opt

# Create clean archive (already prepared)
tar --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='dev_data' --exclude='dev_results' --exclude='*.pkl' \
    --exclude='cluster_data' --exclude='.claude' \
    -czf causal_bayes_opt_cluster.tar.gz .

# Transfer to cluster  
scp causal_bayes_opt_cluster.tar.gz hl1824@login.hpc.ic.ac.uk:/vol/bitbucket/hl1824/
```

#### **1.2 Cluster Setup**
```bash
# SSH to cluster
ssh hl1824@login.hpc.ic.ac.uk

# Extract and setup
cd /vol/bitbucket/hl1824/
tar -xzf causal_bayes_opt_cluster.tar.gz
mv causal_bayes_opt_cluster.tar.gz causal_bayes_opt/
cd causal_bayes_opt

# Setup environment
source cluster/scripts/setup_env.sh
```

#### **1.3 Environment Validation**
```bash
# Test Python/JAX environment
python --version
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}')"

# Run comprehensive integration test
python scripts/dev_workflow.py --quick-test
```

**Success Criteria**:
- ✅ JAX imports successfully and reports devices
- ✅ Integration test passes all 5 checks
- ✅ PARENT_SCALE algorithm runs without errors

### **Phase 2: Production Data Collection Campaign**
**Status**: Ready once Phase 1 complete  
**Estimated Time**: 30 minutes - 5 days (depending on dataset size)  

#### **2.1 Validation Collection**
```bash
# Submit small test job for validation
mkdir -p logs/collection
sbatch --export=DATASET_SIZE=small,BATCH_SIZE=5 cluster/jobs/collect_data.sbatch

# Monitor job
squeue -u hl1824
tail -f logs/collection/collect_data_*.out
```

**Success Criteria**:
- ✅ Job submits successfully
- ✅ Collects 1,000 demonstrations without errors
- ✅ Automatic validation passes
- ✅ Output files generated correctly

#### **2.2 Development Dataset**
```bash
# Submit medium-scale collection (recommended first production run)
sbatch --export=DATASET_SIZE=medium cluster/jobs/collect_data.sbatch
```

**Details**:
- **Size**: 10,000 demonstrations
- **Time**: ~6 hours
- **Storage**: ~1GB
- **Purpose**: Development and initial training experiments

#### **2.3 Production Dataset**
```bash
# Submit large-scale collection
sbatch --export=DATASET_SIZE=large cluster/jobs/collect_data.sbatch
```

**Details**:
- **Size**: 100,000 demonstrations  
- **Time**: ~2 days
- **Storage**: ~10GB
- **Purpose**: Production ACBO training

#### **2.4 Research Scale Dataset**
```bash
# Submit maximum scale collection
sbatch --export=DATASET_SIZE=xlarge cluster/jobs/collect_data.sbatch
```

**Details**:
- **Size**: 500,000 demonstrations
- **Time**: ~5 days  
- **Storage**: ~50GB
- **Purpose**: Research-scale experiments

### **Phase 3: Post-Collection Processing**
**Status**: Ready for execution  
**Estimated Time**: 1-2 hours per dataset  

#### **3.1 Data Download**
```bash
# Download collected data from cluster
rsync -avz --progress hl1824@login.hpc.ic.ac.uk:/vol/bitbucket/hl1824/causal_bayes_opt/data/raw/ ./cluster_data/
```

#### **3.2 Quality Validation**
```bash
# Run comprehensive validation locally
python scripts/validate_dataset.py cluster_data --detailed --export-report --visualizations

# Check validation report
cat cluster_data/validation_report.json
```

#### **3.3 Training Data Preparation**
```bash
# Convert to JAX-compatible training format  
python scripts/prepare_sft_data.py cluster_data --output ./training_data --format pickle

# Split into train/val/test sets
python scripts/split_dataset.py training_data --output ./training_splits --strategy curriculum_aware
```

#### **3.4 Begin ACBO Training**
```bash
# Local training (if desired)
python scripts/train_surrogate_sft.py training_splits/train --warmstart

# Or submit cluster training jobs
ssh hl1824@login.hpc.ic.ac.uk 'cd /vol/bitbucket/hl1824/causal_bayes_opt && sbatch cluster/jobs/train_surrogate.sbatch'
```

---

## 🚀 **Production Commands Reference**

### **Environment Validation Commands**
```bash
# Test SSH connection
ssh hl1824@login.hpc.ic.ac.uk

# Validate cluster environment  
ssh hl1824@login.hpc.ic.ac.uk 'cd /vol/bitbucket/hl1824/causal_bayes_opt && python scripts/dev_workflow.py --quick-test'

# Check available resources
ssh hl1824@login.hpc.ic.ac.uk 'sinfo && squeue'
```

### **Data Collection Commands**
```bash
# Validation job (30 minutes)
sbatch --export=DATASET_SIZE=small,BATCH_SIZE=5 cluster/jobs/collect_data.sbatch

# Development dataset (6 hours)  
sbatch --export=DATASET_SIZE=medium cluster/jobs/collect_data.sbatch

# Production dataset (2 days)
sbatch --export=DATASET_SIZE=large cluster/jobs/collect_data.sbatch

# Research scale (5 days)
sbatch --export=DATASET_SIZE=xlarge cluster/jobs/collect_data.sbatch
```

### **Monitoring Commands**
```bash
# Check job queue
squeue -u hl1824

# Monitor job progress
tail -f logs/collection/collect_data_*.out

# Check for errors
tail -f logs/collection/collect_data_*.err

# Monitor data collection progress
watch -n 300 'ls -la data/raw/ && du -sh data/raw/'  # Update every 5 minutes
```

### **Data Management Commands**
```bash
# Check collected data
ls -la data/raw/
du -sh data/raw/

# Download specific dataset
rsync -avz hl1824@login.hpc.ic.ac.uk:/vol/bitbucket/hl1824/causal_bayes_opt/data/raw/medium/ ./local_medium/

# Clean up cluster storage (after download)
ssh hl1824@login.hpc.ic.ac.uk 'rm -rf /vol/bitbucket/hl1824/causal_bayes_opt/data/raw/medium/'
```

---

## 📊 **Expected Outcomes & Success Metrics**

### **Dataset Collection Targets**

| Dataset | Demonstrations | Storage | Time | Success Rate Target | Use Case |
|---------|----------------|---------|------|-------------------|----------|
| `small` | 1,000 | 100MB | 30 min | >90% | Validation, testing |
| `medium` | 10,000 | 1GB | 6 hours | >85% | Development training |
| `large` | 100,000 | 10GB | 2 days | >80% | Production training |
| `xlarge` | 500,000 | 50GB | 5 days | >75% | Research experiments |

### **Quality Metrics**

#### **Demonstration Quality**
- **Accuracy Distribution**: Mean accuracy >0.7, with good coverage across difficulty levels
- **Graph Type Diversity**: Balanced representation of chain, star, fork, collider structures
- **Node Size Coverage**: Adequate sampling across 3-20 node graphs
- **Curriculum Progression**: Clear difficulty progression from easy to hard problems

#### **Data Integrity**
- **Corruption Rate**: <1% corrupted or incomplete demonstrations
- **Consistency**: All demonstrations follow expected data format [N, d, 3]
- **Metadata Completeness**: 100% of demonstrations have complete metadata
- **Trajectory Validity**: All PARENT_SCALE trajectories are valid and complete

#### **Training Readiness**
- **Split Balance**: Train/val/test splits maintain distribution balance
- **Format Consistency**: All data in JAX-compatible format
- **Memory Efficiency**: Efficient loading and batching for training
- **Reproducibility**: Deterministic splits and consistent random seeds

---

## 🔧 **Troubleshooting Guide**

### **Common Issues & Solutions**

#### **SSH/Connection Issues**
```bash
# Problem: SSH connection fails
# Solution: Ensure VPN is connected
# Check: ping login.hpc.ic.ac.uk

# Problem: Permission denied
# Solution: Verify username and VPN status
# Check: ssh -v hl1824@login.hpc.ic.ac.uk
```

#### **Environment Issues**
```bash
# Problem: Module not found errors
# Solution: Re-run environment setup
source cluster/scripts/setup_env.sh

# Problem: JAX device errors  
# Solution: Check CUDA modules
module list
module avail cuda
```

#### **Job Submission Issues**
```bash
# Problem: sbatch fails
# Solution: Check partition availability
sinfo
squeue

# Problem: Job pending indefinitely
# Solution: Check resource requests and queue
scontrol show job JOBID
```

#### **Storage Issues**
```bash
# Problem: Disk quota exceeded
# Solution: Check usage and clean up
df -h /vol/bitbucket/hl1824/
du -sh /vol/bitbucket/hl1824/causal_bayes_opt/

# Clean up temporary files
find /vol/bitbucket/hl1824/causal_bayes_opt/ -name "*.tmp" -delete
```

#### **Collection Issues**
```bash
# Problem: Low success rate
# Solution: Adjust accuracy threshold
sbatch --export=DATASET_SIZE=medium,MIN_ACCURACY=0.6 cluster/jobs/collect_data.sbatch

# Problem: Memory errors
# Solution: Reduce batch size
sbatch --export=DATASET_SIZE=medium,BATCH_SIZE=50 cluster/jobs/collect_data.sbatch
```

---

## 📈 **Performance Optimization**

### **Current Optimizations**
- ✅ **Serial Processing**: Reliable collection without pickle issues
- ✅ **Checkpointing**: Resume capability for interrupted jobs
- ✅ **Memory Management**: Adaptive batch sizing based on graph complexity
- ✅ **Error Recovery**: Automatic retry logic with progressive difficulty

### **Future Optimizations** (Post-Production)
- 🔄 **Parallel Processing Fix**: Refactor SCM mechanism serialization for parallel speedup
- 🔄 **GPU Acceleration**: Leverage GPU for PARENT_SCALE inference acceleration
- 🔄 **Distributed Collection**: Multi-node collection for maximum throughput
- 🔄 **Intelligent Caching**: Cache inference results for similar graph structures

---

## 📞 **Support & Resources**

### **Imperial HPC Resources**
- **HPC Documentation**: https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/hpc/
- **VPN Setup**: https://www.imperial.ac.uk/admin-services/ict/self-service/connect-communicate/remote-access/vpn/
- **SSH Guide**: https://www.imperial.ac.uk/computing/people/csg/guides/remote-access/ssh/

### **Project Resources**
- **Main Documentation**: `GPU_CLUSTER_GUIDE.md`
- **Technical Details**: `SFT_DATASET_COLLECTION_PLAN.md`
- **Local Testing**: `scripts/dev_workflow.py --help`
- **Validation Tools**: `scripts/validate_cluster_integration.py --help`

### **Contact Points**
- **User Account**: hl1824@ic.ac.uk
- **Project Directory**: `/vol/bitbucket/hl1824/causal_bayes_opt`
- **Log Location**: `/vol/bitbucket/hl1824/causal_bayes_opt/logs/collection/`

---

## 📋 **Action Items Summary**

### **Immediate (Next 1-2 hours)**
- [ ] **Complete VPN/SSH resolution** (user)
- [ ] **Transfer project to cluster** (`scp causal_bayes_opt_cluster.tar.gz`)
- [ ] **Setup cluster environment** (`source cluster/scripts/setup_env.sh`)
- [ ] **Run validation test** (`python scripts/dev_workflow.py --quick-test`)

### **Short Term (Next 1-2 days)**  
- [ ] **Submit validation job** (`sbatch --export=DATASET_SIZE=small`)
- [ ] **Monitor and verify completion**
- [ ] **Submit development dataset** (`sbatch --export=DATASET_SIZE=medium`)
- [ ] **Monitor 6-hour collection process**

### **Medium Term (Next 1-2 weeks)**
- [ ] **Submit production dataset** (`sbatch --export=DATASET_SIZE=large`)
- [ ] **Monitor 2-day collection process**  
- [ ] **Download and validate collected data**
- [ ] **Prepare training datasets**

### **Long Term (Research Goals)**
- [ ] **Submit research scale dataset** (`sbatch --export=DATASET_SIZE=xlarge`)
- [ ] **Begin ACBO training with collected data**
- [ ] **Evaluate training performance and adjust collection strategy**
- [ ] **Implement parallel processing optimizations for future collections**

---

**Status**: Ready for cluster deployment once SSH/VPN issues resolved  
**Next Update**: After successful cluster environment validation  
**Estimated Timeline to Production**: 2-3 days (including data collection)