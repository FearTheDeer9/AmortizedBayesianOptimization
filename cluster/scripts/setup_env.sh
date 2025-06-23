#!/bin/bash
"""
GPU Cluster Environment Setup Script

Sets up the complete environment for ACBO training on Imperial's GPU cluster.
Handles CUDA setup, Python virtual environment, and dependency installation.

Usage:
    bash cluster/scripts/setup_env.sh
    source cluster/scripts/setup_env.sh  # To activate environment in current shell
"""

set -e  # Exit on any error

# Configuration
PROJECT_NAME="acbo_training"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8"
PROJECT_DIR="/vol/bitbucket/${USER}/causal_bayes_opt"
VENV_DIR="/vol/bitbucket/${USER}/envs/${PROJECT_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on the cluster
check_cluster_environment() {
    log_info "Checking cluster environment..."
    
    if [[ ! "$HOSTNAME" =~ "gpucluster" ]]; then
        log_warning "Not on GPU cluster (hostname: $HOSTNAME)"
        log_warning "This script is optimized for Imperial's GPU cluster"
    fi
    
    # Check if we're in a job or on login node
    if [[ -n "$SLURM_JOB_ID" ]]; then
        log_info "Running in Slurm job: $SLURM_JOB_ID"
        log_info "Allocated GPUs: ${CUDA_VISIBLE_DEVICES:-None}"
    else
        log_info "Running on login node"
    fi
}

# Setup CUDA environment
setup_cuda() {
    log_info "Setting up CUDA environment..."
    
    # Load CUDA module if available
    if command -v module &> /dev/null; then
        if module avail cuda 2>&1 | grep -q "cuda/${CUDA_VERSION}"; then
            log_info "Loading CUDA module: cuda/${CUDA_VERSION}"
            module load cuda/${CUDA_VERSION}
        else
            log_warning "CUDA module ${CUDA_VERSION} not found, trying default"
            if module avail cuda 2>&1 | grep -q "cuda/"; then
                module load cuda
            fi
        fi
    fi
    
    # Check CUDA installation
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION_INSTALLED=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        log_success "CUDA ${CUDA_VERSION_INSTALLED} available"
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        export PATH="${CUDA_HOME}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
    else
        log_error "CUDA not found! Please ensure CUDA is installed."
        exit 1
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        log_success "Found ${GPU_COUNT} GPU(s)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        log_warning "nvidia-smi not found - GPU status unknown"
    fi
}

# Create project directory structure
setup_project_directory() {
    log_info "Setting up project directory structure..."
    
    # Create main project directory
    mkdir -p "${PROJECT_DIR}"
    cd "${PROJECT_DIR}"
    
    # Create data directories
    mkdir -p data/{raw,processed,splits,checkpoints}
    mkdir -p logs/{collection,training,validation}
    mkdir -p results/{models,evaluation,plots}
    mkdir -p configs
    
    log_success "Project directory structure created: ${PROJECT_DIR}"
}

# Setup Python virtual environment
setup_python_environment() {
    log_info "Setting up Python virtual environment..."
    
    # Check Python version
    if command -v python${PYTHON_VERSION} &> /dev/null; then
        PYTHON_CMD="python${PYTHON_VERSION}"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        log_warning "Python ${PYTHON_VERSION} not found, using $(python3 --version)"
    else
        log_error "Python not found!"
        exit 1
    fi
    
    # Create virtual environment
    if [ ! -d "${VENV_DIR}" ]; then
        log_info "Creating virtual environment: ${VENV_DIR}"
        ${PYTHON_CMD} -m venv "${VENV_DIR}"
    else
        log_info "Virtual environment already exists: ${VENV_DIR}"
    fi
    
    # Activate virtual environment
    source "${VENV_DIR}/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    log_success "Python environment ready: $(python --version)"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Ensure we're in virtual environment
    if [[ "$VIRTUAL_ENV" != "${VENV_DIR}" ]]; then
        log_error "Virtual environment not activated!"
        exit 1
    fi
    
    # Install JAX with GPU support
    log_info "Installing JAX with GPU support..."
    pip install --upgrade "jax[cuda${CUDA_VERSION//./_}]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    
    # Install core ML dependencies
    log_info "Installing core ML dependencies..."
    pip install \
        numpy \
        scipy \
        matplotlib \
        seaborn \
        scikit-learn \
        h5py \
        tqdm \
        wandb \
        tensorboard
    
    # Install optimization and neural network libraries
    log_info "Installing optimization libraries..."
    pip install \
        optax \
        dm-haiku \
        chex \
        jaxlib
    
    # Install additional scientific computing
    log_info "Installing scientific computing libraries..."
    pip install \
        pandas \
        networkx \
        pyrsistent \
        hypothesis
    
    # Install project in development mode if pyproject.toml exists
    if [ -f "${PROJECT_DIR}/pyproject.toml" ]; then
        log_info "Installing project in development mode..."
        cd "${PROJECT_DIR}"
        pip install -e .
    else
        log_warning "pyproject.toml not found, skipping project installation"
    fi
    
    log_success "Dependencies installed successfully"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Test JAX GPU
    python -c "
import jax
import jax.numpy as jnp
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
print(f'JAX local devices: {jax.local_devices()}')

# Test GPU computation
if jax.devices()[0].platform == 'gpu':
    x = jnp.array([1.0, 2.0, 3.0])
    result = jnp.dot(x, x)
    print(f'GPU computation test: {result}')
    print('✅ JAX GPU setup successful!')
else:
    print('⚠️ JAX using CPU - GPU not available')
"
    
    # Test other key imports
    python -c "
try:
    import numpy as np
    import optax
    import haiku as hk
    import matplotlib.pyplot as plt
    print('✅ All key dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"
    
    log_success "Installation verification complete"
}

# Create activation script
create_activation_script() {
    log_info "Creating environment activation script..."
    
    cat > "${PROJECT_DIR}/activate_env.sh" << EOF
#!/bin/bash
# ACBO Environment Activation Script
# Source this script to activate the environment: source activate_env.sh

# Activate virtual environment
source "${VENV_DIR}/bin/activate"

# Setup CUDA environment
export CUDA_HOME=\$(dirname \$(dirname \$(which nvcc)))
export PATH="\${CUDA_HOME}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH}"

# Set project environment variables
export PROJECT_DIR="${PROJECT_DIR}"
export PYTHONPATH="\${PROJECT_DIR}/src:\${PYTHONPATH}"

# Set JAX configuration for optimal GPU usage
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

echo "🚀 ACBO environment activated!"
echo "Project directory: \${PROJECT_DIR}"
echo "Python: \$(python --version)"
echo "JAX devices: \$(python -c 'import jax; print(jax.devices())')"
EOF
    
    chmod +x "${PROJECT_DIR}/activate_env.sh"
    log_success "Activation script created: ${PROJECT_DIR}/activate_env.sh"
}

# Main setup function
main() {
    echo "🚀 Setting up ACBO training environment on GPU cluster"
    echo "=================================================="
    
    check_cluster_environment
    setup_cuda
    setup_project_directory
    setup_python_environment
    install_dependencies
    verify_installation
    create_activation_script
    
    log_success "Environment setup complete!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Sync your project code to ${PROJECT_DIR}"
    echo "2. Activate environment: source ${PROJECT_DIR}/activate_env.sh"
    echo "3. Test with: python scripts/collect_sft_dataset.py --size small"
    echo "4. Submit training jobs using the Slurm scripts"
    echo ""
    echo "💡 Tips:"
    echo "- Add 'source ${PROJECT_DIR}/activate_env.sh' to your jobs"
    echo "- Use 'squeue -u \$USER' to monitor job status"
    echo "- Check logs in ${PROJECT_DIR}/logs/"
}

# Run main function if script is executed (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi