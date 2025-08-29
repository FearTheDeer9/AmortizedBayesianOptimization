#!/bin/bash
# GPU Environment Setup for JAX Training on Imperial HPC

echo "=== Loading HPC GPU Environment ==="
module purge
module load tools/prod
module load jax/0.4.25-gfbf-2023a-CUDA-12.1.1
module load pydantic/2.5.3-GCCcore-12.3.0
module load dm-haiku/0.0.12-foss-2023a-CUDA-12.1.1
module load scikit-learn/1.3.1-gfbf-2023a

echo "=== Environment Ready ==="
module list
echo ""
python3 -c "import jax; print('JAX devices:', jax.devices())"
