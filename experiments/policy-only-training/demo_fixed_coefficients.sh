#!/bin/bash
# Demonstration script for fixed coefficient SCM convergence testing

echo "========================================="
echo "Fixed Coefficient SCM Convergence Demo"
echo "========================================="
echo ""

# Change to project root
cd ../..

# 1. Test small fork structure with fixed coefficients
echo "1. Testing 4-variable fork with fixed coefficients (zero noise):"
echo "   Command: python train_grpo_single_scm_with_surrogate.py --scm-type fork --num-vars 4 --fixed-coefficients --interventions 50"
echo ""

# 2. Test larger scale-free network
echo "2. Testing 20-variable scale-free network with decreasing coefficients:"
echo "   Command: python train_grpo_single_scm_with_surrogate.py --scm-type scale_free --num-vars 20 --fixed-coefficients --coefficient-pattern decreasing --interventions 100"
echo ""

# 3. Compare coefficient patterns
echo "3. Comparing different coefficient patterns on mixed structure:"
echo "   - Decreasing: [2.0, 1.5, 1.0, 0.5, 0.3]"
echo "   - Alternating: [2.0, -1.5, 1.0, -0.5]"
echo "   - Strong: [2.0, 2.0, 2.0, 2.0]"
echo "   - Mixed: [2.0, 0.5, 1.5, 0.3, 1.0]"
echo ""

# 4. Run comprehensive test suite
echo "4. Run comprehensive convergence test suite:"
echo "   Command: python run_comprehensive_convergence_tests.py --test all --interventions 100"
echo ""

echo "Available options:"
echo "  --fixed-coefficients: Use deterministic coefficients (zero noise)"
echo "  --coefficient-pattern: Choose pattern (decreasing, alternating, strong, mixed)"
echo "  --num-vars: Number of variables (supports any size)"
echo "  --scm-type: Graph structure (fork, chain, scale_free, random, etc.)"
echo ""

echo "Key advantages of this approach:"
echo "  ✅ Scalable to any graph size (4 to 100+ variables)"
echo "  ✅ Deterministic (zero noise) for reproducible results"
echo "  ✅ Preserves original graph generation logic"
echo "  ✅ Flexible coefficient patterns"
echo "  ✅ Easy comparison across structures and sizes"