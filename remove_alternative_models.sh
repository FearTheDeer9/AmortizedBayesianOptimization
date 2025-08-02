#!/bin/bash
# Script to remove alternative model architectures

set -e

echo "=========================================="
echo "REMOVING ALTERNATIVE MODEL ARCHITECTURES"
echo "=========================================="
echo ""

# Function to remove files/directories
remove_item() {
    local item=$1
    if [ -e "$item" ]; then
        rm -rf "$item"
        echo "  ✓ Removed $item"
    else
        echo "  ⚠ Not found: $item"
    fi
}

# 1. Remove old parent set models
echo "1. Removing old parent set models..."
cd src/causal_bayes_opt/avici_integration

# Remove old parent_set models
remove_item "parent_set/model.py"
remove_item "parent_set/jax_model.py"
remove_item "parent_set/mechanism_aware.py"
remove_item "parent_set/unified"

# Keep only the essential files in parent_set
echo "  Keeping essential parent_set files: encoding, enumeration, factory, inference, posterior"

# 2. Remove experimental continuous models
echo ""
echo "2. Removing experimental continuous models..."
remove_item "continuous/improved_model.py"
remove_item "continuous/relationship_aware_model.py"

# Keep the main continuous model
echo "  Keeping continuous/model.py (ContinuousParentSetPredictionModel)"

cd ../../..

# 3. Remove old acquisition models
echo ""
echo "3. Removing old acquisition models..."
cd src/causal_bayes_opt/acquisition

remove_item "enhanced_policy_network.py"
remove_item "enriched"
remove_item "state.py"
remove_item "state_enhanced.py"
remove_item "state_tensor_converter.py"
remove_item "state_tensor_ops.py"
remove_item "vectorized_attention.py"

# Keep the essential acquisition files
echo "  Keeping essential acquisition files: policy.py, grpo.py, rewards.py, etc."

cd ../../..

# 4. Check for any imports of removed models
echo ""
echo "4. Checking for imports of removed models..."
echo "  (This helps identify any files that need updating)"

# Check for imports of removed models
echo ""
echo "Checking for imports of ImprovedContinuousParentSetPredictionModel..."
grep -r "ImprovedContinuousParentSetPredictionModel" src/ || echo "  ✓ No imports found"

echo ""
echo "Checking for imports of RelationshipAwareParentSetModel..."
grep -r "RelationshipAwareParentSetModel" src/ || echo "  ✓ No imports found"

echo ""
echo "Checking for imports of removed parent_set models..."
grep -r "from.*parent_set.model import\|from.*parent_set.jax_model import\|from.*parent_set.mechanism_aware import\|from.*parent_set.unified" src/ || echo "  ✓ No imports found"

echo ""
echo "Checking for imports of removed acquisition models..."
grep -r "enhanced_policy_network\|enriched\|state_enhanced\|state_tensor_converter\|vectorized_attention" src/ | grep -v "Binary file" || echo "  ✓ No imports found"

# 5. Create a summary of what remains
echo ""
echo "=========================================="
echo "MODEL CLEANUP COMPLETE"
echo "=========================================="
echo ""
echo "Remaining model architectures:"
echo ""
echo "Surrogate Models:"
echo "  - ContinuousParentSetPredictionModel (continuous/model.py)"
echo ""
echo "Policy Models:"
echo "  - unified_policy (policies/unified_policy.py)"
echo ""
echo "Next steps:"
echo "  1. Test that model loading/training still works"
echo "  2. Commit these changes"
echo "  3. Move to Phase 4: Implement SCM management module"