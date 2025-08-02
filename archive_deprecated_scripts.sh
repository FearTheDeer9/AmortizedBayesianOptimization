#!/bin/bash
# Script to archive deprecated files systematically

set -e

echo "=========================================="
echo "ARCHIVING DEPRECATED SCRIPTS"
echo "=========================================="
echo ""

ARCHIVE_DIR="archive/deprecated_2024"

# Create subdirectories in archive
mkdir -p "$ARCHIVE_DIR/scripts"
mkdir -p "$ARCHIVE_DIR/examples"
mkdir -p "$ARCHIVE_DIR/experiments"

# Function to move files
move_files() {
    local pattern=$1
    local dest=$2
    local count=0
    
    for file in $pattern; do
        if [ -f "$file" ]; then
            mv "$file" "$dest/"
            ((count++))
        fi
    done
    
    echo "  Moved $count files matching $pattern"
}

# Archive scripts
echo "1. Archiving deprecated scripts..."
cd scripts

# Test scripts
move_files "test_*.py" "../$ARCHIVE_DIR/scripts"
move_files "debug_*.py" "../$ARCHIVE_DIR/scripts"
move_files "validate_*.py" "../$ARCHIVE_DIR/scripts"
move_files "verify_*.py" "../$ARCHIVE_DIR/scripts"
move_files "diagnose_*.py" "../$ARCHIVE_DIR/scripts"
move_files "investigate_*.py" "../$ARCHIVE_DIR/scripts"
move_files "analyze_*.py" "../$ARCHIVE_DIR/scripts"
move_files "inspect_*.py" "../$ARCHIVE_DIR/scripts"

# Old demo/run scripts
move_files "run_simple_demo.py" "../$ARCHIVE_DIR/scripts"
move_files "run_simplified_demo.py" "../$ARCHIVE_DIR/scripts"
move_files "run_full_acbo_demo.py" "../$ARCHIVE_DIR/scripts"
move_files "run_3min_validation.py" "../$ARCHIVE_DIR/scripts"
move_files "run_bc_from_checkpoint.py" "../$ARCHIVE_DIR/scripts"
move_files "run_grpo_workflow.py" "../$ARCHIVE_DIR/scripts"

# Deprecated training scripts
move_files "train_enriched_*.py" "../$ARCHIVE_DIR/scripts"
move_files "train_acquisition_*.py" "../$ARCHIVE_DIR/scripts"
move_files "train_full_scale_grpo.py" "../$ARCHIVE_DIR/scripts"
move_files "enriched_policy_with_learning.py" "../$ARCHIVE_DIR/scripts"

# Other deprecated scripts
move_files "pipeline_train_and_evaluate.py" "../$ARCHIVE_DIR/scripts"
move_files "simple_*.py" "../$ARCHIVE_DIR/scripts"
move_files "quick_*.py" "../$ARCHIVE_DIR/scripts"
move_files "plot_acbo_trajectory_comparison.py" "../$ARCHIVE_DIR/scripts"
move_files "visualize_*.py" "../$ARCHIVE_DIR/scripts"
move_files "compare_grpo_models.py" "../$ARCHIVE_DIR/scripts"
move_files "unified_pipeline.py" "../$ARCHIVE_DIR/scripts"

# Keep evaluate_acbo_methods.py for now (will rename v2 later)

cd ..

# Archive examples
echo ""
echo "2. Archiving deprecated examples..."
cd examples

# Move development folder
if [ -d "development" ]; then
    mv development "../$ARCHIVE_DIR/examples/"
    echo "  Moved development/ folder"
fi

# Move advanced folder
if [ -d "advanced" ]; then
    mv advanced "../$ARCHIVE_DIR/examples/"
    echo "  Moved advanced/ folder"
fi

# Move test files
move_files "test_*.py" "../$ARCHIVE_DIR/examples"

# Move old demos
move_files "demo_*.py" "../$ARCHIVE_DIR/examples"
move_files "bc_training_comparison.py" "../$ARCHIVE_DIR/examples"
move_files "clean_acbo_demo.py" "../$ARCHIVE_DIR/examples"
move_files "complete_workflow_demo.py" "../$ARCHIVE_DIR/examples"
move_files "jax_native_demo.py" "../$ARCHIVE_DIR/examples"
move_files "parent_scale_demo.py" "../$ARCHIVE_DIR/examples"
move_files "universal_acbo_demo.py" "../$ARCHIVE_DIR/examples"
move_files "verify_intervention_strategies.py" "../$ARCHIVE_DIR/examples"
move_files "train_grpo_with_fixes.py" "../$ARCHIVE_DIR/examples"

cd ..

# Archive experiments
echo ""
echo "3. Archiving deprecated experiments..."
cd experiments

# Move deprecated notebooks folder
if [ -d "deprecated_notebooks" ]; then
    mv deprecated_notebooks "../$ARCHIVE_DIR/experiments/"
    echo "  Moved deprecated_notebooks/ folder"
fi

# Move old notebooks
move_files "*.ipynb" "../$ARCHIVE_DIR/experiments"

# Keep Python files in experiments for now (they might be used)

cd ..

# Now rename evaluate_acbo_methods.py to evaluate_acbo_methods.py
echo ""
echo "4. Renaming evaluation script..."
if [ -f "scripts/evaluate_acbo_methods.py" ]; then
    # First archive the old v1
    if [ -f "scripts/evaluate_acbo_methods.py" ]; then
        mv scripts/evaluate_acbo_methods.py "$ARCHIVE_DIR/scripts/"
        echo "  Archived old evaluate_acbo_methods.py"
    fi
    
    # Then rename v2
    mv scripts/evaluate_acbo_methods.py scripts/evaluate_acbo_methods.py
    echo "  Renamed evaluate_acbo_methods.py to evaluate_acbo_methods.py"
fi

# Update shell scripts to use new name
echo ""
echo "5. Updating references in shell scripts..."
for script in *.sh evaluate_comprehensive.sh; do
    if [ -f "$script" ]; then
        sed -i '' 's/evaluate_acbo_methods_v2\.py/evaluate_acbo_methods.py/g' "$script"
        echo "  Updated $script"
    fi
done

echo ""
echo "=========================================="
echo "ARCHIVING COMPLETE"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Deprecated scripts moved to: $ARCHIVE_DIR/scripts/"
echo "  - Deprecated examples moved to: $ARCHIVE_DIR/examples/"
echo "  - Deprecated experiments moved to: $ARCHIVE_DIR/experiments/"
echo "  - evaluate_acbo_methods.py renamed to evaluate_acbo_methods.py"
echo ""
echo "Next steps:"
echo "  1. Test that core functionality still works"
echo "  2. Commit these changes"
echo "  3. Move to Phase 3: Remove alternative model architectures"