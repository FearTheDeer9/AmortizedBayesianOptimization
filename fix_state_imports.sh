#!/bin/bash

# Fix all imports of removed state module
echo "Fixing state module imports..."

# Files that import from ..acquisition.state
files_with_acquisition_state=$(grep -l "from \.\.acquisition\.state import AcquisitionState" src/causal_bayes_opt --include="*.py" -r | grep -v __pycache__)
for file in $files_with_acquisition_state; do
    echo "Fixing $file"
    sed -i '' 's/from \.\.acquisition\.state import AcquisitionState/from ..jax_native.state import TensorBackedAcquisitionState as AcquisitionState/g' "$file"
done

# Files that import from .state 
files_with_local_state=$(grep -l "from \.state import AcquisitionState" src/causal_bayes_opt --include="*.py" -r | grep -v __pycache__)
for file in $files_with_local_state; do
    echo "Fixing $file"
    sed -i '' 's/from \.state import AcquisitionState/from ..jax_native.state import TensorBackedAcquisitionState as AcquisitionState/g' "$file"
done

echo "Done fixing state imports"