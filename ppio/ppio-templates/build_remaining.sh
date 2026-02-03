#!/bin/bash
source .env

DIRS=(
    "psf-requests"
    "pytest-dev-pytest"
    "pylint-dev-pylint"
    "django-django"
    "sympy-sympy"
    "sphinx-doc-sphinx"
    "matplotlib-matplotlib"
    "scikit-learn-scikit-learn"
    "astropy-astropy"
    "pydata-xarray"
    "mwaskom-seaborn"
)

for dir in "${DIRS[@]}"; do
    name="swebench-${dir}"
    echo "========================================"
    echo "Building: $name"
    echo "Time: $(date)"
    echo "========================================"
    
    cd "/root/work/rft-tinker/ppio-templates/$dir"
    ppio-sandbox-cli template build --name "$name" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: $name"
    else
        echo "FAILED: $name"
    fi
    echo ""
done

echo "All builds completed at $(date)"
