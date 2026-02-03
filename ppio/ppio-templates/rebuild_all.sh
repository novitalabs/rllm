#!/bin/bash
set -e

source .env

# List of template directories to rebuild
TEMPLATES=(
    "astropy-astropy"
    "django-django"
    "matplotlib-matplotlib"
    "mwaskom-seaborn"
    "pallets-flask"
    "psf-requests"
    "pydata-xarray"
    "pylint-dev-pylint"
    "pytest-dev-pytest"
    "scikit-learn-scikit-learn"
    "sphinx-doc-sphinx"
    "sympy-sympy"
)

cd /root/work/rft-tinker/ppio-templates

for template in "${TEMPLATES[@]}"; do
    echo "========================================"
    echo "Rebuilding: swebench-$template"
    echo "========================================"
    cd "$template"
    ppio-sandbox-cli template build --name "swebench-$template"
    cd ..
    echo "Done: $template"
done

echo ""
echo "All templates rebuilt!"
