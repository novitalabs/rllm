#!/bin/bash
set -e

source .env

# Define repos
REPOS=(
    "django/django"
    "sympy/sympy"
    "sphinx-doc/sphinx"
    "matplotlib/matplotlib"
    "scikit-learn/scikit-learn"
    "astropy/astropy"
    "pydata/xarray"
    "pytest-dev/pytest"
    "pylint-dev/pylint"
    "psf/requests"
    "mwaskom/seaborn"
    "pallets/flask"
)

# Create templates for each repo
for repo in "${REPOS[@]}"; do
    name=$(echo "$repo" | tr '/' '-')
    dir="/root/work/rft-tinker/ppio-templates/$name"
    
    echo "========================================"
    echo "Creating template for: $repo"
    echo "========================================"
    
    mkdir -p "$dir"
    cd "$dir"
    
    # Create Dockerfile
    cat > ppio.Dockerfile << EOF
FROM image.ppinfra.com/sandbox/code-interpreter:latest

# Install common dependencies
RUN pip install pytest numpy --quiet

# Clone repository with full history
RUN git clone https://github.com/$repo.git /testbed

WORKDIR /testbed

# Try to install in editable mode (may fail for some versions, that's ok)
RUN pip install -e . --quiet 2>/dev/null || echo "Install skipped - will install at runtime"
EOF

    echo "Dockerfile created at $dir/ppio.Dockerfile"
done

echo ""
echo "All Dockerfiles created. Ready to build."
echo "To build a specific template:"
echo "  cd /root/work/rft-tinker/ppio-templates/<repo-name>"
echo "  ppio-sandbox-cli template build --name swebench-<repo-name>"
