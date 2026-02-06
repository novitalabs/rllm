#!/bin/bash
# Build PPIO Sandbox templates for R2E-Gym training repositories
#
# Usage:
#   ./build_r2e_templates.sh [repo_name]
#
# Examples:
#   ./build_r2e_templates.sh pandas     # Build only pandas template
#   ./build_r2e_templates.sh all        # Build all templates
#   ./build_r2e_templates.sh            # Build all templates (default)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."

    if ! command -v ppio-sandbox-cli &> /dev/null; then
        echo "Error: ppio-sandbox-cli not found. Install with: npm i -g ppio-sandbox-cli"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo "Error: Docker is not running"
        exit 1
    fi

    if [ -z "$PPIO_ACCESS_TOKEN" ]; then
        if [ -n "$PPIO_API_KEY" ]; then
            export PPIO_ACCESS_TOKEN="$PPIO_API_KEY"
        else
            echo "Error: PPIO_ACCESS_TOKEN or PPIO_API_KEY not set"
            exit 1
        fi
    fi

    echo "Prerequisites OK"
}

# Build a single template
build_template() {
    local repo_name=$1
    local template_dir="$SCRIPT_DIR/r2e-$repo_name"

    echo ""
    echo "========================================"
    echo "Building template: r2e-$repo_name"
    echo "========================================"

    if [ ! -d "$template_dir" ]; then
        echo "Error: Template directory not found: $template_dir"
        return 1
    fi

    cd "$template_dir"

    # Build with proxy if available
    if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
        echo "Building with proxy..."
        ppio-sandbox-cli tpl build \
            --build-arg HTTP_PROXY="${HTTP_PROXY:-$HTTPS_PROXY}" \
            --build-arg HTTPS_PROXY="${HTTPS_PROXY:-$HTTP_PROXY}"
    else
        echo "Building without proxy..."
        ppio-sandbox-cli tpl build
    fi

    echo "Template r2e-$repo_name built successfully!"
    cd "$SCRIPT_DIR"
}

# List all available templates
list_templates() {
    echo ""
    echo "Available R2E-Gym templates:"
    echo "----------------------------"
    for dir in "$SCRIPT_DIR"/r2e-*/; do
        if [ -d "$dir" ]; then
            name=$(basename "$dir")
            echo "  - ${name#r2e-}"
        fi
    done
}

# Main
main() {
    check_prerequisites

    local target=${1:-all}

    if [ "$target" = "list" ]; then
        list_templates
        return 0
    fi

    if [ "$target" = "all" ]; then
        echo "Building all R2E-Gym templates..."
        for dir in "$SCRIPT_DIR"/r2e-*/; do
            if [ -d "$dir" ]; then
                name=$(basename "$dir")
                build_template "${name#r2e-}" || echo "Warning: Failed to build ${name#r2e-}"
            fi
        done
    else
        build_template "$target"
    fi

    echo ""
    echo "========================================"
    echo "Build complete! List templates with:"
    echo "  ppio-sandbox-cli tpl list"
    echo "========================================"
}

main "$@"
