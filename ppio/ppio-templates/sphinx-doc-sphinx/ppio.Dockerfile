FROM image.ppinfra.com/sandbox/code-interpreter:latest

# Install common dependencies
RUN pip install pytest numpy --quiet

# Clone repository with full history
RUN git clone https://github.com/sphinx-doc/sphinx.git /testbed

WORKDIR /testbed

# Try to install in editable mode (may fail for some versions, that's ok)
RUN pip install -e . --quiet 2>/dev/null || echo "Install skipped - will install at runtime"
# Fix permissions for sandbox user
RUN chmod -R 777 /testbed
