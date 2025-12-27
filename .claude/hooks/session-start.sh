#!/bin/bash
# Session start hook for difflow development

echo "🧪 difflow - Differentiable Flowsheet Framework"
echo "================================================"

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    # Check for common venv locations
    if [[ -d ".venv" ]]; then
        echo "📦 Found .venv - activate with: source .venv/bin/activate"
    elif [[ -d "venv" ]]; then
        echo "📦 Found venv - activate with: source venv/bin/activate"
    else
        echo "⚠️  No virtual environment found. Create one with:"
        echo "   python -m venv .venv && source .venv/bin/activate"
        echo "   pip install -e '.[dev,examples]'"
    fi
else
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
fi

# Check if package is installed
python -c "import difflow" 2>/dev/null
if [[ $? -eq 0 ]]; then
    echo "✅ difflow is installed"
else
    echo "⚠️  difflow not installed. Install with: pip install -e '.[dev,examples]'"
fi

# Check JAX backend
python -c "import jax; print(f'✅ JAX backend: {jax.default_backend()}')" 2>/dev/null

# Show available make commands
echo ""
echo "📋 Available commands:"
echo "   make test      - Run pytest"
echo "   make book      - Build documentation"
echo "   make notebooks - Execute example notebooks"
echo ""
echo "🔧 Slash commands:"
echo "   /test          - Run all tests"
echo "   /test-unit     - Run specific unit tests"
echo "   /lint          - Check code style"
echo "   /example       - Run an example notebook"
