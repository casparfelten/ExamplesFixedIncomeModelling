#!/bin/bash
# Start the QSIG Macro Graph Explorer webapp (Dash)

echo "🚀 Starting QSIG Macro Graph Explorer (Professional Edition)..."
echo "============================================================"

# Activate virtual environment
source ../webapp_env/bin/activate

# Verify dependencies
echo "📦 Checking dependencies..."
python -c "import dash, dash_cytoscape; print('✅ All dependencies found')" || {
    echo "❌ Missing dependencies. Installing..."
    pip install dash dash-bootstrap-components dash-cytoscape
}

# Start Dash app
echo "🌐 Starting webapp on http://localhost:8050"
echo "   Professional Financial Interface"
echo "   Use Ctrl+C to stop the server"
echo ""

python app.py