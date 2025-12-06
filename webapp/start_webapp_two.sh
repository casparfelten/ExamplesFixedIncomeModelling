#!/bin/bash
# Start the QSIG Macro Graph Explorer Flask app (Webapp Two)

echo "🚀 Starting QSIG Macro Graph Explorer (Webapp Two - Flask Edition)..."
echo "============================================================"

# Activate virtual environment
source ../webapp_env/bin/activate

# Verify dependencies
echo "📦 Checking dependencies..."
python -c "import flask; print('✅ Flask found')" || {
    echo "❌ Missing Flask. Installing..."
    pip install flask
}

# Start Flask app
echo "🌐 Starting Flask webapp on http://localhost:5000"
echo "   Clean minimal interface"
echo "   Use Ctrl+C to stop the server"
echo ""

python app_flask.py