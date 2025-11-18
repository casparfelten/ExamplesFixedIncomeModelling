#!/bin/bash
# Helper script to create .env file from template
if [ ! -f .env ]; then
    echo "FRED_API_KEY=your_fred_api_key_here" > .env
    echo "Created .env file. Please edit it and add your FRED API key."
else
    echo ".env file already exists."
fi
