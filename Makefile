.PHONY: venv install download-data clean help

PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python

help:
	@echo "Available targets:"
	@echo "  make venv         - Create virtual environment"
	@echo "  make install      - Install dependencies"
	@echo "  make download-data - Download/refresh raw data from sources"
	@echo "  make clean        - Clean cache and temporary files"

venv:
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created. Run 'make install' to install dependencies."

install: venv
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."

download-data:
	@echo "Downloading data from sources..."
	$(PYTHON_VENV) -c "from src.data.fred_loader import load_all_fred_data; from src.config import FRED_SERIES; import os; from dotenv import load_dotenv; load_dotenv(); api_key = os.getenv('FRED_API_KEY'); load_all_fred_data(FRED_SERIES, api_key) if api_key else print('ERROR: FRED_API_KEY not found in .env file')"
	@echo "Data download complete. Check data/raw/ for downloaded files."

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "Clean complete."

