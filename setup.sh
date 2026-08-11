#!/bin/bash

echo "==========================================="
echo "Setting up Python Environment (Linux/macOS)"
echo "==========================================="

echo "1. Creating Python virtual environment..."
python3 -m venv venv

echo "2. Activating virtual environment..."
source venv/bin/activate

echo "3. Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==========================================="
echo "Setup complete! "
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo "==========================================="
