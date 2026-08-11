@echo off
echo ===========================================
echo Setting up Python Environment (Windows)
echo ===========================================

echo 1. Creating Python virtual environment...
python -m venv venv

echo 2. Activating virtual environment...
call venv\Scripts\activate.bat

echo 3. Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ===========================================
echo Setup complete! 
echo To activate the environment, run:
echo venv\Scripts\activate
echo ===========================================
