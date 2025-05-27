@echo off
echo Creating virtual environment...
C:\Users\20232754\AppData\Local\Programs\Python\Python311\python.exe -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing requirements...
pip install -r requirements.txt

echo Setup complete! Virtual environment is ready to use.
pause 