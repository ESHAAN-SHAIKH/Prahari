@echo off
cd /d "%~dp0"

if not exist .venv (
    echo Creating .venv ...
    uv venv --python 3.11 .venv || python -m venv .venv
)

call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

echo.
echo ====================================================
echo PRAHARI starting on http://localhost:8000
echo ====================================================
if "%PRAHARI_CKPT%"=="" (
    echo No PRAHARI_CKPT set - running synthetic sensor model.
    echo The dashboard badges every frame as synthetic.
)
echo.

uvicorn backend.main:app --host 0.0.0.0 --port 8000
