@echo off
title Lynx RF Quant Launcher
echo ========================================================
echo      Welcome to Lynx RF Quant (V30.41)
echo      Initializing Environment...
echo ========================================================

:: 1. 检查是否安装了依赖 (简单检查)
echo [1/2] Checking Dependencies...
pip install -r requirements.txt

:: 2. 启动 Streamlit
echo.
echo [2/2] Launching Strategy Monitor...
echo.
streamlit run app.py

pause