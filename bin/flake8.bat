@echo off
set current=%~dp0
set root=%current%..
cd %root%
set pythonexe=python

@echo running 'python -m flake8 td3a_cpp tests examples'
%pythonexe% -m flake8 td3a_cpp tests examples setup.py doc/conf.py

if %errorlevel% neq 0 exit /b %errorlevel%
@echo Done Testing.