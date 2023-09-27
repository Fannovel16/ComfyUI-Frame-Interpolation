@echo off

set "requirements_txt=%~dp0\requirements-no-cupy.txt"
set "python_exec=..\..\..\python_embeded\python.exe"

echo Installing ComfyUI Frame Interpolation..

if exist "%python_exec%" (
    echo Installing with ComfyUI Portable
    for /f "delims=" %%i in (%requirements_portable_txt%) do (
        %python_exec% -s -m pip install "%%i"
    )
    %python_exec% -s install-cupy.py
) else (
    echo Installing with system Python
    for /f "delims=" %%i in (%requirements_txt%) do (
        pip install "%%i"
    )
    python install-cupy.py
)

pause