@echo off
cd /d "%~dp0"

echo ================================
echo        rememberMe Launcher
echo ================================
echo.
echo  [1] General mode  - recognize faces from any video or webcam
echo  [2] Classroom mode - attendance + behavior analysis
echo  [3] Exit
echo.
set /p choice="Select option (1/2/3): "

if "%choice%"=="1" goto general
if "%choice%"=="2" goto classroom
if "%choice%"=="3" exit /b
echo Invalid choice. Please enter 1, 2, or 3.
goto :eof

:general
echo.
echo  [1] Use webcam
echo  [2] Use a video file
echo  [3] Use RTSP stream
echo.
set /p src="Select source (1/2/3): "

if "%src%"=="1" (
    python main.py --general --mode webcam %*
    goto done
)
if "%src%"=="2" (
    set /p filepath="Enter video file path: "
    python main.py --general --mode file --source "%filepath%" %*
    goto done
)
if "%src%"=="3" (
    set /p rtsp="Enter RTSP URL: "
    python main.py --general --mode rtsp --source "%rtsp%" %*
    goto done
)
echo Invalid source choice.
goto done

:classroom
python main.py --config config.json %*

:done
echo.
pause
