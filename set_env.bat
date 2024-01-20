@echo off
setlocal enabledelayedexpansion

set "PROJECT_ROOT=%CD%"
echo Set environment variable: PROJECT_ROOT=!PROJECT_ROOT!

echo Verify SUMO_HOME is defined:
if defined SUMO_HOME (
    echo SUMO_HOME is defined: !SUMO_HOME!
) else (
    echo ERROR - SUMO_HOME is not defined. need to install SUMO simulator and make sure that SUMO_HOME is defined.
    exit /b 1
)

echo set_env completed successfully!

endlocal