@echo off

set "PROJECT_ROOT=%CD%"
echo Set environment variable: PROJECT_ROOT=%PROJECT_ROOT%

set "PROJECT_TARGET=%CD%\target"
echo Set environment variable: PROJECT_TARGET=%PROJECT_TARGET%


if not exist "%PROJECT_TARGET%" (
    echo create target dir : %PROJECT_TARGET%
    mkdir "%PROJECT_TARGET%"
)

echo Verify SUMO_HOME is defined:
if defined SUMO_HOME (
    echo SUMO_HOME is defined: %SUMO_HOME%
) else (
    echo ERROR - SUMO_HOME is not defined. need to install SUMO simulator and make sure that SUMO_HOME is defined.
    exit /b 1
)

echo set_env completed successfully!
