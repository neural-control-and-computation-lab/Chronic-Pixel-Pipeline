@echo off
REM Run the full pipeline: sort all 4 surveys, then track neurons across days.
REM
REM Usage (from Anaconda Prompt with cpp env active):
REM   run_all.bat                    -- sort all + track
REM   run_all.bat sort               -- sort only
REM   run_all.bat track              -- track only (assumes sorting done)
REM
REM Prerequisites:
REM   1. Run setup_env.bat first to create the conda environment
REM   2. Set DATA_ROOT to your SpikeGLX data directory:
REM        set DATA_ROOT=C:\Users\YourName\data\TRODES_test
REM   3. Copy the 4 survey directories into DATA_ROOT
REM
REM All outputs (sorting_output, figures, logs) are written inside DATA_ROOT
REM so nothing goes through Dropbox or other cloud sync.
setlocal

if not defined DATA_ROOT set DATA_ROOT=%USERPROFILE%\data\TRODES_test

set LOG_DIR=%DATA_ROOT%\logs

echo ==================================================
echo  Chronic Pixel Pipeline - Full Run
echo  DATA_ROOT: %DATA_ROOT%
echo  Logs:      %LOG_DIR%
echo ==================================================

REM Check data exists
for %%d in (
    "SvyPrb_2026-02-26T11.58.07_g0"
    "SvyPrb_2026-03-02T14.10.26_g0"
    "SvyPrb_2026-03-04T10.17.50_g0"
    "SvyPrb_2026-03-06T10.01.15_g0"
) do (
    if not exist "%DATA_ROOT%\%%~d" (
        echo WARNING: Missing data directory: %DATA_ROOT%\%%~d
    )
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set MODE=%1
if "%MODE%"=="" set MODE=all

if "%MODE%"=="sort" goto sort
if "%MODE%"=="all" goto sort
goto skip_sort

:sort
echo.
echo ==================================================
echo  Step 1: Spike sorting all 4 surveys with KS4
echo ==================================================
powershell -Command "python sort_survey.py all 2>&1 | Tee-Object -FilePath '%LOG_DIR%\sort_all.log'"
if "%MODE%"=="sort" goto done

:skip_sort
if "%MODE%"=="track" goto track
if "%MODE%"=="all" goto track
goto done

:track
echo.
echo ==================================================
echo  Step 2: Cross-day neuron tracking with UnitMatch
echo ==================================================
powershell -Command "python track_neurons.py 20260226 20260302 20260304 20260306 2>&1 | Tee-Object -FilePath '%LOG_DIR%\track_neurons.log'"

:done
echo.
echo ==================================================
echo  Pipeline complete!
echo ==================================================
echo Outputs (all inside DATA_ROOT):
echo   Sorting:  %DATA_ROOT%\sorting_output\survey_*\
echo   Tracking: %DATA_ROOT%\sorting_output\tracking\
echo   Figures:  %DATA_ROOT%\figures\sorting\
echo   Logs:     %LOG_DIR%\

endlocal
