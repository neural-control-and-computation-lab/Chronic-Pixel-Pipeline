@echo off
REM Run the full survey pipeline, end to end.
REM
REM Usage (from Anaconda Prompt with cpp env active):
REM   run_all.bat                    -- full pipeline (all six steps)
REM   run_all.bat sort               -- 1. sort only
REM   run_all.bat metrics            -- 2. precompute waveform quality only
REM   run_all.bat classify           -- 3. apply the neural/noise classifier only
REM   run_all.bat figures            -- 4. generate unit figures only
REM   run_all.bat track              -- 5. track only (assumes sorting done)
REM   run_all.bat aggregate          -- 6. aggregate figures + Dropbox sync only
REM
REM Prerequisites:
REM   1. Run setup_env.bat first to create the conda environment
REM   2. Set DATA_ROOT to your SpikeGLX data directory:
REM        set DATA_ROOT=E:\
REM   3. Survey directories should exist in DATA_ROOT
REM   4. Step 3 needs a trained model in DATA_ROOT\classifier_model\
REM      (see train_classifier.py)
REM
REM All outputs (sorting_output, figures, logs) are written inside DATA_ROOT.
REM The only thing that leaves DATA_ROOT is the explicit figure sync in step 6.
setlocal

if not defined DATA_ROOT set DATA_ROOT=%USERPROFILE%\data\TRODES_test

set LOG_DIR=%DATA_ROOT%\logs

echo ==================================================
echo  Chronic Pixel Pipeline - Full Run
echo  DATA_ROOT: %DATA_ROOT%
echo  Logs:      %LOG_DIR%
echo ==================================================

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set MODE=%1
if "%MODE%"=="" set MODE=all

if "%MODE%"=="sort" goto sort
if "%MODE%"=="all" goto sort
goto skip_sort

:sort
echo.
echo ==================================================
echo  Step 1: Spike sorting all surveys with KS4
echo ==================================================
powershell -Command "python sort_survey.py all 2>&1 | Tee-Object -FilePath '%LOG_DIR%\sort_all.log'"
if "%MODE%"=="sort" goto done

:skip_sort
if "%MODE%"=="metrics" goto metrics
if "%MODE%"=="all" goto metrics
goto skip_metrics

:metrics
echo.
echo ==================================================
echo  Step 2: Precompute waveform quality metrics
echo ==================================================
powershell -Command "python precompute_waveform_quality.py 2>&1 | Tee-Object -FilePath '%LOG_DIR%\precompute_quality.log'"
if "%MODE%"=="metrics" goto done

:skip_metrics
if "%MODE%"=="classify" goto classify
if "%MODE%"=="all" goto classify
goto skip_classify

:classify
echo.
echo ==================================================
echo  Step 3: Apply the neural/noise classifier
echo ==================================================
powershell -Command "python apply_custom_classifier.py all 2>&1 | Tee-Object -FilePath '%LOG_DIR%\apply_classifier.log'"
if "%MODE%"=="classify" goto done

:skip_classify
if "%MODE%"=="figures" goto figures
if "%MODE%"=="all" goto figures
goto skip_figures

:figures
echo.
echo ==================================================
echo  Step 4: Generate single-neuron figures
echo ==================================================
powershell -Command "python run_single_neuron_plots.py 2>&1 | Tee-Object -FilePath '%LOG_DIR%\single_neuron_plots.log'"
if "%MODE%"=="figures" goto done

:skip_figures
if "%MODE%"=="track" goto track
if "%MODE%"=="all" goto track
goto skip_track

:track
echo.
echo ==================================================
echo  Step 5: Cross-day neuron tracking with UnitMatch
echo ==================================================
powershell -Command "python track_neurons.py 2>&1 | Tee-Object -FilePath '%LOG_DIR%\track_neurons.log'"
if "%MODE%"=="track" goto done

:skip_track
if "%MODE%"=="aggregate" goto aggregate
if "%MODE%"=="all" goto aggregate
goto done

:aggregate
echo.
echo ==================================================
echo  Step 6: Aggregate tracking figures + Dropbox sync
echo ==================================================
powershell -Command "python regen_tracking_plots.py 2>&1 | Tee-Object -FilePath '%LOG_DIR%\regen_tracking_plots.log'"

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
