@echo off
::set HSA_OVERRIDE_GFX_VERSION=11.5.1
::set CUDA_VISIBLE_DEVICES=0
::set HIP_VISIBLE_DEVICES=1
:: enable for AMD, leave commented for NVIDIA

::set AMD_SERIALIZE_KERNEL=1 -> crash log
::set AMD_LOG_LEVEL=3 -> logging
:: logging for AMD ROCm

::REM Activate the virtual environment
call .\venv\Scripts\activate.bat

::REM Run Streamlit
streamlit run main.py