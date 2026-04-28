@echo off
setlocal enabledelayedexpansion

REM ---------------------------------------------------------------------------
REM  match_json.bat
REM
REM  Run job_matcher.py and dump the raw JSON result to stdout.
REM  Auto-detects a missing index and builds it on first run.
REM
REM  Usage:
REM    match_json.bat
REM        - default JD: data\job_descriptions\01_senior_python_ml_engineer.txt
REM        - default top-k: from job_matcher.py (10)
REM
REM    match_json.bat data\job_descriptions\02_genai_rag_engineer.txt
REM        - custom JD
REM
REM    match_json.bat data\job_descriptions\02_genai_rag_engineer.txt --top-k 5 --use-llm
REM        - custom JD + any extra job_matcher.py flags
REM
REM    match_json.bat data\job_descriptions\02_genai_rag_engineer.txt --auto-filter --use-llm
REM        - LLM extracts must-haves from the JD, generates per-candidate
REM          reasoning, and produces a final shortlist recommendation.
REM          Requires OPENROUTER_API_KEY in env or in a .env file.
REM
REM  Save to a file (use --output to bypass PowerShell's UTF-16 redirect):
REM    match_json.bat data\job_descriptions\02_genai_rag_engineer.txt ^
REM        --auto-filter --use-llm --output result.json
REM ---------------------------------------------------------------------------

REM Always run from the directory this bat file lives in.
cd /d "%~dp0"

REM Make sure Python writes UTF-8 (rich/JSON with non-ASCII excerpts).
set "PYTHONIOENCODING=utf-8"

REM First positional argument is the JD path (with a sensible default).
set "JD=%~1"
if "%JD%"=="" set "JD=data\job_descriptions\01_senior_python_ml_engineer.txt"

if not exist "%JD%" (
    echo ERROR: Job description file not found: "%JD%" 1>&2
    exit /b 1
)

REM Build the index on first run if it isn't there yet.
if not exist "chroma_db\" (
    echo [match_json] chroma_db\ not found - building the index... 1>&2
    python resume_rag.py --build --json 1>&2
    if errorlevel 1 (
        echo ERROR: Failed to build the index. 1>&2
        exit /b 1
    )
)

REM Drop the JD (first arg) and forward any remaining args to job_matcher.py.
set "EXTRA="
:next_arg
shift
if "%~1"=="" goto :run
set "EXTRA=!EXTRA! %~1"
goto :next_arg

:run
python job_matcher.py --jd "%JD%" --json !EXTRA!
exit /b %errorlevel%
