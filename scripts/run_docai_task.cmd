@echo off
rem Unattended Document AI evaluation chain, started by Windows Task Scheduler (task "DocAI_eval_queue").
rem run_queue.py waits until the shared ..\.llm_lock has been free for 10 minutes, so this starts only
rem after the RAG evaluation queue has finished. Prompt v1 is kept frozen (no dev tuning unattended);
rem the approve threshold stays at the default 1.0. Every eval caches per receipt and resumes.
title Document AI evaluation queue - do not close
cd /d "%~dp0.."
set LOG=results\logs_docai_task.out
echo started %date% %time% >> %LOG%
".venv\Scripts\python.exe" scripts\run_queue.py --jobs ocr_dev ocr_test regex dev_7b >> %LOG% 2>&1
".venv\Scripts\python.exe" scripts\score.py --split dev >> %LOG% 2>&1
".venv\Scripts\python.exe" scripts\error_analysis.py --split dev >> %LOG% 2>&1
".venv\Scripts\python.exe" scripts\run_queue.py --jobs test_7b test_3b gold_test_7b >> %LOG% 2>&1
".venv\Scripts\python.exe" scripts\score.py --split test >> %LOG% 2>&1
".venv\Scripts\python.exe" scripts\error_analysis.py --split test >> %LOG% 2>&1
echo pulling vision model >> %LOG%
ollama pull qwen2.5vl:3b >> %LOG% 2>&1
".venv\Scripts\python.exe" scripts\run_queue.py --jobs vision_test_30 >> %LOG% 2>&1
".venv\Scripts\python.exe" scripts\score.py --split test >> %LOG% 2>&1
echo finished %date% %time% >> %LOG%
