# PROGRESS (resume file)

**STATUS: COMPLETE (2026-09-30).** All evaluations ran, scored, and are written up in README.md. Nothing remains to do.

All commands run from this folder in PowerShell, using the project venv: `.venv\Scripts\python ...`

## Lock rule (coordinator, 2026-09-29)
`..\.llm_lock` covers ALL CPU-heavy batch jobs, not only Ollama: batch OCR (`scripts\run_ocr.py`),
batch LLM evals, vision runs, model pulls > 1 GB. Run them only while holding the lock (RAG has priority; poll).
Without the lock: code, pytest, the regex baseline on already-OCR'd receipts, the app skeleton, and
OCR/LLM smoke tests on a handful of receipts (`eval_extract.py --smoke`).
onnxruntime threads are capped by `config.yaml` `ocr.threads: 4`, so OCR latency is measured consistently.
**Smoke-test lesson:** a 7B smoke test while RAG was running made Ollama RELOAD qwen2.5:7b (load 110 s),
because our num_ctx (4096) differs from RAG's (16384). That also cost RAG a reload. Do NOT smoke-test a model
RAG has loaded while RAG is running. Free RAM was 2.9 GB then, too little to load a second model safely.

## Status
- [x] 1. venv (Python 3.13), pinned requirements.txt. Data: `.venv\Scripts\python scripts\prepare_data.py`
      CORD v2 (HF naver-clova-ix/cord-v2 @ 7f0115a4, licence CC BY 4.0 from the dataset card).
      dev = CORD validation (100), test = CORD test (100). Gold in data/gold/{dev,test}.jsonl (mapping: src/schema.py).
      CORD v2 has no store-name label, so store_name is extracted but not scored.
- [x] 2. OCR code (src/ocr.py: RapidOCR 3.9.2, PP-OCR ONNX; deskew + y-overlap line grouping; PIL input
      because RapidOCR treats numpy arrays as BGR). Smoke-tested: ~6-10 s/receipt under contention.
      BATCH OCR NOT DONE: stopped on coordinator request. The 80 preliminary dev rows (default threads) are archived
      in data/ocr_prelim/ and used only to develop the regex baseline. data/ocr/ is empty and needs the locked re-run.
- [x] 3. Schema, normalisation, rules, confidence, routing (src/schema.py, normalize.py, validation.py).
      Tax-inclusive rule variants were added after checking the rules on dev gold. Gold auto-approve (routing ceiling):
      dev 90%, test 86%. pytest: 69 tests passing (`.venv\Scripts\python -m pytest -q`).
- [x] 4. Regex baseline (src/baseline.py), tuned on the prelim dev OCR only, then frozen.
      Prelim dev (80 receipts): total 0.888, subtotal 0.888, tax 0.925, item F1 0.726, exact 0.375 (results/prelim/).
      NOT yet run on the final OCR, dev or test (it is instant once OCR exists).
- [x] 5. Extraction pipeline (src/extract.py prompt v1, src/llm.py, JSON schema in `format`, temp 0, receipt text wrapped
      as data). Smoke test OK (dev validation_003, 7B: all fields correct, auto-approved). Only issue seen: the gold has
      tax=0 ("PB1: 0") and the model returned null, a candidate prompt rule for v2 ("a printed 0 is 0, not null").
- [x] Eval tooling: scripts/eval_extract.py (resumable JSONL per receipt, takes the lock, --smoke), scripts/score.py
      (metrics, routing curve, subset table, latency), scripts/error_analysis.py (OCR / extraction / normalisation buckets),
      scripts/run_queue.py (runs jobs in order; each job waits until the lock has been free 10 min, then takes it).
      Tracing (data/traces.db) and review queue (data/reviews.db) in src/tracing.py. src/lock.py = shared lock helper.
- [x] 8a. Streamlit app written (streamlit_app.py: Extract / Review queue / Evaluation / Monitoring). Boot check done;
      Evaluation and Review pages render. Final verification needs test results.
- [x] 9a. README drafted: static sections done, results sections marked "not run yet".

## Remaining (all need the lock except where noted)
1. OCR, then the regex baseline (~25 min):
   `.venv\Scripts\python scripts\run_queue.py --jobs ocr_dev ocr_test regex`
2. Dev tuning, 7B (~45-60 s/receipt when idle, so ~75 min):
   `.venv\Scripts\python scripts\run_queue.py --jobs dev_7b`, then `scripts\score.py --split dev` and
   `scripts\error_analysis.py --split dev`.
   Look at the errors; if the prompt changes, bump PROMPT_VERSION in src/extract.py (new cache file) and re-run dev.
   Pick `validation.approve_threshold` from results/routing_curve_dev.csv (default 1.0 = all rules pass, nothing unverified).
   Record the choice here.
3. Test runs (one at a time): `--jobs test_7b test_3b` (~75 + ~35 min). Optional upper bound: `gold_test_7b`.
4. Vision: check free RAM, then pull under the lock (`ollama pull qwen2.5vl:3b`, ~3.2 GB).
   Time 2-3 receipts first with `eval_extract.py --split test --system vision --limit 3`. Then choose N so the run
   fits the CPU budget: `--jobs vision_test_N`. The score.py subset table compares all systems on the same N receipts.
5. `scripts\score.py --split test`; `scripts\error_analysis.py --split test`.
6. App verification (lock not needed for screenshots, but Extract with an LLM is a single request):
   `.venv\Scripts\streamlit run streamlit_app.py --server.port 8611`; screenshot every page with
   msedge --headless (see CONVENTIONS); measure the streamlit process RSS (psutil) and Ollama model memory
   (`curl localhost:11434/api/ps`); write results/memory.json; stop the server.
7. Fill in the README results, error analysis and "Runs on an 8 GB laptop" sections, with test numbers only.

## 2026-09-30 10:43: unattended eval chain started (owner request)
- Windows Task Scheduler task "DocAI_eval_queue" -> scripts\run_docai_task.cmd (log results\logs_docai_task.out). It waits for RAG to release ..\.llm_lock, then runs: ocr_dev, ocr_test, regex, dev_7b, score/error_analysis dev, test_7b, test_3b, gold_test_7b, score/error_analysis test, pulls qwen2.5vl:3b, runs vision_test_30, and re-scores test.
- Unattended decisions: prompt v1 kept FROZEN (no dev tuning without review; the candidate tax=0 fix was NOT applied). approve_threshold stays at the default 1.0. Pick a threshold later from results/routing_curve_dev.csv (dev only) and re-score test. Record both in the README.
- If it stops: clear a stale ..\.llm_lock and run `schtasks /Run /TN "DocAI_eval_queue"`; every eval resumes from its per-receipt cache. Delete the task when finished: `schtasks /Delete /TN "DocAI_eval_queue" /F`.

## 2026-09-30 22:40-23:10: finished (all done)
- The eval chain finished at 21:35 (log results/logs_docai_task.out). The scheduled task "DocAI_eval_queue" was already deleted.
- Every test number in the README was checked against results/metrics_test.csv and metrics_test_subset.csv.
- **Approve threshold: kept 1.0, chosen on dev only** (results/routing_curve_dev.csv). 0.8 would add 6 dev approvals, 4 exact and 2 wrong, at the same precision (0.710 vs 0.714). That is not clearly better, and lower thresholds drop sharply.
  Re-scored with `scripts\score.py --split test --threshold 1.0`: metrics_test.csv is byte-identical.
- **Error analysis written into the README** from results/error_analysis_test.json and errors_test.csv. Scalars: 10 OCR, 37 extraction (29 missed, mostly subtotal/tax; 2 wrong value; 6 spurious, mostly CORD label gaps), 0 normalisation. Items: 18 OCR, 15 missed, 18 wrong line total, 39 extra.
- **App verified.**
  - Ran `streamlit run streamlit_app.py --server.port 8611`, made one Extract request (test_000, 7B; LLM 54.7 s; routed to review because line_math failed), and screenshotted all 4 pages through headless Edge (CDP, real-time wait).
  - Fixes: clipped chart legends; routing chart squashed (legend removed, caption added); item money now formatted like the scalars; the per-receipt browser defaults to the main system; a subset caption; deep link `/?sample=<id>&run=1`.
  - Note: the default page lives at `/`, so `/extract` shows a "Page not found" toast. Use `/`.
  - App RSS 169 MB (private 492 MB); OCR engine +111 MB; qwen2.5:7b 4.71 GiB in Ollama (results/memory.json). Server stopped.
  - The Extract run left trace #431 in data/traces.db and review #1 (test_000) in data/reviews.db.
- **pytest:** 69 passed.
- **Open (next steps, not done):** prompt v2 on dev (subtotal/tax always, tax=0, no modifiers as items, currency only if printed), then a single test score; vision on all 100 receipts on the GPU laptop; 3B/vision memory not measured.
