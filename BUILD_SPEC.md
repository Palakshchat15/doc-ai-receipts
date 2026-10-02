# Build spec: Document AI, receipts to validated JSON

Follow `../CONVENTIONS.md` for all shared rules.

## Goal
Turn receipt images into structured, validated JSON, with a human-review queue for low-confidence results. Measure it field by field on a public labelled dataset.

## Data
- **CORD v2** (Consolidated Receipt Dataset, NAVER Clova), from Hugging Face `naver-clova-ix/cord-v2`. It has train, validation and test splits (about 800 / 100 / 100) with JSON ground truth.
- **Check the licence** on the dataset card (expected CC BY 4.0) and attribute it.
- **If CORD can't be downloaded or used,** fall back to another public receipt/invoice dataset with field labels and a clear licence, and document the choice.
- **Splits:** use CORD's validation split as dev (tuning) and its test split as test (reported numbers).

## Target schema (Pydantic)
- `store_name` (if labelled);
- `items`: list of {`name`, `quantity`, `unit_price` (if labelled), `line_total`};
- `subtotal`, `tax`, `service_charge` (if labelled), `total`;
- `currency` if derivable.

Map CORD's label keys (e.g. `menu.nm`, `menu.cnt`, `menu.price`, `sub_total.subtotal_price`, `sub_total.tax_price`, `total.total_price`) to this schema. Document the mapping.

## Pipeline

**OCR**
- A CPU OCR library installable with pip and no admin rights. Preferred: RapidOCR (`rapidocr-onnxruntime`, PaddleOCR models in ONNX). Keep the text boxes and their positions.
- Reconstruct reading-order lines from the boxes (group by y, then sort by x).

**Extraction**
- An LLM via Ollama structured outputs (the `format` parameter with a JSON schema), `temperature=0`.
- Pass the OCR lines; treat the document text as data, not instructions.

**Validation**
- Parse into Pydantic and normalise numbers: thousands separators, decimal commas, currency symbols.
- Business rules with tolerances:
  - the items sum to the subtotal;
  - subtotal + tax (+ service) = total;
  - quantity x unit price = line total.
- Build a confidence signal from rule results, schema validity and field presence.

**Review routing**
- Auto-approve receipts that pass all rules. Send the rest to a **review queue**, where the app lets a user correct and approve. Store the reviews in SQLite.

**Compare these systems on test:**

| System | What it is |
|---|---|
| (a) OCR + regex/heuristic baseline | total and subtotal by keyword rules; items not attempted, or naive |
| (b) OCR + `qwen2.5:7b` | the main system |
| (c) OCR + `qwen2.5:3b` | the small-model option |
| (d) a **vision LLM** end to end on the image | e.g. `qwen2.5vl:3b`, or the best available in Ollama; OCR-free. Run it on test only if CPU time allows, otherwise on a subset, and say which |

## Metrics
- **Field-level** accuracy / F1 for each scalar field, after normalisation.
- **Item-level** F1: match predicted items to gold items by name similarity plus price. Report the item count accuracy.
- **Whole-receipt** exact-match rate.
- **Auto-approve rate** and accuracy **of the auto-approved set**: does the review routing catch errors? Show the trade-off curve if you use a threshold.
- Latency per stage (OCR, LLM), p50/p95, and memory.
- **Error analysis:** OCR errors vs extraction errors vs normalisation errors, with examples.

## Streamlit app
1. **Extract:** upload an image or choose a test sample; show the image with OCR boxes drawn, the extracted JSON, the rule checks with pass/fail, and the routing decision.
2. **Review queue:** list, edit and approve items.
3. **Evaluation:** system comparison tables and charts, a per-receipt browser, and error buckets.
4. **Monitoring:** traces.
