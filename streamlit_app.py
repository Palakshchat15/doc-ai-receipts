"""Document AI: receipts -> validated JSON. Pages: Extract / Review queue / Evaluation / Monitoring.

Run:  .venv\\Scripts\\streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import gold_for, load_config, ocr_for, path, read_jsonl  # noqa: E402

st.set_page_config(page_title="Receipts to validated JSON", page_icon=":receipt:", layout="wide")

ATTRIBUTION = ("Data: **CORD v2** (Consolidated Receipt Dataset, NAVER Clova; Park et al., 2019), "
               "CC BY 4.0, via Hugging Face `naver-clova-ix/cord-v2`. OCR: RapidOCR (Apache-2.0).")
CFG = load_config()
RES = path("results")
MODES = {"OCR + LLM": "ocr_llm", "Vision LLM (no OCR)": "vision", "OCR + regex baseline (no LLM)": "regex"}
# Fixed system -> colour mapping (reference categorical slots in fixed order; colour follows the system).
SYSTEM_ORDER = ["(a) OCR + regex", "(b) OCR + qwen2.5:7b", "(c) OCR + qwen2.5:3b", "(d) vision LLM",
                "gold OCR + LLM (upper bound)"]
SYSTEM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def color_scale(names) -> alt.Scale:
    keys = sorted(set(names), key=lambda n: next((i for i, k in enumerate(SYSTEM_ORDER) if n.startswith(k[:4])), 99))
    dom, rng = [], []
    for n in keys:
        i = next((i for i, k in enumerate(SYSTEM_ORDER) if n.startswith(k[:4])), len(SYSTEM_COLORS) - 1)
        dom.append(n)
        rng.append(SYSTEM_COLORS[min(i, len(SYSTEM_COLORS) - 1)])
    return alt.Scale(domain=dom, range=rng)


def recessive(chart: alt.Chart) -> alt.Chart:
    return chart.configure_axis(grid=False, domainColor="#888", tickColor="#888", labelColor="#666",
                                titleColor="#666").configure_view(strokeWidth=0)


# ---------------------------------------------------------------------------------------------
# cached resources
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading OCR models ...")
def get_ocr():
    from src.ocr import get_engine

    get_engine()
    return True


@st.cache_resource
def tracer():
    from src.tracing import Tracer

    return Tracer(path("traces_db"))


@st.cache_resource
def reviews():
    from src.tracing import ReviewStore

    return ReviewStore(path("reviews_db"))


@st.cache_resource
def deep_link_results() -> dict:
    return {}


@st.cache_data
def test_rows() -> dict:
    return {r["id"]: r for r in gold_for("test")}


@st.cache_data
def test_ocr() -> dict:
    return ocr_for("test")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_json(name: str):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt_money(v):
    return "-" if v is None else f"{v:,.2f}".rstrip("0").rstrip(".")


def receipt_tables(rec: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.DataFrame(rec.get("items") or [], columns=["name", "quantity", "unit_price", "line_total"])
    for c in ("unit_price", "line_total"):  # same money format as the scalar table
        items[c] = items[c].astype(object).map(lambda v: fmt_money(v) if isinstance(v, (int, float)) and pd.notna(v) else "-")
    scal = pd.DataFrame([{"field": k, "value": rec.get(k)} for k in
                         ["store_name", "subtotal", "discount", "tax", "service_charge", "total", "currency"]])
    scal["value"] = scal["value"].astype(object).where(scal["value"].notna(), None).map(
        lambda v: "-" if v is None else (fmt_money(v) if isinstance(v, (int, float)) else str(v)))
    return items, scal


def rules_table(vd: dict) -> pd.DataFrame:
    icon = {"pass": "PASS", "fail": "FAIL", "skip": "skip"}
    return pd.DataFrame([{"rule": r["name"], "result": icon.get(r["status"], r["status"]), "detail": r["detail"]}
                         for r in vd["rules"]])


def show_decision(vd: dict):
    if vd["decision"] == "auto_approve":
        st.success(f"Routing: AUTO-APPROVE (confidence {vd['confidence']:.2f}; every rule passed)")
    else:
        st.warning(f"Routing: HUMAN REVIEW (confidence {vd['confidence']:.2f}; "
                   f"{'; '.join(vd['reasons']) or 'below threshold'})")


# ---------------------------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------------------------
def page_extract():
    from src.llm import OllamaUnavailable, check_ollama

    st.title("Extract: receipt image to validated JSON")
    st.caption("OCR (RapidOCR) -> reading-order lines -> LLM with a JSON schema (Ollama structured outputs, "
               "temperature 0) -> Pydantic + number normalisation -> arithmetic rules -> auto-approve or review.")
    with st.sidebar:
        st.subheader("Settings")
        mode_label = st.radio("System", list(MODES), index=0)
        mode = MODES[mode_label]
        model = None
        ollama_ok = True
        if mode == "ocr_llm":
            models = CFG["llm"]["models_evaluated"]
            model = st.selectbox("Extraction model (Ollama)", models,
                                 index=models.index(CFG["llm"]["model"]) if CFG["llm"]["model"] in models else 0)
        elif mode == "vision":
            model = st.text_input("Vision model (Ollama)", CFG["llm"]["vision_model"])
        if model:
            try:
                check_ollama(model)
                st.success(f"Ollama OK: {model}")
            except OllamaUnavailable as e:
                st.error(str(e))
                ollama_ok = False

    src = st.radio("Input", ["Test-set sample", "Upload an image"], horizontal=True)
    img_path, rid, cached_ocr, gold = None, None, None, None
    if src == "Test-set sample":
        rows = test_rows()
        ids = list(rows)
        want = st.query_params.get("sample")
        rid = st.selectbox("CORD test receipt", ids, index=ids.index(want) if want in rows else 0)
        img_path = ROOT / rows[rid]["image"]
        cached_ocr = test_ocr().get(rid)
        gold = rows[rid]["gold"]
    else:
        up = st.file_uploader("Receipt photo or scan", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            d = path("cache") / "uploads"
            d.mkdir(parents=True, exist_ok=True)
            img_path = d / f"{int(time.time())}_{Path(up.name).name}"
            img_path.write_bytes(up.getvalue())
            rid = f"upload:{up.name}"
    if img_path is None:
        st.info("Choose a sample or upload an image.")
        return

    go = st.button("Extract", type="primary", disabled=not ollama_ok)
    key = f"{rid}|{mode}|{model}"
    # Deep link ?sample=<id>&run=1 (used for headless screenshots): runs once per key, then reuses the result.
    autorun = st.query_params.get("run") == "1" and src == "Test-set sample" and rid == want
    if autorun and not go:
        hit = deep_link_results().get(key)
        if hit is not None:
            st.session_state["extract"] = (key, hit)
        elif (st.session_state.get("extract") or (None,))[0] != key:
            go = ollama_ok
    if go:
        with st.spinner("Running OCR and extraction (CPU: OCR ~5-10 s, 7B LLM ~1-2 min, 3B ~30-60 s) ..."):
            try:
                out = run_one(img_path, rid, mode, model, cached_ocr)
                st.session_state["extract"] = (key, out)
                if autorun:
                    deep_link_results()[key] = out
            except OllamaUnavailable as e:
                st.error(str(e))
                return
            except Exception as e:  # noqa: BLE001
                st.error(f"Extraction failed: {type(e).__name__}: {e}")
                return

    left, right = st.columns([1, 1.3])
    res = st.session_state.get("extract")
    res = res[1] if res and res[0] == key else None
    with left:
        if res and res.get("ocr"):
            boxes = res["ocr"]["boxes"]
        elif cached_ocr and mode != "vision":
            boxes = cached_ocr["boxes"]
        else:
            boxes = None
        if boxes:
            from src.ocr import draw_boxes

            st.image(draw_boxes(img_path, boxes), caption=f"{rid}: {len(boxes)} OCR boxes", use_container_width=True)
        else:
            st.image(str(img_path), caption=rid, use_container_width=True)
    with right:
        if not res:
            if cached_ocr and mode != "vision":
                with st.expander("OCR lines (cached from the evaluation run)", expanded=True):
                    st.code("\n".join(cached_ocr["lines"]), language=None)
            st.info("Press Extract.")
            return
        vd = res["validation"]
        show_decision(vd)
        c = st.columns(4)
        c[0].metric("OCR", f"{res['t_ocr']:.1f} s")
        c[1].metric("LLM", f"{res['t_llm']:.1f} s")
        c[2].metric("Tokens in / out", f"{res.get('prompt_tokens', 0)} / {res.get('completion_tokens', 0)}")
        c[3].metric("Confidence", f"{vd['confidence']:.2f}")
        if res.get("error"):
            st.error(f"Output did not validate: {res['error']}")
        st.subheader("Rule checks")
        st.dataframe(rules_table(vd), hide_index=True, use_container_width=True)
        if res.get("receipt"):
            items, scal = receipt_tables(res["receipt"])
            st.subheader("Extracted fields")
            a, b = st.columns([1, 2])
            a.dataframe(scal, hide_index=True, use_container_width=True)
            b.dataframe(items, hide_index=True, use_container_width=True)
            with st.expander("Validated JSON"):
                st.json(res["receipt"])
        if gold is not None and res.get("receipt"):
            with st.expander("Compare with the CORD gold labels"):
                from src.evaluation import score_receipt
                from src.schema import Receipt

                s = score_receipt(Receipt(**res["receipt"]), Receipt(**gold))
                st.write({k: s[k] for k in ["total_correct", "subtotal_correct", "tax_correct",
                                            "service_charge_correct", "item_tp", "item_fp", "item_fn",
                                            "exact_match"]})
                gi, gs = receipt_tables(gold)
                a, b = st.columns([1, 2])
                a.dataframe(gs, hide_index=True, use_container_width=True)
                b.dataframe(gi, hide_index=True, use_container_width=True)
        if res.get("ocr"):
            with st.expander("OCR lines given to the LLM (treated as data, not instructions)"):
                st.code("\n".join(res["ocr"]["lines"]), language=None)
        if res.get("review_id"):
            st.caption(f"Added to the review queue as #{res['review_id']}.")


def run_one(img_path, rid, mode, model, cached_ocr) -> dict:
    from src.baseline import extract_regex
    from src.extract import PROMPT_VERSION, run_pipeline
    from src.ocr import run_ocr
    from src.validation import validate

    t0 = time.perf_counter()
    if mode == "regex":
        get_ocr()
        ocr = cached_ocr or run_ocr(img_path)
        rec = extract_regex(ocr["lines"])
        v = validate(rec, CFG["validation"])
        out = {"mode": mode, "model": "regex", "ocr": ocr, "t_ocr": ocr["t_ocr"], "t_llm": 0.0, "receipt": rec,
               "error": None, "validation": v, "llm": {}}
    else:
        if mode == "ocr_llm" and cached_ocr is None:
            get_ocr()
        out = run_pipeline(img_path, model=model, mode=mode, ocr_result=cached_ocr)
    v = out["validation"]
    rec = out["receipt"]
    llm = out.get("llm") or {}
    res = {
        "mode": mode, "model": out["model"], "ocr": out.get("ocr"), "t_ocr": out["t_ocr"], "t_llm": out["t_llm"],
        "prompt_tokens": llm.get("prompt_tokens", 0), "completion_tokens": llm.get("completion_tokens", 0),
        "receipt": rec.model_dump() if rec else None, "error": out.get("error"), "validation": v.to_dict(),
    }
    t_total = time.perf_counter() - t0
    tracer().log(source="app", receipt=rid, mode=mode, model=out["model"],
                 prompt_version=PROMPT_VERSION if mode != "regex" else "-",
                 t_ocr=out["t_ocr"], t_llm=out["t_llm"], t_validate=out.get("t_validate"), t_total=t_total,
                 prompt_tokens=res["prompt_tokens"], completion_tokens=res["completion_tokens"],
                 n_boxes=len(out["ocr"]["boxes"]) if out.get("ocr") else None,
                 input="\n".join(out["ocr"]["lines"]) if out.get("ocr") else "(image)",
                 output=res["receipt"], confidence=v.confidence, decision=v.decision,
                 rules=v.to_dict()["rules"], error=out.get("error"))
    if v.decision == "review":
        res["review_id"] = reviews().add(rid, str(img_path), out["model"], v.confidence, v.reasons,
                                         res["receipt"] or {})
    return res


# ---------------------------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------------------------
def page_review():
    from src.schema import Receipt
    from src.validation import validate

    st.title("Review queue")
    st.caption("Receipts that failed a rule (or scored below the confidence threshold) wait here. Edit the "
               "fields, watch the rules re-run, then approve. Stored in data/reviews.db (SQLite).")
    store = reviews()
    pending = store.list("pending")
    done = store.list()
    done = done[done.status != "pending"]
    c = st.columns(3)
    c[0].metric("Pending", len(pending))
    c[1].metric("Approved", int((done.status == "approved").sum()) if len(done) else 0)
    c[2].metric("Rejected", int((done.status == "rejected").sum()) if len(done) else 0)

    with st.expander("Load the main system's test-set receipts that were routed to review"):
        f = sorted(path("preds").glob("test__ocr_llm__" + CFG["llm"]["model"].replace(":", "-") + "__*.jsonl"))
        if not f:
            st.caption("No test predictions for the main model yet.")
        elif st.button("Load into the queue"):
            n = 0
            rows = test_rows()
            for p in read_jsonl(f[-1]):
                if p.get("exception") or p["validation"]["decision"] != "review" or store.has_receipt(p["id"]):
                    continue
                store.add(p["id"], str(ROOT / rows[p["id"]]["image"]), p["model"], p["validation"]["confidence"],
                          p["validation"]["reasons"], p.get("pred") or {})
                n += 1
            st.success(f"Added {n} receipts.")
            st.rerun()

    if pending.empty:
        st.info("Nothing to review.")
    else:
        opts = {f"#{r.id}  {r.receipt}  (conf {r.confidence:.2f})": r for r in pending.itertuples()}
        sel = opts[st.selectbox("Pending receipt", list(opts))]
        pred = json.loads(sel.predicted or "{}")
        left, right = st.columns([1, 1.4])
        with left:
            if sel.image_path and Path(sel.image_path).exists():
                st.image(sel.image_path, use_container_width=True)
            st.caption("Reasons: " + ", ".join(json.loads(sel.reasons or "[]")))
        with right:
            st.subheader("Correct the fields")
            cols = st.columns(3)
            vals = {}
            for i, k in enumerate(["subtotal", "discount", "tax", "service_charge", "total"]):
                v = pred.get(k)
                txt = cols[i % 3].text_input(k, "" if v is None else fmt_money(v), key=f"rv{sel.id}{k}")
                vals[k] = txt.strip() or None
            vals["store_name"] = cols[2].text_input("store_name", pred.get("store_name") or "", key=f"rv{sel.id}s") or None
            items = pd.DataFrame(pred.get("items") or [], columns=["name", "quantity", "unit_price", "line_total"])
            ed = st.data_editor(items, num_rows="dynamic", use_container_width=True, key=f"rv{sel.id}items")
            vals["items"] = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in ed.to_dict("records")
                             if str(r.get("name") or "").strip()]
            vals["currency"] = pred.get("currency")
            try:
                corrected = Receipt(**vals)
                vd = validate(corrected, CFG["validation"]).to_dict()
                st.dataframe(rules_table(vd), hide_index=True, use_container_width=True)
                ok = True
            except Exception as e:  # noqa: BLE001
                st.error(f"Not valid yet: {e}")
                ok = False
            b1, b2 = st.columns(2)
            if b1.button("Approve", type="primary", disabled=not ok):
                store.resolve(sel.id, corrected.model_dump(), "approved")
                st.rerun()
            if b2.button("Reject"):
                store.resolve(sel.id, None, "rejected")
                st.rerun()
    if len(done):
        st.subheader("Reviewed")
        d = done.copy()
        d["reviewed"] = pd.to_datetime(d.ts_reviewed, unit="s")
        st.dataframe(d[["id", "receipt", "model", "confidence", "status", "reviewed"]], hide_index=True,
                     use_container_width=True)


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------
FIELD_COLS = {"total_acc": "total", "subtotal_acc": "subtotal", "tax_acc": "tax",
              "service_charge_acc": "service", "item_f1": "item F1", "exact_match": "whole receipt"}


def page_eval():
    st.title("Evaluation (CORD v2 test split, 100 receipts)")
    split = st.radio("Split", ["test", "dev"], horizontal=True,
                     help="Test = reported numbers (scored once). Dev = CORD validation, used for tuning.")
    m = load_csv(f"metrics_{split}.csv")
    if m is None:
        st.info(f"No results for {split} yet. Run scripts/eval_extract.py then scripts/score.py --split {split}.")
        return
    sysm = m[m.system.str.startswith("(") | m.system.str.startswith("gold OCR")].copy()
    ceil = m[m.system.str.startswith("gold labels")]
    st.subheader("System comparison")
    show = sysm[["system", "n"] + list(FIELD_COLS) + ["item_count_acc", "auto_approve_rate", "approved_exact",
                                                      "errors_caught", "llm_p50", "llm_p95", "total_p50"]]
    st.dataframe(show.rename(columns={**FIELD_COLS, "item_count_acc": "item count acc",
                                      "auto_approve_rate": "auto-approve rate",
                                      "approved_exact": "exact match | approved",
                                      "errors_caught": "share of errors sent to review",
                                      "llm_p50": "LLM p50 s", "llm_p95": "LLM p95 s", "total_p50": "total p50 s"})
                 .style.format(precision=3), hide_index=True, use_container_width=True)
    if len(ceil):
        st.caption(f"Routing ceiling: the gold labels themselves pass the rules on "
                   f"{ceil.iloc[0]['auto_approve_rate']:.0%} of {split} receipts (a perfect extractor could not be "
                   "auto-approved more often than that).")
    sub = load_csv(f"metrics_{split}_subset.csv")
    if sub is not None:
        st.caption(f"Some systems ran on a subset ({int(sub.n.iloc[0])} receipts). All systems on that subset:")
        st.dataframe(sub[["system", "n"] + list(FIELD_COLS) + ["auto_approve_rate", "llm_p50"]]
                     .rename(columns=FIELD_COLS).style.format(precision=3), hide_index=True, use_container_width=True)

    long = sysm.melt(id_vars="system", value_vars=list(FIELD_COLS), var_name="metric", value_name="score")
    long["metric"] = long.metric.map(FIELD_COLS)
    ch = alt.Chart(long).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("system:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title=None),
        color=alt.Color("system:N", scale=color_scale(long.system), legend=alt.Legend(orient="bottom", title=None, columns=2, labelLimit=0)),
        row=alt.Row("metric:N", title=None, sort=list(FIELD_COLS.values()),
                    header=alt.Header(labelAngle=0, labelAlign="left")),
        tooltip=["system", "metric", alt.Tooltip("score:Q", format=".3f")],
    ).properties(height=alt.Step(14), width=520)
    a, b = st.columns([1.2, 1])
    with a:
        st.markdown("**Accuracy by field** (field accuracy; items = F1; whole receipt = exact match)")
        if (sysm.n < sysm.n.max()).any():
            st.caption("Systems with n below the full split ran on a subset; compare them in the subset table above.")
        st.altair_chart(recessive(ch), use_container_width=False)
    with b:
        curve = load_csv(f"routing_curve_{split}.csv")
        if curve is not None:
            st.markdown("**Routing trade-off**: approve receipts with confidence >= t")
            st.caption("Colours as in the left chart; hover a point for its threshold. The threshold was chosen on dev only.")
            cc = alt.Chart(curve.dropna()).mark_line(point=alt.OverlayMarkDef(size=64), strokeWidth=2).encode(
                x=alt.X("approve_rate:Q", title="auto-approve rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("approved_exact:Q", title="exact match within the approved set", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("system:N", scale=color_scale(curve.system), legend=None),
                tooltip=["system", alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("approve_rate:Q", format=".2f"),
                         alt.Tooltip("approved_exact:Q", format=".2f"), alt.Tooltip("approved_total_acc:Q", format=".2f")],
            ).properties(height=330)
            st.altair_chart(recessive(cc), use_container_width=True)

    st.subheader("Latency and throughput (CPU, i5-1235U)")
    lat = sysm[["system", "ocr_p50", "ocr_p95", "llm_p50", "llm_p95", "total_p50", "total_p95",
                "throughput_per_min", "prompt_tok_mean", "completion_tok_mean"]]
    st.dataframe(lat.style.format(precision=1), hide_index=True, use_container_width=True)
    mem = load_json("memory.json")
    if mem:
        st.caption("Memory: " + "; ".join(f"{k}: {v}" for k, v in mem.items()))

    errs = load_json(f"error_analysis_{split}.json")
    if errs:
        st.subheader("Error buckets (main system)")
        st.caption(errs.get("method", ""))
        bk = pd.DataFrame(errs["buckets"])
        st.dataframe(bk[["bucket", "count", "description"]], hide_index=True, use_container_width=True)
        for b_ in errs["buckets"]:
            if b_.get("examples"):
                with st.expander(f"{b_['bucket']}: examples"):
                    st.dataframe(pd.DataFrame(b_["examples"]), hide_index=True, use_container_width=True)

    st.subheader("Per-receipt browser")
    per = load_csv(f"per_receipt_{split}.csv")
    if per is None:
        return
    systems = list(per.system.unique())
    s1, s2, s3 = st.columns([2, 2, 1])
    main_ix = next((i for i, n in enumerate(systems) if str(n).startswith("(b)")), 0)
    sysname = s1.selectbox("System", systems, index=main_ix)
    only_err = s3.checkbox("errors only", value=True)
    ps = per[per.system == sysname]
    if only_err:
        ps = ps[~ps.exact_match]
    if ps.empty:
        st.info("No receipts.")
        return
    rid = s2.selectbox("Receipt", list(ps.id))
    row = ps[ps.id == rid].iloc[0]
    gold_rows = {r["id"]: r for r in gold_for(split)}
    g = gold_rows[rid]
    predfile = find_pred_file(split, sysname)
    pred = next((p for p in read_jsonl(predfile) if p["id"] == rid), None) if predfile else None
    l, r = st.columns([1, 1.6])
    l.image(str(ROOT / g["image"]), use_container_width=True)
    with r:
        st.write({k: bool(row[k]) for k in ["total_correct", "subtotal_correct", "tax_correct",
                                            "service_charge_correct", "exact_match"]} |
                 {"items tp/fp/fn": f"{row.item_tp}/{row.item_fp}/{row.item_fn}", "decision": row.decision,
                  "confidence": row.confidence})
        gi, gs = receipt_tables(g["gold"])
        pi, psc = receipt_tables((pred or {}).get("pred") or {})
        comp = gs.rename(columns={"value": "gold"}).assign(predicted=psc["value"])
        st.dataframe(comp, hide_index=True, use_container_width=True)
        st.markdown("Gold items")
        st.dataframe(gi, hide_index=True, use_container_width=True)
        st.markdown("Predicted items")
        st.dataframe(pi, hide_index=True, use_container_width=True)
        if pred and pred.get("validation"):
            st.dataframe(rules_table(pred["validation"]), hide_index=True, use_container_width=True)


@st.cache_resource
def _score_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("score", ROOT / "scripts" / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _label_of(stem: str) -> str:
    return _score_module().label(stem)


def find_pred_file(split: str, sysname: str):
    for f in sorted(path("preds").glob(f"{split}__*.jsonl")):
        if _label_of(f.stem) == sysname:
            return f
    return None


# ---------------------------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------------------------
def page_monitor():
    st.title("Monitoring (data/traces.db)")
    db = path("traces_db")
    if not db.exists():
        st.info("No traces yet.")
        return
    df = tracer().read(2000)
    if df.empty:
        st.info("No traces yet.")
        return
    src = st.radio("Source", ["app", "eval", "all"], horizontal=True)
    if src != "all":
        df = df[df.source == src]
    if df.empty:
        st.info(f"No {src} traces yet.")
        return
    df["time"] = pd.to_datetime(df.ts, unit="s")
    c = st.columns(4)
    c[0].metric("Requests", len(df))
    c[1].metric("p50 total latency", f"{df.t_total.median():.1f} s")
    c[2].metric("p95 total latency", f"{df.t_total.quantile(0.95):.1f} s")
    c[3].metric("Auto-approved", f"{(df.decision == 'auto_approve').mean() * 100:.0f}%")
    st.subheader("Latency by stage (seconds)")
    grp = df.groupby(["mode", "model"])[["t_ocr", "t_llm", "t_total"]]
    st.dataframe(pd.concat({"p50": grp.median(), "p95": grp.quantile(0.95)}, axis=1).round(2),
                 use_container_width=True)
    st.subheader("Tokens per request")
    st.dataframe(df.groupby(["mode", "model"])[["prompt_tokens", "completion_tokens"]].mean().round(0),
                 use_container_width=True)
    st.subheader("Recent requests")
    recent = df.head(50)[["id", "time", "source", "receipt", "mode", "model", "t_ocr", "t_llm", "t_total",
                          "prompt_tokens", "completion_tokens", "confidence", "decision", "error"]]
    st.dataframe(recent, use_container_width=True, hide_index=True)
    tid = st.selectbox("Inspect trace", list(df.head(50).id))
    t = df[df.id == tid].iloc[0]
    a, b = st.columns(2)
    a.markdown("**Input (OCR text)**")
    a.code(t.input or "", language=None)
    b.markdown("**Output (JSON)**")
    try:
        b.json(json.loads(t.output) if t.output else {})
    except Exception:  # noqa: BLE001
        b.code(str(t.output))
    if t.rules:
        st.dataframe(pd.DataFrame(json.loads(t.rules)), hide_index=True, use_container_width=True)


nav = st.navigation([
    st.Page(page_extract, title="Extract", default=True),   # served at the root URL (url_path on the default page gave "Page not found")
    st.Page(page_review, title="Review queue", url_path="review"),
    st.Page(page_eval, title="Evaluation", url_path="evaluation"),
    st.Page(page_monitor, title="Monitoring", url_path="monitoring"),
])
with st.sidebar:
    st.caption(ATTRIBUTION)
nav.run()
