import math

from src.baseline import amounts, extract_regex
from src.evaluation import match_items, routing_curve, score_receipt, summarise
from src.ocr import reconstruct_lines
from src.schema import Item, Receipt


def box(x0, y0, x1, y1, text):
    return {"quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "text": text, "score": 0.9}


def test_reconstruct_lines_groups_by_y_and_sorts_by_x():
    boxes = [box(300, 102, 360, 120, "24,000"), box(10, 100, 120, 118, "JASMINE"),
             box(10, 140, 100, 158, "TOTAL"), box(300, 141, 360, 159, "24,000"),
             box(130, 99, 160, 119, "TEA")]
    lines = reconstruct_lines(boxes)
    assert len(lines) == 2
    assert lines[0].startswith("JASMINE TEA") and lines[0].endswith("24,000")
    assert "   " in lines[0]     # wide gap kept as a column break
    assert lines[1].split() == ["TOTAL", "24,000"]


def test_reconstruct_lines_handles_skew():
    # a 5-degree rotated receipt: the right-hand price sits ~26 px lower than its label
    a = math.radians(5)

    def rbox(x0, y0, w, h, text):
        pts = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
        rp = [[x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)] for x, y in pts]
        return {"quad": rp, "text": text, "score": 0.9}

    boxes = [rbox(10, 100, 120, 18, "COFFEE"), rbox(300, 100, 60, 18, "18,000"),
             rbox(10, 130, 100, 18, "TOTAL"), rbox(300, 130, 60, 18, "18,000")]
    lines = reconstruct_lines(boxes)
    assert [ln.split() for ln in lines] == [["COFFEE", "18,000"], ["TOTAL", "18,000"]]


def test_reconstruct_lines_empty():
    assert reconstruct_lines([]) == []


def test_amounts():
    assert amounts("1X M-Caramel Black Tea @28,000 28,000") == [1.0, 28000.0, 28000.0]
    assert amounts("TOTAL") == []
    assert amounts("1 120,000") == [1.0, 120000.0]      # qty then price, not 1,120,000
    assert amounts("TOTAL 20 000") == [20000.0]
    assert amounts("TOTAL 121, 768") == [121768.0]


def test_regex_ocr_damaged_labels():
    assert extract_regex(["SUBTTL 243,000", "3TOTAL 283,338"]).total == 283338
    assert extract_regex(["SUBTTL 243,000", "3TOTAL 283,338"]).subtotal == 243000
    assert extract_regex(["Total Tax Included: Rp5272.73", "Total: Rp58000.00"]).total == 58000


def test_regex_baseline():
    lines = ["STORE ABC", "1 POTATO BREAD   19,000", "2 GREEN TEA   52.000", "SUBTOTAL   71,000",
             "PB1 10%   7,100", "TOTAL   78,100", "CASH   100,000", "CHANGE   21,900"]
    r = extract_regex(lines)
    assert r.subtotal == 71000 and r.tax == 7100 and r.total == 78100
    assert [(i.name, i.quantity, i.line_total) for i in r.items] == [("POTATO BREAD", 1, 19000),
                                                                      ("GREEN TEA", 2, 52000)]


def test_regex_baseline_total_on_next_line_and_grand_total():
    r = extract_regex(["TOTAL QTY 3", "Grand Total", "45,000", "TOTAL PAID 50,000"])
    assert r.total == 45000


def test_item_matching_requires_price_and_name():
    gold = [Item(name="JASMINE MT (L)", line_total=24000), Item(name="COCONUT JELLY", line_total=4000)]
    pred = [Item(name="JASMINE MT L", line_total=24000), Item(name="COCONUT JELLY", line_total=5000)]
    pairs = match_items(pred, gold)
    assert [(i, j) for i, j, _ in pairs] == [(0, 0)]
    assert len(match_items(pred, gold, use_price=False)) == 2


def test_score_receipt_and_summary():
    gold = Receipt(items=[Item(name="A", quantity=2, line_total=10000), Item(name="B", line_total=3000)],
                   subtotal=13000, total=13000)
    perfect = score_receipt(gold, gold)
    assert perfect["exact_match"] and perfect["item_tp"] == 2 and perfect["qty_correct"] == 1
    wrong = Receipt(items=[Item(name="A", quantity=2, line_total=10000)], subtotal=13000, total=13001,
                    tax=100)
    s = score_receipt(wrong, gold)
    assert not s["total_correct"] and s["total_fp"] == 1 and s["total_fn"] == 1
    assert s["tax_fp"] == 1 and s["tax_fn"] == 0 and not s["tax_correct"]
    assert s["service_charge_correct"]          # both absent counts as correct
    assert s["item_tp"] == 1 and s["item_fn"] == 1 and not s["exact_match"]
    none = score_receipt(None, gold)
    assert not none["schema_valid"] and none["item_fn"] == 2
    m = summarise([perfect, s])
    assert m["exact_match"] == 0.5 and m["total_acc"] == 0.5
    assert abs(m["item_f1"] - (2 * (3 / 3) * (3 / 4) / (1 + 3 / 4))) < 1e-9


def test_routing_curve_monotone_threshold():
    scores = [{"exact_match": True, "total_correct": True}, {"exact_match": False, "total_correct": False}]
    rows = routing_curve(scores, [1.0, 0.5])
    assert rows[0]["threshold"] == 1.0 and rows[0]["approve_rate"] == 0.5 and rows[0]["approved_exact"] == 1.0
    assert rows[-1]["approve_rate"] == 1.0
