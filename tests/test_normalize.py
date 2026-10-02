import pytest

from src.normalize import detect_currency, money_equal, norm_name, parse_amount, parse_quantity


@pytest.mark.parametrize("s,expected", [
    ("60.000", 60000.0),          # Indonesian thousands dot
    ("24,000", 24000.0),          # thousands comma
    ("1,555,200", 1555200.0),
    ("1.213.000", 1213000.0),
    ("70000.00", 70000.0),        # real decimals
    ("12,50", 12.5),              # decimal comma
    ("1.234,56", 1234.56),
    ("1,234.56", 1234.56),
    ("17500", 17500.0),
    ("Rp 39.600", 39600.0),       # currency word
    ("Rp. 39.600", 39600.0),
    ("IDR 5,000", 5000.0),
    ("$12.99", 12.99),
    ("@120,000", 120000.0),       # unit-price marker
    ("-60.000", -60000.0),
    ("-2 600", -2600.0),          # space as a thousands separator
    ("(0)", 0.0),
    ("[145,900]", 145900.0),
    ("0", 0.0),
    ("0.00", 0.0),
    (" 5.455 ", 5455.0),
    (12000, 12000.0),
    (12.5, 12.5),
])
def test_parse_amount(s, expected):
    assert parse_amount(s) == pytest.approx(expected)


@pytest.mark.parametrize("s", [None, "", "abc", "Rp", "-", True])
def test_parse_amount_none(s):
    assert parse_amount(s) is None


def test_parse_amount_list_takes_first_number():
    assert parse_amount(["x", "1,213,000", "60,000"]) == 1213000.0


def test_parenthesised_is_negative():
    assert parse_amount("(5,000)") == -5000.0


@pytest.mark.parametrize("s,expected", [
    ("2", 2.0), ("1X", 1.0), ("x3", 3.0), ("2.00", 2.0), ("(2", 2.0), ("1.00xITEMs", 1.0),
    (3, 3.0), ("1,5", 1.5),
])
def test_parse_quantity(s, expected):
    assert parse_quantity(s) == expected


def test_parse_quantity_none():
    assert parse_quantity("pcs") is None
    assert parse_quantity(None) is None


def test_detect_currency():
    assert detect_currency("TOTAL Rp 50.000") == "IDR"
    assert detect_currency("Total $12.00") == "USD"
    assert detect_currency("TOTAL 50.000") is None
    assert detect_currency("GRAPE JUICE") is None   # 'rp' inside a word is not a currency


def test_norm_name():
    assert norm_name(" j.stb  promo ") == "J STB PROMO"
    assert norm_name("JASMINE MT ( L )") == "JASMINE MT L"
    assert norm_name(None) == ""


def test_money_equal():
    assert money_equal(100.0, 100.4, abs_tol=0.5)
    assert not money_equal(100.0, 101.0, abs_tol=0.5)
    assert money_equal(None, None)
    assert not money_equal(None, 5.0)
    assert money_equal(100000, 100400, abs_tol=1, rel_tol=0.005)
