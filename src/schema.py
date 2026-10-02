"""Target schema (Pydantic) and the CORD v2 -> schema mapping.

CORD label key                      -> schema field
menu[].nm                            -> items[].name
menu[].cnt                           -> items[].quantity
menu[].unitprice                     -> items[].unit_price
menu[].price                         -> items[].line_total   (menu[].itemsubtotal wins if present;
                                                              then menu[].price is the unit price)
menu[].sub (modifiers / add-ons)     -> not separate items (see README, design decisions)
sub_total.subtotal_price             -> subtotal
sub_total.discount_price             -> discount   (stored as a positive magnitude)
sub_total.tax_price                  -> tax
sub_total.service_price              -> service_charge
total.total_price                    -> total
(no label)                           -> store_name, currency (extracted, not scored)
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .normalize import parse_amount, parse_quantity

MONEY_FIELDS = ["subtotal", "discount", "tax", "service_charge", "total"]
SCORED_FIELDS = ["subtotal", "tax", "service_charge", "total"]


class Item(BaseModel):
    name: str = ""
    quantity: float | None = None
    unit_price: float | None = None
    line_total: float | None = None

    @field_validator("quantity", mode="before")
    @classmethod
    def _q(cls, v: Any):
        return parse_quantity(v)

    @field_validator("unit_price", "line_total", mode="before")
    @classmethod
    def _m(cls, v: Any):
        p = parse_amount(v)
        return abs(p) if p is not None else None

    @field_validator("name", mode="before")
    @classmethod
    def _n(cls, v: Any):
        return "" if v is None else str(v).strip()


class Receipt(BaseModel):
    store_name: str | None = None
    items: list[Item] = Field(default_factory=list)
    subtotal: float | None = None
    discount: float | None = None
    tax: float | None = None
    service_charge: float | None = None
    total: float | None = None
    currency: str | None = None

    @field_validator(*MONEY_FIELDS, mode="before")
    @classmethod
    def _money(cls, v: Any):
        p = parse_amount(v)
        return abs(p) if p is not None else None

    @field_validator("items", mode="before")
    @classmethod
    def _items(cls, v: Any):
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        return [i for i in v if isinstance(i, (dict, Item))]

    @field_validator("store_name", "currency", mode="before")
    @classmethod
    def _s(cls, v: Any):
        if v is None:
            return None
        s = str(v).strip()
        return s or None


# JSON schema handed to Ollama's `format` parameter. Money values are strings exactly as
# printed, so the LLM copies rather than converts; our normaliser parses them (the spec's
# "normalise numbers" step). Quantities likewise.
LLM_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "store_name": {"type": ["string", "null"]},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": ["string", "null"]},
                    "unit_price": {"type": ["string", "null"]},
                    "line_total": {"type": ["string", "null"]},
                },
                "required": ["name", "quantity", "unit_price", "line_total"],
            },
        },
        "subtotal": {"type": ["string", "null"]},
        "discount": {"type": ["string", "null"]},
        "tax": {"type": ["string", "null"]},
        "service_charge": {"type": ["string", "null"]},
        "total": {"type": ["string", "null"]},
        "currency": {"type": ["string", "null"]},
    },
    "required": ["store_name", "items", "subtotal", "discount", "tax", "service_charge", "total",
                 "currency"],
}


def _first(v):
    if isinstance(v, list):
        return v[0] if v else None
    return v


def cord_to_receipt(gt_parse: dict) -> Receipt:
    """Map a CORD v2 `gt_parse` dict to our Receipt (gold)."""
    menu = gt_parse.get("menu") or []
    if isinstance(menu, dict):
        menu = [menu]
    items = []
    for m in menu:
        if not isinstance(m, dict):
            continue
        name = _first(m.get("nm"))
        if name is None:
            continue
        price = _first(m.get("price"))
        unit = _first(m.get("unitprice"))
        line = price
        if m.get("itemsubtotal") is not None:
            line = _first(m.get("itemsubtotal"))
            if unit is None:
                unit = price
        items.append({"name": name, "quantity": _first(m.get("cnt")), "unit_price": unit,
                      "line_total": line})
    st = gt_parse.get("sub_total") or {}
    tt = gt_parse.get("total") or {}
    if isinstance(st, list):
        st = st[0] if st else {}
    if isinstance(tt, list):
        tt = tt[0] if tt else {}
    return Receipt(
        items=items,
        subtotal=st.get("subtotal_price"),
        discount=st.get("discount_price"),
        tax=st.get("tax_price"),
        service_charge=st.get("service_price"),
        total=tt.get("total_price"),
    )
