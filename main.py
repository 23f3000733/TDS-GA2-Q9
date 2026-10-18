import json
import os
import re
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

AIPIPE_URL = os.getenv("AIPIPE_URL", "https://aipipe.org/openrouter/v1")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN", "")
AIPIPE_MODEL = os.getenv("AIPIPE_MODEL", "openai/gpt-4.1-nano")
USE_LLM = os.getenv("USE_LLM", "1") == "1"

app = FastAPI(title="Invoice Extraction Service")


class ExtractRequest(BaseModel):
    text: str = Field(..., description="Free-form invoice text")


class Invoice(BaseModel):
    vendor: str
    amount: float
    currency: str
    date: str

    @field_validator("currency")
    @classmethod
    def _cur(cls, v: str) -> str:
        v = (v or "").strip().upper()
        return v if re.fullmatch(r"[A-Z]{3}", v) else "USD"

    @field_validator("date")
    @classmethod
    def _date(cls, v: str) -> str:
        m = re.search(r"\d{4}-\d{2}-\d{2}", v or "")
        return m.group(0) if m else ""


# ---------------------------------------------------------------- regex path
CURRENCIES = {"USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF"}
SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
SUFFIX = (r"(?:Industries\s+)?(?:Ltd\.?|Limited|Inc\.?|LLC|L\.L\.C\.|GmbH|Corp\.?|"
          r"Corporation|Co\.?|PLC|plc|AG|S\.A\.|Pvt\.?(?:\s+Ltd\.?)?|LLP)")
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _vendor(text: str) -> Optional[str]:
    m = re.search(r"(?:vendor|supplier|bill(?:ed)?\s+from|from|issued\s+by|payee)\s*[:\-]\s*([^\n,;|]+)",
                  text, re.I)
    if m:
        return m.group(1).strip().rstrip(".") if not re.search(SUFFIX + r"$", m.group(1)) \
            else m.group(1).strip()
    m = re.search(r"((?:[A-Z][\w&'\-]*\s+){0,4}[A-Z][\w&'\-]*\s+" + SUFFIX + r")", text)
    if m:
        name = m.group(1).strip()
        name = re.sub(r"^(?:Invoice|Bill|From|Payment|Please|Pay|To|Dear|Received|Issued|By)\s+(?:from\s+|by\s+|to\s+)?",
                      "", name)
        return name.strip()
    m = re.search(r"\b(?:from|by)\s+([A-Z][\w&'\-]*(?:\s+[A-Z][\w&'\-]*){0,3})", text)
    return m.group(1).strip() if m else None


def _currency(text: str) -> Optional[str]:
    for m in re.finditer(r"\b([A-Z]{3})\b", text):
        if m.group(1) in CURRENCIES:
            return m.group(1)
    for s, c in SYMBOLS.items():
        if s in text:
            return c
    return None


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def _amount(text: str) -> Optional[float]:
    cur = r"(?:USD|EUR|GBP|INR|[$€£₹])?\s*"
    strong = r"(?:total\s+(?:amount\s+)?due|amount\s+due|balance\s+due|grand\s+total|total\s+payable|amount\s+payable)"
    weak = r"(?:total|amount)"
    m = re.search(strong + r"[^\d\n$€£₹]{0,25}" + cur + NUM, text, re.I)
    if m:
        return _to_float(m.group(1))
    m = (re.search(r"(?:USD|EUR|GBP|INR|[$€£₹])\s*" + NUM, text)
         or re.search(NUM + r"\s*(?:USD|EUR|GBP|INR)\b", text))
    if m:
        return _to_float(m.group(1))
    m = re.search(weak + r"[^\d\n$€£₹]{0,25}" + cur + NUM, text, re.I)
    return _to_float(m.group(1)) if m else None


def _parse_date(s: str) -> Optional[str]:
    s = s.strip()
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3})[a-z]*\.?,?\s+(\d{4})", s)
    if m and m.group(2).lower() in MONTHS:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    m = re.search(r"([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s)
    if m and m.group(1).lower() in MONTHS:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"
    return None


def _date(text: str) -> Optional[str]:
    m = re.search(r"(?:due(?:\s+date)?|pay(?:ment)?\s+by|payable\s+by|due\s+on|due\s+by)[^\n\d]{0,20}([^\n]{0,30})",
                  text, re.I)
    if m:
        d = _parse_date(m.group(1))
        if d:
            return d
    iso = re.findall(r"\d{4}-\d{2}-\d{2}", text)
    if iso:
        return iso[-1]
    return _parse_date(text)


def regex_extract(text: str) -> dict:
    return {"vendor": _vendor(text), "amount": _amount(text),
            "currency": _currency(text), "date": _date(text)}


# ---------------------------------------------------------------- LLM path (AI Pipe)
async def llm_extract(text: str) -> dict:
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"vendor": {"type": "string"}, "amount": {"type": "number"},
                             "currency": {"type": "string"}, "date": {"type": "string"}},
              "required": ["vendor", "amount", "currency", "date"]}
    prompt = ("Extract from this invoice: vendor (company name exactly as written), "
              "amount (total amount due, plain number), currency (3-letter ISO code, uppercase), "
              "date (payment DUE date as YYYY-MM-DD). Reply with JSON only.\n\nInvoice:\n" + text[:4000])
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            f"{AIPIPE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
            json={"model": AIPIPE_MODEL, "temperature": 0,
                  "response_format": {"type": "json_schema",
                                      "json_schema": {"name": "invoice", "strict": True, "schema": schema}},
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])


def verified(llm: dict, text: str) -> dict:
    """Keep an LLM field only if it is grounded in the invoice text."""
    out = {}
    v = str(llm.get("vendor") or "").strip()
    if v and v.lower() in text.lower():
        out["vendor"] = v
    try:
        a = float(llm.get("amount"))
        nums = {round(float(n.replace(",", "")), 2) for n in re.findall(r"\d[\d,]*\.?\d*", text)
                if re.search(r"\d", n) and n.replace(",", "").replace(".", "", 1).isdigit()}
        if round(a, 2) in nums:
            out["amount"] = a
    except Exception:
        pass
    c = str(llm.get("currency") or "").strip().upper()
    if re.fullmatch(r"[A-Z]{3}", c):
        out["currency"] = c
    d = str(llm.get("date") or "")
    m = re.search(r"\d{4}-\d{2}-\d{2}", d)
    if m:
        try:
            datetime.strptime(m.group(0), "%Y-%m-%d")
            out["date"] = m.group(0)
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------- endpoint
@app.post("/extract", response_model=Invoice)
async def extract(req: ExtractRequest):
    text = req.text or ""
    result = {}
    if USE_LLM and AIPIPE_TOKEN and text.strip():
        try:
            result = verified(await llm_extract(text), text)
        except Exception:
            result = {}
    rx = regex_extract(text)
    for k, v in rx.items():  # regex fills anything the LLM missed or failed to ground
        if result.get(k) in (None, "") and v not in (None, ""):
            result[k] = v

    try:
        return Invoice(
            vendor=str(result.get("vendor") or ""),
            amount=float(result.get("amount") or 0.0),
            currency=str(result.get("currency") or "USD"),
            date=str(result.get("date") or ""),
        )
    except Exception:
        return Invoice(vendor="", amount=0.0, currency="USD", date="")


@app.exception_handler(Exception)
async def _any(_, exc):  # never leak a 500
    return JSONResponse(status_code=422, content={"detail": "could not process input"})


@app.get("/")
def health():
    return {"status": "ok", "endpoint": "POST /extract"}
