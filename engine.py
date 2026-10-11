"""Evaluate the formulas of Kakushadze's "101 Formulaic Alphas" on panels of daily data.

Each formula is parsed from its published text and evaluated directly on arrays of shape
(days, stocks). Operators follow the paper's appendix. Three inputs are approximated:
  vwap      = (high + low + close) / 3        (no intraday data)
  industry  = GICS sub-industry               (only sector and sub-industry are available)
  cap       = not available, so Alpha 56 is skipped
"""
import math
import re

import bottleneck as bn
import numpy as np
import pandas as pd

TOKEN = re.compile(r"\s*(\d+\.?\d*|\.\d+|[A-Za-z_][A-Za-z0-9_.]*|\|\||==|[-+*/^<?:(),])")


def tokenize(text):
    out, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            raise SyntaxError(f"cannot read {text[pos:pos + 20]!r}")
        out.append(m.group(1)); pos = m.end()
    return out


class Data:
    """Panels as float arrays (days x stocks) plus group labels for neutralising."""
    def __init__(self, panels, groups):
        self.p, self.groups = panels, groups
        c = panels["close"]
        self.p["returns"] = np.vstack([np.full((1, c.shape[1]), np.nan), c[1:] / c[:-1] - 1])
        self.p["vwap"] = (panels["high"] + panels["low"] + c) / 3
        self.dollar_volume = c * panels["volume"]

    def get(self, name):
        if name in self.p:
            return self.p[name]
        if name.startswith("adv"):
            self.p[name] = bn.move_mean(self.dollar_volume, int(name[3:]), axis=0)
            return self.p[name]
        raise KeyError(name)


def _n(d): return max(int(math.floor(float(d))), 1)

def rank(x): return pd.DataFrame(x).rank(axis=1, pct=True).values

def delay(x, d):
    d = _n(d); out = np.full_like(x, np.nan); out[d:] = x[:-d]; return out

def delta(x, d): return x - delay(x, d)

def covariance(x, y, d):
    d = _n(d); ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(ok, x, np.nan), np.where(ok, y, np.nan)
    return (bn.move_mean(x * y, d, axis=0) - bn.move_mean(x, d, axis=0) * bn.move_mean(y, d, axis=0)) * d / max(d - 1, 1)

def correlation(x, y, d):
    d = _n(d)
    if d < 2: return np.full_like(x, np.nan)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.where(ok, x, np.nan), np.where(ok, y, np.nan)
    sx, sy = bn.move_std(x, d, axis=0, ddof=1), bn.move_std(y, d, axis=0, ddof=1)
    with np.errstate(all="ignore"):
        c = covariance(x, y, d) / (sx * sy)
    c[~np.isfinite(c)] = np.nan
    return np.clip(c, -1, 1)

def scale(x, a=1.0):
    with np.errstate(all="ignore"):
        return x * float(a) / np.nansum(np.abs(x), axis=1, keepdims=True)

def signedpower(x, a):
    with np.errstate(all="ignore"):
        return np.sign(x) * np.abs(x) ** a

def decay_linear(x, d):
    d = _n(d); out = np.zeros_like(x)
    for k in range(d):
        out += (d - k) * (delay(x, k) if k else x)
    return out / (d * (d + 1) / 2)

def ts_min(x, d): return bn.move_min(x, _n(d), axis=0)
def ts_max(x, d): return bn.move_max(x, _n(d), axis=0)
def ts_argmax(x, d): d = _n(d); return d - bn.move_argmax(x, d, axis=0)      # 1 = oldest day in the window, d = today
def ts_argmin(x, d): d = _n(d); return d - bn.move_argmin(x, d, axis=0)
def ts_rank(x, d): return (bn.move_rank(x, _n(d), axis=0) + 1) / 2           # 0 = lowest in the window, 1 = highest
def ts_sum(x, d): return bn.move_sum(x, _n(d), axis=0)
def stddev(x, d): return bn.move_std(x, _n(d), axis=0, ddof=1)

def product(x, d):
    d = _n(d); out = np.ones_like(x)
    for k in range(d):
        out *= (delay(x, k) if k else x)
    return out

def _min(a, b): return ts_min(a, b) if np.isscalar(b) else np.fmin(a, b)
def _max(a, b): return ts_max(a, b) if np.isscalar(b) else np.fmax(a, b)

def _log(x):
    with np.errstate(all="ignore"):
        out = np.log(x)
    return np.where(np.isfinite(out), out, np.nan)

FUNCS = dict(rank=rank, delay=delay, delta=delta, correlation=correlation, covariance=covariance, scale=scale,
             signedpower=signedpower, decay_linear=decay_linear, ts_min=ts_min, ts_max=ts_max, ts_argmax=ts_argmax,
             ts_argmin=ts_argmin, ts_rank=ts_rank, sum=ts_sum, stddev=stddev, product=product, min=_min, max=_max,
             abs=np.abs, log=_log, sign=np.sign)


class Parser:
    """Recursive descent over: ?:  ||  < ==  + -  * /  unary -  ^  calls and brackets."""
    def __init__(self, text, data):
        self.t, self.i, self.data = tokenize(text), 0, data

    def peek(self): return self.t[self.i] if self.i < len(self.t) else None
    def take(self, expected=None):
        tok = self.t[self.i]; self.i += 1
        if expected and tok != expected: raise SyntaxError(f"expected {expected}, got {tok}")
        return tok

    def parse(self):
        v = self.ternary()
        if self.peek() is not None: raise SyntaxError(f"unexpected {self.peek()}")
        return v

    def ternary(self):
        cond = self.logical_or()
        if self.peek() == "?":
            self.take(); a = self.ternary(); self.take(":"); b = self.ternary()
            return np.where(np.nan_to_num(np.asarray(cond, float)) > 0, a, b)
        return cond

    def logical_or(self):
        v = self.compare()
        while self.peek() == "||":
            self.take(); w = self.compare()
            v = ((np.nan_to_num(np.asarray(v, float)) > 0) | (np.nan_to_num(np.asarray(w, float)) > 0)).astype(float)
        return v

    def compare(self):
        v = self.additive()
        while self.peek() in ("<", "=="):
            op = self.take(); w = self.additive()
            with np.errstate(invalid="ignore"):
                v = ((v < w) if op == "<" else (v == w)).astype(float) if isinstance(v, np.ndarray) or isinstance(w, np.ndarray) else float((v < w) if op == "<" else (v == w))
        return v

    def additive(self):
        v = self.term()
        while self.peek() in ("+", "-"):
            v = v + self.term() if self.take() == "+" else v - self.term()
        return v

    def term(self):
        v = self.unary()
        while self.peek() in ("*", "/"):
            op = self.take(); w = self.unary()
            with np.errstate(all="ignore"):
                v = v * w if op == "*" else v / w
            if isinstance(v, np.ndarray): v = np.where(np.isfinite(v), v, np.nan)
        return v

    def unary(self):
        if self.peek() == "-":
            self.take(); return -self.unary()
        return self.power()

    def power(self):
        v = self.primary()
        if self.peek() == "^":
            self.take(); w = self.unary()
            with np.errstate(all="ignore"):
                v = np.power(v, w)
            if isinstance(v, np.ndarray): v = np.where(np.isfinite(v), v, np.nan)
        return v

    def primary(self):
        tok = self.take()
        if tok == "(":
            v = self.ternary(); self.take(")"); return v
        if tok[0].isdigit() or tok[0] == ".":
            return float(tok)
        name = tok.lower()
        if self.peek() == "(":
            self.take(); args = []
            while self.peek() != ")":
                args.append(self.ternary())
                if self.peek() == ",": self.take()
            self.take(")")
            if name == "indneutralize":
                return self.neutralize(args[0], args[1])
            return FUNCS[name](*args)
        if name.startswith("indclass."):
            return name.split(".")[1]
        return self.data.get(name)

    def neutralize(self, x, level):
        labels = self.data.groups["sector" if level == "sector" else "sub_industry"]
        out = np.full_like(x, np.nan)
        for g in np.unique(labels):
            cols = labels == g
            with np.errstate(all="ignore"):
                out[:, cols] = x[:, cols] - np.nanmean(x[:, cols], axis=1, keepdims=True)
        return out


def evaluate(formula, data):
    v = Parser(formula, data).parse()
    v = np.asarray(v, float)
    return np.where(np.isfinite(v), v, np.nan)
