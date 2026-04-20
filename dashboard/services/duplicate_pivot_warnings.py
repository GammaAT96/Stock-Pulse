"""Once-per-(component, symbol-universe) warnings for duplicate (trade_date, symbol) rows before pivot."""

from __future__ import annotations

import hashlib
import logging
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

_WARNED_DUP_KEYS: set[tuple[str, tuple[str, ...]]] = set()
_LOCK = Lock()


def clear_duplicate_pivot_warnings_cache() -> None:
    """Reset dedupe keys (e.g. between tests or long-lived workers)."""
    with _LOCK:
        _WARNED_DUP_KEYS.clear()


def warn_once_duplicate_trade_symbol_rows(logger: logging.Logger, location: str, work: "pd.DataFrame") -> None:
    """
    If ``work`` has duplicate (trade_date, symbol) rows, log at most once per process for
    ``(location, first 10 sorted symbols)`` — avoids log floods on very dirty panels.
    """
    if work.empty or "symbol" not in work.columns or "trade_date" not in work.columns:
        return
    w = work.sort_values(["trade_date", "symbol"])
    dupe_count = int(w.duplicated(subset=["trade_date", "symbol"]).sum())
    if dupe_count <= 0:
        return

    sym_uniq = sorted({str(s) for s in w["symbol"].dropna().unique()})
    key = (location, tuple(sym_uniq[:10]))
    fingerprint = hashlib.md5(",".join(sym_uniq).encode()).hexdigest()[:8]

    with _LOCK:
        if key in _WARNED_DUP_KEYS:
            return
        _WARNED_DUP_KEYS.add(key)

    dup_mask = w.duplicated(subset=["trade_date", "symbol"], keep=False)
    dup_syms = w.loc[dup_mask, "symbol"].dropna().astype(str).unique().tolist()
    n_symbols = int(w["symbol"].nunique())
    shown = dup_syms[:5]
    more = f" (+{len(dup_syms) - 5} more symbols)" if len(dup_syms) > 5 else ""

    logger.warning(
        "[%s] %s duplicate rows across %s symbols (fp=%s); sample: %s%s",
        location,
        dupe_count,
        n_symbols,
        fingerprint,
        shown,
        more,
        extra={
            "dupe_count": dupe_count,
            "n_symbols": n_symbols,
            "sample_symbols": list(shown),
            "affected_symbol_count": len(dup_syms),
            "pivot_component": location,
            "fingerprint": fingerprint,
        },
    )
