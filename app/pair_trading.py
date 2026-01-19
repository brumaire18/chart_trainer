from itertools import combinations, islice
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from app.data_loader import load_price_csv


MIN_PAIR_SAMPLES = 120


def generate_pair_candidates(
    listed_df: pd.DataFrame,
    symbols: List[str],
    sector17: Optional[str] = None,
    sector33: Optional[str] = None,
    max_pairs: int = 50,
) -> List[Tuple[str, str]]:
    available = set(symbols)
    df = listed_df.copy()
    if "code" not in df.columns:
        return []
    df["code"] = df["code"].astype(str).str.zfill(4)
    df = df[df["code"].isin(available)]
    if sector17 and "sector17" in df.columns:
        df = df[df["sector17"].astype(str) == sector17]
    if sector33 and "sector33" in df.columns:
        df = df[df["sector33"].astype(str) == sector33]
    codes = sorted(df["code"].dropna().unique().tolist())
    return list(islice(combinations(codes, 2), max_pairs))


def evaluate_pair_candidates(pairs: Iterable[Tuple[str, str]]) -> pd.DataFrame:
    results = []
    for symbol_a, symbol_b in pairs:
        metrics = compute_pair_metrics(symbol_a, symbol_b)
        if metrics is not None:
            results.append(metrics)
    return pd.DataFrame(results)


def compute_pair_metrics(symbol_a: str, symbol_b: str) -> Optional[dict]:
    df_pair = _prepare_pair_frame(symbol_a, symbol_b)
    if len(df_pair) < MIN_PAIR_SAMPLES:
        return None
    log_a = np.log(df_pair["close_a"])
    log_b = np.log(df_pair["close_b"])
    beta, alpha = np.polyfit(log_b, log_a, 1)
    spread = log_a - (beta * log_b + alpha)
    spread_std = float(spread.std(ddof=0))
    if spread_std == 0 or np.isnan(spread_std):
        return None
    p_value = _compute_cointegration_pvalue(log_a, log_b)
    half_life = _compute_half_life(spread)
    zscore = (spread - spread.mean()) / spread_std
    return {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "p_value": p_value,
        "half_life": half_life,
        "beta": float(beta),
        "spread_mean": float(spread.mean()),
        "spread_std": spread_std,
        "spread_latest": float(spread.iloc[-1]),
        "zscore_latest": float(zscore.iloc[-1]),
    }


def compute_spread_series(symbol_a: str, symbol_b: str) -> Tuple[pd.DataFrame, dict]:
    df_pair = _prepare_pair_frame(symbol_a, symbol_b)
    if df_pair.empty:
        return df_pair, {}
    log_a = np.log(df_pair["close_a"])
    log_b = np.log(df_pair["close_b"])
    beta, alpha = np.polyfit(log_b, log_a, 1)
    spread = log_a - (beta * log_b + alpha)
    spread_mean = float(spread.mean())
    spread_std = float(spread.std(ddof=0))
    zscore = (spread - spread_mean) / spread_std if spread_std else np.nan
    df_pair = df_pair.assign(
        spread=spread,
        zscore=zscore,
    )
    metrics = {
        "beta": float(beta),
        "alpha": float(alpha),
        "spread_mean": spread_mean,
        "spread_std": spread_std,
    }
    return df_pair, metrics


def _prepare_pair_frame(symbol_a: str, symbol_b: str) -> pd.DataFrame:
    df_a = load_price_csv(symbol_a).loc[:, ["date", "close"]]
    df_b = load_price_csv(symbol_b).loc[:, ["date", "close"]]
    df_a = df_a.rename(columns={"close": "close_a"})
    df_b = df_b.rename(columns={"close": "close_b"})
    df_pair = pd.merge(df_a, df_b, on="date", how="inner").dropna()
    df_pair = df_pair[(df_pair["close_a"] > 0) & (df_pair["close_b"] > 0)]
    return df_pair.sort_values("date")


def _compute_cointegration_pvalue(series_a: pd.Series, series_b: pd.Series) -> float:
    _, p_value, _ = coint(series_a, series_b)
    return float(p_value)


def _compute_half_life(spread: pd.Series) -> Optional[float]:
    spread_lag = spread.shift(1)
    delta = spread - spread_lag
    df = pd.concat([spread_lag, delta], axis=1).dropna()
    if df.empty:
        return None
    slope, _ = np.polyfit(df.iloc[:, 0], df.iloc[:, 1], 1)
    if slope >= 0:
        return None
    return float(-np.log(2) / slope)
