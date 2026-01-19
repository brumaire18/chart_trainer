"""Pair trading utilities for generating candidates and evaluating spreads."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def generate_pairs_by_sector(
    listed_master_df: pd.DataFrame,
    min_symbols: int = 2,
) -> pd.DataFrame:
    """Generate pair candidates using sector17/sector33 columns.

    Args:
        listed_master_df: DataFrame that includes code/name and sector columns.
        min_symbols: Minimum symbols per sector to generate pairs.

    Returns:
        DataFrame with pair metadata (pair name, symbols, sector info).
    """
    if min_symbols < 2:
        raise ValueError("min_symbols must be >= 2")

    sector_cols = [col for col in ["sector17", "sector33"] if col in listed_master_df.columns]
    if not sector_cols:
        raise ValueError("listed_master_df must contain sector17 or sector33 column")

    pairs: List[Dict[str, object]] = []
    df = listed_master_df.copy()
    if "code" not in df.columns:
        raise ValueError("listed_master_df must contain 'code' column")

    df["code"] = df["code"].astype(str)
    name_map = df.set_index("code")["name"].to_dict() if "name" in df.columns else {}

    for sector_col in sector_cols:
        sector_df = df.dropna(subset=[sector_col, "code"])
        for sector_value, group in sector_df.groupby(sector_col):
            codes = group["code"].astype(str).tolist()
            if len(codes) < min_symbols:
                continue
            for idx, code_x in enumerate(codes[:-1]):
                for code_y in codes[idx + 1 :]:
                    pairs.append(
                        {
                            "pair": f"{code_x}-{code_y}",
                            "symbol_x": code_x,
                            "symbol_y": code_y,
                            "name_x": name_map.get(code_x),
                            "name_y": name_map.get(code_y),
                            "sector_type": sector_col,
                            "sector_code": sector_value,
                        }
                    )

    return pd.DataFrame(pairs)


def estimate_beta_ols(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Estimate hedge ratio with OLS using log prices.

    Args:
        x: Price series of the first asset.
        y: Price series of the second asset.

    Returns:
        (beta, c) where spread = log(y) - (beta * log(x) + c)
    """
    df = pd.concat([x, y], axis=1).dropna()
    if df.empty:
        return np.nan, np.nan
    x_log = np.log(df.iloc[:, 0])
    y_log = np.log(df.iloc[:, 1])
    x_log.name = "x"
    y_log.name = "y"

    exog = sm.add_constant(x_log)
    model = sm.OLS(y_log, exog).fit()
    beta = model.params["x"]
    c = model.params["const"]
    return float(beta), float(c)


def compute_spread(x: pd.Series, y: pd.Series, beta: float, c: float) -> pd.Series:
    """Compute spread series from log prices and hedge ratio."""
    df = pd.concat([x, y], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    x_log = np.log(df.iloc[:, 0])
    y_log = np.log(df.iloc[:, 1])
    spread = y_log - (beta * x_log + c)
    spread.name = "spread"
    return spread


def zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling z-score for a spread series."""
    if window <= 1:
        raise ValueError("window must be > 1")
    spread = pd.Series(spread).copy()
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std(ddof=0)
    return (spread - rolling_mean) / rolling_std


def adf_test(spread: pd.Series) -> Dict[str, float]:
    """Run ADF test on a spread series.

    Returns dict with adf_stat, p_value, and nobs.
    """
    series = pd.Series(spread).dropna()
    if len(series) < 5:
        return {"adf_stat": np.nan, "p_value": np.nan, "nobs": float(len(series))}
    result = adfuller(series, autolag="AIC")
    return {"adf_stat": float(result[0]), "p_value": float(result[1]), "nobs": float(result[3])}


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate half-life of mean reversion for a spread series."""
    series = pd.Series(spread).dropna()
    if len(series) < 2:
        return np.nan
    lagged = series.shift(1).iloc[1:]
    lagged.name = "lagged"
    delta = series.diff().iloc[1:]
    exog = sm.add_constant(lagged)
    model = sm.OLS(delta, exog).fit()
    beta = model.params["lagged"]
    if beta >= 0:
        return np.inf
    return float(-np.log(2) / beta)


def estimate_beta_stability(x: pd.Series, y: pd.Series, window: int = 60) -> float:
    """Estimate beta stability using rolling OLS beta standard deviation."""
    if window < 5:
        raise ValueError("window must be >= 5")
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < window:
        return np.nan

    betas: List[float] = []
    for end in range(window, len(df) + 1):
        sub = df.iloc[end - window : end]
        beta, _ = estimate_beta_ols(sub.iloc[:, 0], sub.iloc[:, 1])
        if not np.isnan(beta):
            betas.append(beta)

    if not betas:
        return np.nan
    return float(np.std(betas, ddof=1)) if len(betas) > 1 else 0.0


def evaluate_pair_metrics(
    x: pd.Series,
    y: pd.Series,
    pair_name: Optional[str] = None,
    window: int = 60,
) -> pd.DataFrame:
    """Evaluate pair metrics and return as a single-row DataFrame."""
    beta, c = estimate_beta_ols(x, y)
    spread = compute_spread(x, y, beta, c)
    adf_result = adf_test(spread)
    half_life = estimate_half_life(spread)
    beta_stability = estimate_beta_stability(x, y, window=window)

    result = {
        "pair": pair_name or f"{getattr(x, 'name', 'x')}-{getattr(y, 'name', 'y')}",
        "beta": beta,
        "c": c,
        "adf_p_value": adf_result["p_value"],
        "adf_stat": adf_result["adf_stat"],
        "half_life": half_life,
        "beta_stability": beta_stability,
        "n_obs": float(len(spread.dropna())),
    }
    return pd.DataFrame([result])


def evaluate_pairs(
    price_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """Evaluate multiple pairs and return a summary DataFrame."""
    results: List[Dict[str, object]] = []
    for _, row in pairs_df.iterrows():
        symbol_x = row.get("symbol_x")
        symbol_y = row.get("symbol_y")
        if symbol_x not in price_df.columns or symbol_y not in price_df.columns:
            continue
        metrics = evaluate_pair_metrics(
            price_df[symbol_x],
            price_df[symbol_y],
            pair_name=row.get("pair"),
            window=window,
        ).iloc[0]
        payload = metrics.to_dict()
        for key in ["sector_type", "sector_code", "name_x", "name_y"]:
            if key in row and pd.notna(row[key]):
                payload[key] = row[key]
        results.append(payload)

    return pd.DataFrame(results)
