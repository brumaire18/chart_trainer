"""Pair trading utilities for generating candidates and evaluating spreads."""

from dataclasses import dataclass
from itertools import combinations, islice, product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def generate_pairs_by_sector(
    listed_master_df: Optional[pd.DataFrame] = None,
    sector_col: Optional[str] = None,
    min_symbols: int = 2,
    max_pairs_per_sector: int = 50,
) -> pd.DataFrame:
    """Generate pair candidates using sector17/sector33 columns.

    Args:
        listed_master_df: DataFrame that includes code/name and sector columns.
            When omitted, listed_master.csv is loaded automatically.
        sector_col: Target sector column name. If None, sector17/sector33 are used.
        min_symbols: Minimum symbols per sector to generate pairs.
        max_pairs_per_sector: Maximum pairs to generate per sector.

    Returns:
        DataFrame with pair metadata (pair name, symbols, sector info).
    """
    if min_symbols < 2:
        raise ValueError("min_symbols must be >= 2")
    if max_pairs_per_sector < 1:
        raise ValueError("max_pairs_per_sector must be >= 1")

    if listed_master_df is None:
        from app.jquants_fetcher import load_listed_master

        listed_master_df = load_listed_master()

    df = listed_master_df.copy()
    if "code" not in df.columns:
        raise ValueError("listed_master_df must contain 'code' column")

    available_sector_cols = [col for col in ["sector17", "sector33"] if col in df.columns]
    if sector_col:
        sector_cols = [sector_col] if sector_col in df.columns else []
    else:
        sector_cols = available_sector_cols
    if not sector_cols:
        raise ValueError("listed_master_df must contain sector17 or sector33 column")

    pairs: List[Dict[str, object]] = []
    df["code"] = df["code"].astype(str).str.zfill(4)
    name_map = df.set_index("code")["name"].to_dict() if "name" in df.columns else {}

    for resolved_sector_col in sector_cols:
        sector_df = df.dropna(subset=[resolved_sector_col, "code"])
        for sector_value, group in sector_df.groupby(resolved_sector_col):
            codes = group["code"].astype(str).tolist()
            if len(codes) < min_symbols:
                continue
            for code_x, code_y in islice(combinations(codes, 2), max_pairs_per_sector):
                pairs.append(
                    {
                        "pair": f"{code_x}-{code_y}",
                        "symbol_x": code_x,
                        "symbol_y": code_y,
                        "name_x": name_map.get(code_x),
                        "name_y": name_map.get(code_y),
                        "sector_type": resolved_sector_col,
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

try:
    from statsmodels.tsa.stattools import coint
except ImportError:  # pragma: no cover - handled gracefully when optional dependency missing
    coint = None

from app.data_loader import load_price_csv


MIN_PAIR_SAMPLES = 120


@dataclass(frozen=True)
class PairTradeConfig:
    """Configuration for pair trading backtests."""

    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_holding_days: int = 20


def backtest_pairs(
    pairs: Union[Iterable[Tuple[str, str]], pd.DataFrame],
    config: PairTradeConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest pair trading strategy for multiple pairs."""
    trades: List[Dict[str, object]] = []
    pair_rows = _normalize_pair_list(pairs)

    for symbol_x, symbol_y, pair_name in pair_rows:
        df_pair = _prepare_pair_prices(symbol_x, symbol_y, start_date, end_date)
        if len(df_pair) <= config.lookback:
            continue
        trades.extend(_backtest_single_pair(df_pair, symbol_x, symbol_y, pair_name, config))

    trades_df = pd.DataFrame(trades)
    summary_df = _summarize_trades(trades_df)
    return trades_df, summary_df


def optimize_pair_trade_parameters(
    pairs: Union[Iterable[Tuple[str, str]], pd.DataFrame],
    param_grid: Dict[str, List[float]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_trades: int = 5,
) -> pd.DataFrame:
    """Grid search for pair trading parameters."""
    grid_keys = ["lookback", "entry_z", "exit_z", "stop_z", "max_holding_days"]
    for key in grid_keys:
        if key not in param_grid:
            raise ValueError(f"param_grid must include '{key}'")

    results: List[Dict[str, object]] = []
    for values in product(*(param_grid[key] for key in grid_keys)):
        config = PairTradeConfig(
            lookback=int(values[0]),
            entry_z=float(values[1]),
            exit_z=float(values[2]),
            stop_z=float(values[3]),
            max_holding_days=int(values[4]),
        )
        trades_df, _ = backtest_pairs(
            pairs,
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        trade_count = len(trades_df)
        if trade_count < min_trades:
            continue
        win_rate = float((trades_df["pnl"] > 0).mean()) if trade_count else 0.0
        total_pnl = float(trades_df["pnl"].sum()) if trade_count else 0.0
        avg_pnl = float(trades_df["pnl"].mean()) if trade_count else 0.0
        results.append(
            {
                "lookback": config.lookback,
                "entry_z": config.entry_z,
                "exit_z": config.exit_z,
                "stop_z": config.stop_z,
                "max_holding_days": config.max_holding_days,
                "trade_count": trade_count,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
            }
        )
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("total_pnl", ascending=False).reset_index(drop=True)


def _normalize_pair_list(
    pairs: Union[Iterable[Tuple[str, str]], pd.DataFrame],
) -> List[Tuple[str, str, str]]:
    if isinstance(pairs, pd.DataFrame):
        if {"symbol_x", "symbol_y"}.issubset(pairs.columns):
            source = pairs
            symbol_x_col = "symbol_x"
            symbol_y_col = "symbol_y"
            pair_col = "pair" if "pair" in pairs.columns else None
        elif {"symbol_a", "symbol_b"}.issubset(pairs.columns):
            source = pairs
            symbol_x_col = "symbol_a"
            symbol_y_col = "symbol_b"
            pair_col = None
        else:
            raise ValueError("pairs DataFrame must include symbol_x/symbol_y or symbol_a/symbol_b")
        return [
            (
                str(row[symbol_x_col]).zfill(4),
                str(row[symbol_y_col]).zfill(4),
                str(row[pair_col]) if pair_col and pd.notna(row[pair_col]) else None,
            )
            for _, row in source.iterrows()
        ]

    normalized = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("pairs must be tuples of (symbol_x, symbol_y)")
        symbol_x, symbol_y = pair
        normalized.append((str(symbol_x).zfill(4), str(symbol_y).zfill(4), None))
    return normalized


def _prepare_pair_prices(
    symbol_x: str,
    symbol_y: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    df_x = load_price_csv(symbol_x).loc[:, ["date", "close"]].rename(columns={"close": "close_x"})
    df_y = load_price_csv(symbol_y).loc[:, ["date", "close"]].rename(columns={"close": "close_y"})
    df_pair = pd.merge(df_x, df_y, on="date", how="inner").dropna()
    df_pair = df_pair[(df_pair["close_x"] > 0) & (df_pair["close_y"] > 0)]
    if start_date:
        df_pair = df_pair[df_pair["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df_pair = df_pair[df_pair["date"] <= pd.to_datetime(end_date)]
    return df_pair.sort_values("date").reset_index(drop=True)


def _backtest_single_pair(
    df_pair: pd.DataFrame,
    symbol_x: str,
    symbol_y: str,
    pair_name: Optional[str],
    config: PairTradeConfig,
) -> List[Dict[str, object]]:
    trades: List[Dict[str, object]] = []
    position = 0
    entry_idx = None
    entry_spread = None
    entry_z = None

    for idx in range(config.lookback, len(df_pair)):
        window = df_pair.iloc[idx - config.lookback : idx]
        beta, c = estimate_beta_ols(window["close_x"], window["close_y"])
        if np.isnan(beta) or np.isnan(c):
            continue
        spread_window = compute_spread(window["close_x"], window["close_y"], beta, c)
        spread_mean = spread_window.mean()
        spread_std = spread_window.std(ddof=0)
        if spread_std == 0 or np.isnan(spread_std):
            continue
        current_row = df_pair.iloc[idx]
        current_spread = compute_spread(
            pd.Series([current_row["close_x"]]),
            pd.Series([current_row["close_y"]]),
            beta,
            c,
        ).iloc[0]
        zscore_value = (current_spread - spread_mean) / spread_std
        date_value = current_row["date"]

        if position == 0:
            if zscore_value >= config.entry_z:
                position = -1
                entry_idx = idx
                entry_spread = current_spread
                entry_z = zscore_value
            elif zscore_value <= -config.entry_z:
                position = 1
                entry_idx = idx
                entry_spread = current_spread
                entry_z = zscore_value
            continue

        holding_days = idx - entry_idx if entry_idx is not None else 0
        exit_reason = None
        if abs(zscore_value) <= config.exit_z:
            exit_reason = "exit_z"
        elif abs(zscore_value) >= config.stop_z:
            exit_reason = "stop_z"
        elif holding_days >= config.max_holding_days:
            exit_reason = "max_holding_days"

        if exit_reason:
            pnl = position * (current_spread - entry_spread)
            trades.append(
                {
                    "pair": pair_name or f"{symbol_x}-{symbol_y}",
                    "symbol_x": symbol_x,
                    "symbol_y": symbol_y,
                    "entry_date": df_pair.iloc[entry_idx]["date"],
                    "exit_date": date_value,
                    "entry_z": float(entry_z),
                    "exit_z": float(zscore_value),
                    "exit_reason": exit_reason,
                    "holding_days": holding_days,
                    "pnl": float(pnl),
                }
            )
            position = 0
            entry_idx = None
            entry_spread = None
            entry_z = None

    return trades


def _summarize_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    grouped = trades_df.groupby("pair")
    summary = grouped.agg(
        trade_count=("pnl", "size"),
        win_rate=("pnl", lambda x: float((x > 0).mean()) if len(x) else 0.0),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    )
    summary = summary.reset_index()
    return summary.sort_values("total_pnl", ascending=False).reset_index(drop=True)


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
    if coint is None:
        warnings.warn(
            "statsmodels is not available; cointegration p-values will be NaN. "
            "Install statsmodels to enable cointegration statistics.",
            RuntimeWarning,
        )
        return float("nan")
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
