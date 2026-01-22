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
    df = _exclude_index_etfs(df)

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


def compute_min_pair_samples(recent_window: int, long_window: Optional[int] = None) -> int:
    """Compute minimum samples needed for pair metrics based on windows."""
    return max(recent_window + 1, (long_window or 0) + 1)


def _trim_pair_history(df_pair: pd.DataFrame, history_window: Optional[int]) -> pd.DataFrame:
    if history_window is None:
        return df_pair
    if history_window <= 0:
        return df_pair.iloc[0:0]
    if len(df_pair) <= history_window:
        return df_pair
    return df_pair.tail(history_window).reset_index(drop=True)


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
    try:
        df_x = load_price_csv(symbol_x).loc[:, ["date", "close"]].rename(
            columns={"close": "close_x"}
        )
    except FileNotFoundError:
        return pd.DataFrame()
    try:
        df_y = load_price_csv(symbol_y).loc[:, ["date", "close"]].rename(
            columns={"close": "close_y"}
        )
    except FileNotFoundError:
        return pd.DataFrame()
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
    df = _exclude_index_etfs(df)
    df["code"] = df["code"].astype(str).str.zfill(4)
    df = df[df["code"].isin(available)]
    if sector17 and "sector17" in df.columns:
        df = df[df["sector17"].astype(str) == sector17]
    if sector33 and "sector33" in df.columns:
        df = df[df["sector33"].astype(str) == sector33]
    codes = sorted(df["code"].dropna().unique().tolist())
    return list(islice(combinations(codes, 2), max_pairs))


def _is_etf_name(name: str) -> bool:
    upper_name = name.upper()
    return "ETF" in upper_name or "上場投信" in name


def _extract_etf_index_tag(name: str) -> Optional[str]:
    if not name or not _is_etf_name(name):
        return None
    normalized = (
        name.upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("・", "")
        .replace("＆", "")
    )
    index_aliases = {
        "TOPIX": ["TOPIX"],
        "NIKKEI225": ["日経225", "NIKKEI225", "NIKKEI 225", "NIKKEI２２５"],
        "JPXNIKKEI400": ["JPX日経400", "JPXNIKKEI400", "JPX-NIKKEI400"],
        "S&P500": ["S&P500", "S&P 500", "SP500"],
        "NASDAQ100": ["NASDAQ100", "NASDAQ 100", "NASDAQ-100"],
        "DOW": ["DOW", "ダウ", "NYDOW", "DOWJONES"],
        "MSCI": ["MSCI"],
        "FTSE": ["FTSE"],
        "REIT": ["REIT"],
        "RUSSELL": ["RUSSELL", "ラッセル"],
        "TOPIXCORE30": ["TOPIXCORE30", "TOPIX CORE30", "TOPIXコア30"],
        "TOPIXSMALL": ["TOPIXSMALL", "TOPIX SMALL"],
    }
    for tag, candidates in index_aliases.items():
        for keyword in candidates:
            if keyword.upper().replace(" ", "").replace("-", "") in normalized:
                return tag
    return None


def _build_etf_index_map(listed_df: pd.DataFrame) -> Dict[str, Optional[str]]:
    if listed_df is None or listed_df.empty:
        return {}
    if "code" not in listed_df.columns or "name" not in listed_df.columns:
        return {}
    df = listed_df.loc[:, ["code", "name"]].copy()
    df["code"] = df["code"].astype(str).str.zfill(4)
    return {
        row.code: _extract_etf_index_tag(str(row.name))
        for row in df.itertuples(index=False)
    }


def _is_topix_etf_tag(tag: Optional[str]) -> bool:
    return bool(tag) and str(tag).upper().startswith("TOPIX")


def _exclude_index_etfs(listed_df: pd.DataFrame) -> pd.DataFrame:
    if listed_df is None or listed_df.empty:
        return listed_df
    if "name" not in listed_df.columns:
        return listed_df
    df = listed_df.copy()
    df["__etf_tag__"] = df["name"].apply(lambda name: _extract_etf_index_tag(str(name)))
    df = df[df["__etf_tag__"].isna()].drop(columns=["__etf_tag__"])
    return df


def generate_pairs_by_sector_candidates(
    listed_df: pd.DataFrame,
    symbols: List[str],
    sector17: Optional[str] = None,
    sector33: Optional[str] = None,
    max_pairs_per_sector: Optional[int] = 50,
) -> List[Tuple[str, str]]:
    available = set(symbols)
    df = listed_df.copy()
    if "code" not in df.columns:
        return []
    df = _exclude_index_etfs(df)
    df["code"] = df["code"].astype(str).str.zfill(4)
    df = df[df["code"].isin(available)]
    if sector17 and "sector17" in df.columns:
        df = df[df["sector17"].astype(str) == sector17]
    if sector33 and "sector33" in df.columns:
        df = df[df["sector33"].astype(str) == sector33]
    if sector33 and "sector33" in df.columns:
        sector_col = "sector33"
    elif sector17 and "sector17" in df.columns:
        sector_col = "sector17"
    elif "sector33" in df.columns:
        sector_col = "sector33"
    elif "sector17" in df.columns:
        sector_col = "sector17"
    else:
        return []
    pairs: List[Tuple[str, str]] = []
    for _, group in df.dropna(subset=[sector_col]).groupby(sector_col):
        codes = sorted(group["code"].dropna().unique().tolist())
        combos = combinations(codes, 2)
        if max_pairs_per_sector is None:
            pairs.extend(list(combos))
        else:
            pairs.extend(list(islice(combos, max_pairs_per_sector)))
    return pairs


def generate_anchor_pair_candidates(
    listed_df: pd.DataFrame,
    symbols: List[str],
    anchor_symbol: str,
    sector17: Optional[str] = None,
    sector33: Optional[str] = None,
    max_pairs: int = 50,
) -> List[Tuple[str, str]]:
    available = set(symbols)
    anchor_symbol = str(anchor_symbol).zfill(4)
    if anchor_symbol not in available:
        return []
    df = listed_df.copy()
    if "code" not in df.columns:
        return []
    df = _exclude_index_etfs(df)
    df["code"] = df["code"].astype(str).str.zfill(4)
    df = df[df["code"].isin(available)]
    if sector17 and "sector17" in df.columns:
        df = df[df["sector17"].astype(str) == sector17]
    if sector33 and "sector33" in df.columns:
        df = df[df["sector33"].astype(str) == sector33]
    if sector33 and "sector33" in df.columns:
        sector_col = "sector33"
    elif sector17 and "sector17" in df.columns:
        sector_col = "sector17"
    elif "sector33" in df.columns:
        sector_col = "sector33"
    elif "sector17" in df.columns:
        sector_col = "sector17"
    else:
        return []
    codes = sorted(df["code"].dropna().unique().tolist())
    if anchor_symbol not in codes:
        return []
    anchor_sector = (
        df.loc[df["code"] == anchor_symbol, sector_col].astype(str).iloc[0]
        if sector_col in df.columns
        else None
    )
    if anchor_sector is not None and sector_col in df.columns:
        df = df[df[sector_col].astype(str) == anchor_sector]
        codes = sorted(df["code"].dropna().unique().tolist())
    candidates = [(anchor_symbol, code) for code in codes if code != anchor_symbol]
    return list(islice(candidates, max_pairs))


def evaluate_pair_candidates(
    pairs: Iterable[Tuple[str, str]],
    recent_window: int = 60,
    long_window: Optional[int] = None,
    min_similarity: Optional[float] = None,
    min_long_similarity: Optional[float] = None,
    min_return_corr: Optional[float] = None,
    max_p_value: Optional[float] = None,
    max_half_life: Optional[float] = None,
    max_abs_zscore: Optional[float] = None,
    min_avg_volume: Optional[float] = None,
    preselect_top_n: Optional[int] = None,
    listed_df: Optional[pd.DataFrame] = None,
    history_window: Optional[int] = None,
) -> pd.DataFrame:
    etf_index_map = _build_etf_index_map(listed_df) if listed_df is not None else {}
    scored_pairs = []
    if preselect_top_n is not None and preselect_top_n > 0:
        for symbol_a, symbol_b in pairs:
            if _is_index_etf_symbol(symbol_a, etf_index_map) or _is_index_etf_symbol(
                symbol_b, etf_index_map
            ):
                continue
            if _is_same_index_etf_pair(symbol_a, symbol_b, etf_index_map):
                continue
            score = _compute_quick_pair_score(
                symbol_a,
                symbol_b,
                recent_window=recent_window,
                min_avg_volume=min_avg_volume,
                history_window=history_window,
            )
            if score is not None:
                scored_pairs.append((score, (symbol_a, symbol_b)))
        scored_pairs.sort(key=lambda item: item[0], reverse=True)
        target_pairs = [pair for _, pair in scored_pairs[:preselect_top_n]]
    else:
        target_pairs = [pair for pair in pairs]
    results = []
    for symbol_a, symbol_b in target_pairs:
        if _is_index_etf_symbol(symbol_a, etf_index_map) or _is_index_etf_symbol(
            symbol_b, etf_index_map
        ):
            continue
        if _is_same_index_etf_pair(symbol_a, symbol_b, etf_index_map):
            continue
        metrics = compute_pair_metrics(
            symbol_a,
            symbol_b,
            recent_window=recent_window,
            long_window=long_window,
            min_similarity=min_similarity,
            min_long_similarity=min_long_similarity,
            min_return_corr=min_return_corr,
            min_avg_volume=min_avg_volume,
            history_window=history_window,
        )
        if metrics is None:
            continue
        if max_p_value is not None:
            p_value = metrics.get("p_value")
            if p_value is None or np.isnan(p_value):
                pass
            elif p_value > max_p_value:
                continue
        if max_half_life is not None:
            half_life = metrics.get("half_life")
            if half_life is None or np.isnan(half_life) or half_life > max_half_life:
                continue
        if max_abs_zscore is not None:
            zscore_latest = metrics.get("zscore_latest")
            if (
                zscore_latest is None
                or np.isnan(zscore_latest)
                or abs(zscore_latest) > max_abs_zscore
            ):
                continue
        results.append(metrics)
    return pd.DataFrame(results)


def _is_same_index_etf_pair(
    symbol_a: str, symbol_b: str, etf_index_map: Dict[str, Optional[str]]
) -> bool:
    if not etf_index_map:
        return False
    tag_a = etf_index_map.get(str(symbol_a).zfill(4))
    tag_b = etf_index_map.get(str(symbol_b).zfill(4))
    return tag_a is not None and tag_a == tag_b


def _is_topix_etf_symbol(symbol: str, etf_index_map: Dict[str, Optional[str]]) -> bool:
    if not etf_index_map:
        return False
    tag = etf_index_map.get(str(symbol).zfill(4))
    return _is_topix_etf_tag(tag)


def _is_index_etf_symbol(symbol: str, etf_index_map: Dict[str, Optional[str]]) -> bool:
    if not etf_index_map:
        return False
    tag = etf_index_map.get(str(symbol).zfill(4))
    return tag is not None


def _compute_quick_pair_score(
    symbol_a: str,
    symbol_b: str,
    recent_window: int,
    min_avg_volume: Optional[float],
    history_window: Optional[int],
) -> Optional[float]:
    df_pair = _prepare_pair_frame(symbol_a, symbol_b)
    df_pair = _trim_pair_history(df_pair, history_window)
    min_samples = compute_min_pair_samples(recent_window)
    if len(df_pair) < min_samples:
        return None
    volume_window = df_pair.tail(recent_window)
    avg_volume_a = float(volume_window["volume_a"].mean())
    avg_volume_b = float(volume_window["volume_b"].mean())
    if min_avg_volume is not None:
        if avg_volume_a < min_avg_volume or avg_volume_b < min_avg_volume:
            return None
    similarity = compute_recent_shape_similarity(
        df_pair["close_a"], df_pair["close_b"], window=recent_window
    )
    if np.isnan(similarity):
        return None
    return_corr = compute_return_correlation(
        df_pair["close_a"], df_pair["close_b"], window=recent_window
    )
    if np.isnan(return_corr):
        return None
    log_a = np.log(df_pair["close_a"])
    log_b = np.log(df_pair["close_b"])
    beta, alpha = np.polyfit(log_b, log_a, 1)
    spread = log_a - (beta * log_b + alpha)
    spread_std = float(spread.std(ddof=0))
    if spread_std == 0 or np.isnan(spread_std):
        return None
    zscore_latest = float((spread.iloc[-1] - spread.mean()) / spread_std)
    half_life = _compute_half_life(spread)
    if half_life is None or np.isnan(half_life):
        return None
    zscore_score = min(abs(zscore_latest), 4.0) * 10.0
    half_life_score = max(0.0, 30.0 - float(half_life)) / 3.0
    similarity_score = float(similarity) * 100.0
    return_score = float(return_corr) * 50.0
    return float(zscore_score + similarity_score + half_life_score + return_score)


def compute_pair_metrics(
    symbol_a: str,
    symbol_b: str,
    recent_window: int = 60,
    min_similarity: Optional[float] = None,
    long_window: Optional[int] = None,
    min_long_similarity: Optional[float] = None,
    min_return_corr: Optional[float] = None,
    min_avg_volume: Optional[float] = None,
    history_window: Optional[int] = None,
) -> Optional[dict]:
    df_pair = _prepare_pair_frame(symbol_a, symbol_b)
    df_pair = _trim_pair_history(df_pair, history_window)
    min_samples = compute_min_pair_samples(recent_window, long_window)
    if len(df_pair) < min_samples:
        return None
    if min_avg_volume is not None:
        volume_window = df_pair.tail(recent_window)
        avg_volume_a = float(volume_window["volume_a"].mean())
        avg_volume_b = float(volume_window["volume_b"].mean())
        if avg_volume_a < min_avg_volume or avg_volume_b < min_avg_volume:
            return None
    recent_similarity = compute_recent_shape_similarity(
        df_pair["close_a"],
        df_pair["close_b"],
        window=recent_window,
    )
    if min_similarity is not None:
        if np.isnan(recent_similarity) or recent_similarity < min_similarity:
            return None
    recent_return_corr = compute_return_correlation(
        df_pair["close_a"],
        df_pair["close_b"],
        window=recent_window,
    )
    if min_return_corr is not None:
        if np.isnan(recent_return_corr) or recent_return_corr < min_return_corr:
            return None
    long_similarity = np.nan
    long_return_corr = np.nan
    if long_window is not None:
        long_similarity = compute_recent_shape_similarity(
            df_pair["close_a"],
            df_pair["close_b"],
            window=long_window,
        )
        if min_long_similarity is not None:
            if not np.isnan(long_similarity) and long_similarity < min_long_similarity:
                return None
        long_return_corr = compute_return_correlation(
            df_pair["close_a"],
            df_pair["close_b"],
            window=long_window,
        )
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
        "recent_similarity": recent_similarity,
        "recent_return_corr": recent_return_corr,
        "long_similarity": float(long_similarity) if long_window is not None else np.nan,
        "long_return_corr": float(long_return_corr) if long_window is not None else np.nan,
        "p_value": p_value,
        "half_life": half_life,
        "beta": float(beta),
        "spread_mean": float(spread.mean()),
        "spread_std": spread_std,
        "spread_latest": float(spread.iloc[-1]),
        "zscore_latest": float(zscore.iloc[-1]),
    }


def compute_recent_shape_similarity(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 60,
) -> float:
    """Compute recent chart shape similarity using correlation of normalized log prices."""
    if window < 5:
        raise ValueError("window must be >= 5")
    df = pd.concat([series_a, series_b], axis=1).dropna()
    if len(df) < window:
        return np.nan
    tail = df.tail(window)
    log_a = np.log(tail.iloc[:, 0])
    log_b = np.log(tail.iloc[:, 1])
    norm_a = log_a - log_a.iloc[0]
    norm_b = log_b - log_b.iloc[0]
    std_a = norm_a.std(ddof=0)
    std_b = norm_b.std(ddof=0)
    if std_a == 0 or std_b == 0 or np.isnan(std_a) or np.isnan(std_b):
        return np.nan
    corr = float(np.corrcoef(norm_a, norm_b)[0, 1])
    return corr


def compute_return_correlation(
    series_a: pd.Series,
    series_b: pd.Series,
    window: int = 60,
) -> float:
    """Compute correlation of log returns for the recent window."""
    if window < 5:
        raise ValueError("window must be >= 5")
    df = pd.concat([series_a, series_b], axis=1).dropna()
    if len(df) < window + 1:
        return np.nan
    tail = df.tail(window + 1)
    log_returns = np.log(tail).diff().dropna()
    if len(log_returns) < window:
        return np.nan
    std_a = log_returns.iloc[:, 0].std(ddof=0)
    std_b = log_returns.iloc[:, 1].std(ddof=0)
    if std_a == 0 or std_b == 0 or np.isnan(std_a) or np.isnan(std_b):
        return np.nan
    return float(np.corrcoef(log_returns.iloc[:, 0], log_returns.iloc[:, 1])[0, 1])


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
    df_a = load_price_csv(symbol_a).loc[:, ["date", "close", "volume"]]
    df_b = load_price_csv(symbol_b).loc[:, ["date", "close", "volume"]]
    df_a = df_a.rename(columns={"close": "close_a", "volume": "volume_a"})
    df_b = df_b.rename(columns={"close": "close_b", "volume": "volume_b"})
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
