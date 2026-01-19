from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import load_price_csv
from .jquants_fetcher import load_listed_master


@dataclass
class PairTradeConfig:
    lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_holding_days: int = 20
    use_log_price: bool = True


def generate_pairs_by_sector(
    sector_col: str = "sector33",
    min_symbols: int = 2,
    max_pairs_per_sector: Optional[int] = None,
) -> List[Tuple[str, str]]:
    df_master = load_listed_master()
    if sector_col not in df_master.columns:
        raise ValueError(f"{sector_col} が listed_master に存在しません。")
    df = df_master.dropna(subset=["code", sector_col]).copy()
    df["code"] = df["code"].astype(str).str.zfill(4)

    pairs: List[Tuple[str, str]] = []
    for _, group in df.groupby(sector_col):
        symbols = sorted(group["code"].unique().tolist())
        if len(symbols) < min_symbols:
            continue
        count = 0
        for i in range(len(symbols) - 1):
            for j in range(i + 1, len(symbols)):
                pairs.append((symbols[i], symbols[j]))
                count += 1
                if max_pairs_per_sector is not None and count >= max_pairs_per_sector:
                    break
            if max_pairs_per_sector is not None and count >= max_pairs_per_sector:
                break
    return pairs


def _prepare_pair_dataframe(
    symbol_a: str,
    symbol_b: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_log_price: bool = True,
) -> pd.DataFrame:
    df_a = load_price_csv(symbol_a).loc[:, ["date", "close"]].rename(
        columns={"close": "close_a"}
    )
    df_b = load_price_csv(symbol_b).loc[:, ["date", "close"]].rename(
        columns={"close": "close_b"}
    )
    df = pd.merge(df_a, df_b, on="date", how="inner").sort_values("date")
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    if use_log_price:
        df["x"] = np.log(df["close_a"])
        df["y"] = np.log(df["close_b"])
    else:
        df["x"] = df["close_a"]
        df["y"] = df["close_b"]
    return df.reset_index(drop=True)


def _estimate_beta(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) == 0:
        return 0.0, 0.0
    design = np.column_stack([y, np.ones_like(y)])
    coef, _, _, _ = np.linalg.lstsq(design, x, rcond=None)
    beta = float(coef[0])
    intercept = float(coef[1])
    return beta, intercept


def _rolling_beta(series_x: pd.Series, series_y: pd.Series, window: int) -> pd.DataFrame:
    betas = np.full(len(series_x), np.nan, dtype=float)
    intercepts = np.full(len(series_x), np.nan, dtype=float)
    for idx in range(window - 1, len(series_x)):
        x = series_x.iloc[idx - window + 1 : idx + 1].to_numpy()
        y = series_y.iloc[idx - window + 1 : idx + 1].to_numpy()
        beta, intercept = _estimate_beta(x, y)
        betas[idx] = beta
        intercepts[idx] = intercept
    return pd.DataFrame({"beta": betas, "intercept": intercepts})


def _compute_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    return float(drawdown.min())


def backtest_pair(
    symbol_a: str,
    symbol_b: str,
    config: Optional[PairTradeConfig] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    cfg = config or PairTradeConfig()
    df = _prepare_pair_dataframe(
        symbol_a,
        symbol_b,
        start_date=start_date,
        end_date=end_date,
        use_log_price=cfg.use_log_price,
    )
    if df.empty or len(df) < cfg.lookback:
        return pd.DataFrame(), pd.DataFrame(), {}

    beta_df = _rolling_beta(df["x"], df["y"], cfg.lookback)
    df = pd.concat([df, beta_df], axis=1)
    df["spread"] = df["x"] - df["beta"] * df["y"] - df["intercept"]
    df["spread_mean"] = df["spread"].rolling(cfg.lookback).mean()
    df["spread_std"] = df["spread"].rolling(cfg.lookback).std(ddof=0)
    df["zscore"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

    position = 0
    entry_spread = 0.0
    entry_date = None
    entry_z = 0.0
    holding_days = 0
    prev_spread = np.nan

    trades: List[Dict[str, float]] = []
    daily_records: List[Dict[str, float]] = []
    cumulative_pnl = 0.0

    for idx, row in df.iterrows():
        if idx < cfg.lookback:
            prev_spread = row["spread"]
            continue
        zscore = float(row["zscore"])
        spread = float(row["spread"])

        daily_pnl = 0.0
        if position != 0 and not np.isnan(prev_spread):
            daily_pnl = position * (spread - prev_spread)
            cumulative_pnl += daily_pnl

        if position == 0:
            if zscore >= cfg.entry_z:
                position = -1
                entry_spread = spread
                entry_date = row["date"]
                entry_z = zscore
                holding_days = 0
            elif zscore <= -cfg.entry_z:
                position = 1
                entry_spread = spread
                entry_date = row["date"]
                entry_z = zscore
                holding_days = 0
        else:
            holding_days += 1
            exit_signal = abs(zscore) <= cfg.exit_z
            stop_signal = abs(zscore) >= cfg.stop_z
            time_signal = holding_days >= cfg.max_holding_days
            if exit_signal or stop_signal or time_signal:
                trade_pnl = position * (spread - entry_spread)
                trades.append(
                    {
                        "symbol_a": symbol_a,
                        "symbol_b": symbol_b,
                        "entry_date": entry_date,
                        "exit_date": row["date"],
                        "position": position,
                        "entry_spread": entry_spread,
                        "exit_spread": spread,
                        "entry_z": entry_z,
                        "exit_z": zscore,
                        "holding_days": holding_days,
                        "pnl": trade_pnl,
                    }
                )
                position = 0
                entry_spread = 0.0
                entry_date = None
                entry_z = 0.0
                holding_days = 0

        daily_records.append(
            {
                "date": row["date"],
                "spread": spread,
                "zscore": zscore,
                "position": position,
                "daily_pnl": daily_pnl,
                "cumulative_pnl": cumulative_pnl,
            }
        )
        prev_spread = spread

    trades_df = pd.DataFrame(trades)
    daily_df = pd.DataFrame(daily_records)
    if trades_df.empty:
        summary = {
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "trade_count": 0,
            "total_pnl": 0.0,
            "avg_trade_pnl": 0.0,
            "win_rate": 0.0,
            "avg_holding_days": 0.0,
            "max_drawdown": 0.0,
        }
        return trades_df, daily_df, summary

    win_rate = float((trades_df["pnl"] > 0).mean())
    equity_curve = daily_df["cumulative_pnl"]
    summary = {
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "trade_count": int(len(trades_df)),
        "total_pnl": float(trades_df["pnl"].sum()),
        "avg_trade_pnl": float(trades_df["pnl"].mean()),
        "win_rate": win_rate,
        "avg_holding_days": float(trades_df["holding_days"].mean()),
        "max_drawdown": _compute_drawdown(equity_curve),
    }
    return trades_df, daily_df, summary


def backtest_pairs(
    pairs: Iterable[Tuple[str, str]],
    config: Optional[PairTradeConfig] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trade_frames: List[pd.DataFrame] = []
    summary_records: List[Dict[str, float]] = []

    for symbol_a, symbol_b in pairs:
        trades_df, _, summary = backtest_pair(
            symbol_a,
            symbol_b,
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        if not trades_df.empty:
            trade_frames.append(trades_df)
        if summary:
            summary_records.append(summary)

    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_records).sort_values(
        ["total_pnl", "win_rate"], ascending=False
    )
    return trades_all, summary_df


def optimize_pair_trade_parameters(
    pairs: Iterable[Tuple[str, str]],
    param_grid: Dict[str, List[float]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_trades: int = 5,
) -> pd.DataFrame:
    """
    パラメータのグリッドサーチを行い、ペア全体の平均成績を評価する。

    param_grid には lookback, entry_z, exit_z, stop_z, max_holding_days を渡す。
    """

    grid_keys = ["lookback", "entry_z", "exit_z", "stop_z", "max_holding_days"]
    missing_keys = [key for key in grid_keys if key not in param_grid]
    if missing_keys:
        raise ValueError(f"param_grid に不足しているキーがあります: {missing_keys}")

    eval_records: List[Dict[str, float]] = []

    for values in product(
        param_grid["lookback"],
        param_grid["entry_z"],
        param_grid["exit_z"],
        param_grid["stop_z"],
        param_grid["max_holding_days"],
    ):
        config = PairTradeConfig(
            lookback=int(values[0]),
            entry_z=float(values[1]),
            exit_z=float(values[2]),
            stop_z=float(values[3]),
            max_holding_days=int(values[4]),
        )
        _, summary_df = backtest_pairs(
            pairs,
            config=config,
            start_date=start_date,
            end_date=end_date,
        )
        if summary_df.empty:
            continue

        valid = summary_df[summary_df["trade_count"] >= min_trades]
        if valid.empty:
            continue

        eval_records.append(
            {
                "lookback": config.lookback,
                "entry_z": config.entry_z,
                "exit_z": config.exit_z,
                "stop_z": config.stop_z,
                "max_holding_days": config.max_holding_days,
                "avg_total_pnl": float(valid["total_pnl"].mean()),
                "avg_win_rate": float(valid["win_rate"].mean()),
                "avg_trade_pnl": float(valid["avg_trade_pnl"].mean()),
                "pair_count": int(valid.shape[0]),
            }
        )

    if not eval_records:
        return pd.DataFrame()

    return pd.DataFrame(eval_records).sort_values(
        ["avg_total_pnl", "avg_win_rate"], ascending=False
    )
