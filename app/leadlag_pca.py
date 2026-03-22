from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


BASELINE_MODES = {"mompca", "plainpca", "pca_sub", "double"}

DIAG_TRADABLE = "tradable"
DIAG_INSUFFICIENT_SYMBOLS = "insufficient_symbols"
DIAG_MISSING_DATA = "missing_data"
DIAG_INSUFFICIENT_WINDOW = "insufficient_window"
DIAG_QUANTILE_UNAVAILABLE = "quantile_unavailable"
DIAG_NO_DATA = "no_data"

_EPS = 1e-12


@dataclass
class LeadLagConfig:
    lookback: int = 60
    lambda_reg: float = 0.9
    n_components: int = 3
    quantile_q: float = 0.3
    baseline_mode: str = "pca_sub"
    one_way_cost_bps: float = 10.0
    prior_start: Optional[str] = None
    prior_end: Optional[str] = None
    min_jp_symbols: int = 6
    min_us_symbols: int = 3
    momentum_lookback: int = 20
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def __post_init__(self) -> None:
        if int(self.lookback) <= 1:
            raise ValueError("lookback must be > 1")
        if not (0.0 <= float(self.lambda_reg) <= 1.0):
            raise ValueError("lambda_reg must be in [0, 1]")
        if int(self.n_components) <= 0:
            raise ValueError("n_components must be positive")
        if not (0.0 < float(self.quantile_q) < 0.5):
            raise ValueError("quantile_q must be in (0, 0.5)")
        if str(self.baseline_mode).lower() not in BASELINE_MODES:
            raise ValueError(
                "baseline_mode must be one of: {}".format(",".join(sorted(BASELINE_MODES)))
            )
        if int(self.min_jp_symbols) <= 1:
            raise ValueError("min_jp_symbols must be > 1")
        if int(self.min_us_symbols) <= 0:
            raise ValueError("min_us_symbols must be positive")
        if int(self.momentum_lookback) <= 0:
            raise ValueError("momentum_lookback must be positive")


def _to_ts(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    return pd.to_datetime(value, errors="coerce").normalize()


def _normalize_corr_matrix(corr: np.ndarray) -> np.ndarray:
    corr = np.array(corr, dtype=float, copy=True)
    if corr.size == 0:
        return corr
    corr = (corr + corr.T) / 2.0
    diag = np.diag(corr).copy()
    diag = np.where(diag <= 0, 1.0, diag)
    std = np.sqrt(diag)
    denom = np.outer(std, std)
    denom[denom <= 0] = 1.0
    corr = corr / denom
    corr = np.clip(corr, -0.999, 0.999)
    np.fill_diagonal(corr, 1.0)
    return corr


def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = ["{}{}".format(prefix, col) for col in renamed.columns]
    return renamed


def _empty_backtest_result(config: LeadLagConfig) -> Dict[str, Any]:
    empty_signals = pd.DataFrame(columns=["date", "symbol", "signal", "baseline_mode"])
    empty_weights = pd.DataFrame(columns=["date", "symbol", "weight", "side"])
    empty_daily = pd.DataFrame(
        columns=[
            "date",
            "gross_return",
            "trading_cost",
            "net_return",
            "turnover",
            "equity_curve",
            "gross_exposure",
            "long_exposure",
            "short_exposure",
            "long_count",
            "short_count",
            "is_empty_position",
        ]
    )
    empty_summary = pd.DataFrame(
        [
            {
                "cumulative_return": 0.0,
                "annual_return": 0.0,
                "annual_vol": 0.0,
                "sharpe_like": np.nan,
                "max_drawdown": 0.0,
                "win_rate": np.nan,
                "trading_days": 0,
                "empty_position_days": 0,
                "total_days": 0,
                "lambda_reg": float(config.lambda_reg),
                "baseline_mode": str(config.baseline_mode),
                "quantile_q": float(config.quantile_q),
                "lookback": int(config.lookback),
                "n_components": int(config.n_components),
                "avg_regularization_delta": 0.0,
                "nonzero_regularization_days": 0,
            }
        ]
    )
    diagnostics = {
        "counts": {DIAG_NO_DATA: 1},
        "daily": pd.DataFrame(
            [{"date": pd.NaT, "status": "no_trade", "reason": DIAG_NO_DATA}]
        ),
        "regularization": {"avg_delta": 0.0, "nonzero_days": 0},
        "config": asdict(config),
    }
    return {
        "signals": empty_signals,
        "weights": empty_weights,
        "daily_returns": empty_daily,
        "summary": empty_summary,
        "diagnostics": diagnostics,
    }


def build_prior_exposure_matrix(
    prior_us_returns: pd.DataFrame,
    prior_jp_returns: pd.DataFrame,
    min_obs: int = 20,
) -> pd.DataFrame:
    us_cols = list(prior_us_returns.columns)
    jp_cols = list(prior_jp_returns.columns)
    result = pd.DataFrame(0.0, index=jp_cols, columns=us_cols, dtype=float)
    if prior_us_returns.empty or prior_jp_returns.empty:
        return result

    common_index = prior_us_returns.index.intersection(prior_jp_returns.index)
    if len(common_index) < int(min_obs):
        return result

    us = prior_us_returns.loc[common_index].astype(float)
    jp = prior_jp_returns.loc[common_index].astype(float)
    combined = pd.concat(
        [_prefix_columns(us, "US::"), _prefix_columns(jp, "JP::")], axis=1
    ).dropna(axis=0, how="any")
    if len(combined) < int(min_obs):
        return result

    us_pref = [col for col in combined.columns if str(col).startswith("US::")]
    jp_pref = [col for col in combined.columns if str(col).startswith("JP::")]
    if not us_pref or not jp_pref:
        return result

    corr = combined.corr().fillna(0.0)
    cross = corr.loc[jp_pref, us_pref]
    cross.index = [str(col).replace("JP::", "", 1) for col in cross.index]
    cross.columns = [str(col).replace("US::", "", 1) for col in cross.columns]
    result.loc[cross.index, cross.columns] = cross
    return result.clip(-0.999, 0.999)


def build_prior_corr_matrix(
    prior_exposure: pd.DataFrame,
    us_columns: Sequence[str],
    jp_columns: Sequence[str],
) -> pd.DataFrame:
    us_cols = [str(col) for col in us_columns]
    jp_cols = [str(col) for col in jp_columns]
    labels = ["US::{}".format(col) for col in us_cols] + [
        "JP::{}".format(col) for col in jp_cols
    ]
    if not labels:
        return pd.DataFrame()

    corr = np.eye(len(labels), dtype=float)
    exposure = (
        prior_exposure.reindex(index=jp_cols, columns=us_cols).fillna(0.0).to_numpy(dtype=float)
    )
    n_us = len(us_cols)
    n_jp = len(jp_cols)
    if n_us > 0 and n_jp > 0:
        corr[n_us:, :n_us] = exposure
        corr[:n_us, n_us:] = exposure.T

    corr = (corr + corr.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, 1e-8, None)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    corr_psd = _normalize_corr_matrix(corr_psd)
    return pd.DataFrame(corr_psd, index=labels, columns=labels)


def compute_regularized_corr(
    sample_corr: pd.DataFrame,
    prior_corr: pd.DataFrame,
    lambda_reg: float,
) -> pd.DataFrame:
    labels = sample_corr.index.union(sample_corr.columns).union(prior_corr.index).union(
        prior_corr.columns
    )
    if len(labels) == 0:
        return pd.DataFrame()

    sample = sample_corr.reindex(index=labels, columns=labels).fillna(0.0).astype(float)
    prior = prior_corr.reindex(index=labels, columns=labels).fillna(0.0).astype(float)
    sample_values = _normalize_corr_matrix(sample.to_numpy(dtype=float))
    prior_values = _normalize_corr_matrix(prior.to_numpy(dtype=float))

    lam = float(np.clip(lambda_reg, 0.0, 1.0))
    reg = (1.0 - lam) * sample_values + lam * prior_values
    reg = _normalize_corr_matrix(reg)
    return pd.DataFrame(reg, index=labels, columns=labels)


def extract_block_eigenvectors(
    regularized_corr: pd.DataFrame,
    us_columns: Sequence[str],
    jp_columns: Sequence[str],
    n_components: int,
) -> Dict[str, Any]:
    us_cols = [str(col) for col in us_columns]
    jp_cols = [str(col) for col in jp_columns]
    k = max(int(n_components), 1)

    if regularized_corr.empty or not us_cols or not jp_cols:
        return {
            "us_loadings": pd.DataFrame(),
            "jp_loadings": pd.DataFrame(),
            "singular_values": pd.Series(dtype=float),
        }

    cross = (
        regularized_corr.reindex(index=jp_cols, columns=us_cols).fillna(0.0).astype(float)
    )
    cross_values = cross.to_numpy(dtype=float)
    if cross_values.size == 0:
        return {
            "us_loadings": pd.DataFrame(),
            "jp_loadings": pd.DataFrame(),
            "singular_values": pd.Series(dtype=float),
        }

    u, s, vt = np.linalg.svd(cross_values, full_matrices=False)
    k = min(k, len(s))
    component_names = ["pc_{}".format(i + 1) for i in range(k)]

    us_loadings = pd.DataFrame(vt.T[:, :k], index=us_cols, columns=component_names)
    jp_loadings = pd.DataFrame(u[:, :k], index=jp_cols, columns=component_names)
    singular_values = pd.Series(s[:k], index=component_names, dtype=float)
    return {
        "us_loadings": us_loadings,
        "jp_loadings": jp_loadings,
        "singular_values": singular_values,
    }


def compute_pca_sub_signal(
    us_signal: pd.Series,
    pca_components: Dict[str, Any],
    baseline_mode: str = "pca_sub",
    jp_prev_returns: Optional[pd.Series] = None,
    jp_momentum: Optional[pd.Series] = None,
) -> pd.Series:
    mode = str(baseline_mode).lower()
    if mode not in BASELINE_MODES:
        raise ValueError("baseline_mode must be one of: {}".format(",".join(sorted(BASELINE_MODES))))

    us_loadings = pca_components.get("us_loadings", pd.DataFrame())
    jp_loadings = pca_components.get("jp_loadings", pd.DataFrame())
    singular_values = pca_components.get("singular_values", pd.Series(dtype=float))
    if us_loadings.empty or jp_loadings.empty:
        return pd.Series(dtype=float)

    x = pd.to_numeric(us_signal.reindex(us_loadings.index), errors="coerce")
    if x.isna().any():
        return pd.Series(dtype=float)

    latent = us_loadings.to_numpy(dtype=float).T @ x.to_numpy(dtype=float)
    singular = pd.to_numeric(
        pd.Series(singular_values).reindex(us_loadings.columns), errors="coerce"
    ).fillna(0.0)
    plain = jp_loadings.to_numpy(dtype=float) @ (latent * singular.to_numpy(dtype=float))
    plain_signal = pd.Series(plain, index=jp_loadings.index, dtype=float)

    prev = (
        pd.to_numeric(jp_prev_returns.reindex(plain_signal.index), errors="coerce").fillna(0.0)
        if jp_prev_returns is not None
        else pd.Series(0.0, index=plain_signal.index, dtype=float)
    )
    mom = (
        pd.to_numeric(jp_momentum.reindex(plain_signal.index), errors="coerce").fillna(0.0)
        if jp_momentum is not None
        else pd.Series(0.0, index=plain_signal.index, dtype=float)
    )

    if mode == "plainpca":
        signal = plain_signal
    elif mode == "pca_sub":
        signal = plain_signal - prev
    elif mode == "mompca":
        signal = plain_signal + plain_signal.abs() * np.sign(mom)
    else:  # double
        signal = (2.0 * plain_signal) - prev + (0.5 * mom)

    signal = signal.replace([np.inf, -np.inf], np.nan).dropna()
    return signal


def build_quantile_long_short_weights(
    signal: pd.Series,
    quantile_q: float = 0.3,
) -> Tuple[pd.Series, Dict[str, Any]]:
    q = float(quantile_q)
    if not (0.0 < q < 0.5):
        raise ValueError("quantile_q must be in (0, 0.5)")

    clean_signal = pd.to_numeric(signal, errors="coerce").dropna()
    weights = pd.Series(0.0, index=clean_signal.index, dtype=float)
    meta: Dict[str, Any] = {
        "reason": DIAG_QUANTILE_UNAVAILABLE,
        "lower_threshold": np.nan,
        "upper_threshold": np.nan,
        "long_count": 0,
        "short_count": 0,
    }
    if clean_signal.empty:
        return weights, meta

    if int(clean_signal.nunique(dropna=True)) < 2:
        return weights, meta

    lower = float(clean_signal.quantile(q))
    upper = float(clean_signal.quantile(1.0 - q))
    meta["lower_threshold"] = lower
    meta["upper_threshold"] = upper
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return weights, meta

    long_symbols = list(clean_signal[clean_signal > upper].index)
    short_symbols = list(clean_signal[clean_signal < lower].index)
    if not long_symbols or not short_symbols:
        n_side = int(np.floor(len(clean_signal) * q))
        if n_side < 1 or 2 * n_side > len(clean_signal):
            return weights, meta
        ranked = clean_signal.sort_values()
        short_symbols = list(ranked.index[:n_side])
        long_symbols = list(ranked.index[-n_side:])

    if set(long_symbols).intersection(short_symbols):
        return weights, meta
    if len(long_symbols) == 0 or len(short_symbols) == 0:
        return weights, meta

    weights.loc[long_symbols] = 1.0 / float(len(long_symbols))
    weights.loc[short_symbols] = -1.0 / float(len(short_symbols))
    meta["reason"] = DIAG_TRADABLE
    meta["long_count"] = int(len(long_symbols))
    meta["short_count"] = int(len(short_symbols))
    return weights, meta


def evaluate_long_short_returns(
    weights: pd.DataFrame,
    target_returns: pd.DataFrame,
    one_way_cost_bps: float = 10.0,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "gross_return",
                "trading_cost",
                "net_return",
                "turnover",
                "equity_curve",
                "gross_exposure",
                "long_exposure",
                "short_exposure",
                "long_count",
                "short_count",
                "is_empty_position",
            ]
        )

    weight_df = weights.sort_index().fillna(0.0).astype(float)
    return_df = (
        target_returns.reindex(index=weight_df.index, columns=weight_df.columns)
        .fillna(0.0)
        .astype(float)
    )

    gross_return = (weight_df * return_df).sum(axis=1)
    turnover = weight_df.diff().abs().sum(axis=1)
    if len(turnover) > 0:
        turnover.iloc[0] = weight_df.iloc[0].abs().sum()

    cost_rate = float(one_way_cost_bps) / 10000.0
    trading_cost = turnover * cost_rate
    net_return = gross_return - trading_cost
    equity_curve = (1.0 + net_return).cumprod()

    long_exposure = weight_df.clip(lower=0.0).sum(axis=1)
    short_exposure = weight_df.clip(upper=0.0).sum(axis=1)
    gross_exposure = weight_df.abs().sum(axis=1)
    long_count = (weight_df > _EPS).sum(axis=1).astype(int)
    short_count = (weight_df < -_EPS).sum(axis=1).astype(int)
    is_empty = gross_exposure <= _EPS

    daily = pd.DataFrame(
        {
            "date": weight_df.index,
            "gross_return": gross_return.to_numpy(dtype=float),
            "trading_cost": trading_cost.to_numpy(dtype=float),
            "net_return": net_return.to_numpy(dtype=float),
            "turnover": turnover.to_numpy(dtype=float),
            "equity_curve": equity_curve.to_numpy(dtype=float),
            "gross_exposure": gross_exposure.to_numpy(dtype=float),
            "long_exposure": long_exposure.to_numpy(dtype=float),
            "short_exposure": short_exposure.to_numpy(dtype=float),
            "long_count": long_count.to_numpy(dtype=int),
            "short_count": short_count.to_numpy(dtype=int),
            "is_empty_position": is_empty.to_numpy(dtype=bool),
        }
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    return daily.sort_values("date").reset_index(drop=True)


def summarize_leadlag_backtest(
    daily_returns: pd.DataFrame,
    config: LeadLagConfig,
    avg_regularization_delta: float = 0.0,
    nonzero_regularization_days: int = 0,
) -> pd.DataFrame:
    if daily_returns.empty:
        return _empty_backtest_result(config)["summary"]

    daily = daily_returns.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if daily.empty:
        return _empty_backtest_result(config)["summary"]

    ann_factor = 252.0
    total_days = int(len(daily))
    cumulative_return = float((1.0 + daily["net_return"]).prod() - 1.0)
    years = max(total_days / ann_factor, 1.0 / ann_factor)
    annual_return = float((1.0 + cumulative_return) ** (1.0 / years) - 1.0)
    annual_vol = float(daily["net_return"].std(ddof=1) * np.sqrt(ann_factor))
    sharpe_like = annual_return / annual_vol if annual_vol > 0 else np.nan

    equity = daily["equity_curve"]
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    trading_mask = ~daily["is_empty_position"].astype(bool)
    trading_days = int(trading_mask.sum())
    empty_position_days = int((~trading_mask).sum())
    if trading_days > 0:
        win_rate = float((daily.loc[trading_mask, "net_return"] > 0).mean())
    else:
        win_rate = np.nan

    summary_row = {
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe_like": sharpe_like,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trading_days": trading_days,
        "empty_position_days": empty_position_days,
        "total_days": total_days,
        "lambda_reg": float(config.lambda_reg),
        "baseline_mode": str(config.baseline_mode),
        "quantile_q": float(config.quantile_q),
        "lookback": int(config.lookback),
        "n_components": int(config.n_components),
        "avg_regularization_delta": float(avg_regularization_delta),
        "nonzero_regularization_days": int(nonzero_regularization_days),
    }
    return pd.DataFrame([summary_row])


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def run_leadlag_pca_backtest(
    aligned_df: pd.DataFrame,
    config: Optional[LeadLagConfig] = None,
    join_key: str = "path_group",
) -> Dict[str, Any]:
    cfg = config if config is not None else LeadLagConfig()
    if aligned_df is None or aligned_df.empty:
        return _empty_backtest_result(cfg)

    df = aligned_df.copy()
    date_col = _resolve_column(df, ["trade_date_jp", "date_jp", "trade_date", "date"])
    us_return_col = _resolve_column(df, ["us_close_to_close_return", "us_return", "signal_return"])
    jp_return_col = _resolve_column(df, ["jp_open_to_close_return", "jp_return", "target_return"])
    us_symbol_col = _resolve_column(df, [join_key, "symbol_us", "code_us", "symbol", "code"])
    jp_symbol_col = _resolve_column(df, ["symbol_jp", "code_jp", "symbol", "code"])
    if (
        date_col is None
        or us_return_col is None
        or jp_return_col is None
        or us_symbol_col is None
        or jp_symbol_col is None
    ):
        return _empty_backtest_result(cfg)

    df["__date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=["__date"])
    if df.empty:
        return _empty_backtest_result(cfg)

    start_ts = _to_ts(cfg.start_date)
    end_ts = _to_ts(cfg.end_date)
    if start_ts is not None:
        df = df[df["__date"] >= start_ts]
    if end_ts is not None:
        df = df[df["__date"] <= end_ts]
    if df.empty:
        return _empty_backtest_result(cfg)

    df[us_symbol_col] = df[us_symbol_col].astype(str)
    df[jp_symbol_col] = df[jp_symbol_col].astype(str)
    df[us_return_col] = pd.to_numeric(df[us_return_col], errors="coerce")
    df[jp_return_col] = pd.to_numeric(df[jp_return_col], errors="coerce")

    us_matrix = (
        df.pivot_table(
            index="__date",
            columns=us_symbol_col,
            values=us_return_col,
            aggfunc="mean",
        )
        .sort_index()
        .astype(float)
    )
    jp_matrix = (
        df.pivot_table(
            index="__date",
            columns=jp_symbol_col,
            values=jp_return_col,
            aggfunc="mean",
        )
        .sort_index()
        .astype(float)
    )
    common_dates = us_matrix.index.intersection(jp_matrix.index).sort_values()
    if len(common_dates) == 0:
        return _empty_backtest_result(cfg)

    us_matrix = us_matrix.reindex(common_dates)
    jp_matrix = jp_matrix.reindex(common_dates)

    prior_start_ts = _to_ts(cfg.prior_start)
    prior_end_ts = _to_ts(cfg.prior_end)
    prior_mask = pd.Series(True, index=common_dates)
    if prior_start_ts is not None:
        prior_mask &= common_dates >= prior_start_ts
    if prior_end_ts is not None:
        prior_mask &= common_dates <= prior_end_ts
    prior_dates = common_dates[prior_mask.to_numpy()]
    if len(prior_dates) < max(int(cfg.lookback), 20):
        fallback_len = min(len(common_dates), max(int(cfg.lookback), int(len(common_dates) * 0.3)))
        prior_dates = common_dates[:fallback_len]

    prior_exposure = build_prior_exposure_matrix(
        prior_us_returns=us_matrix.loc[prior_dates],
        prior_jp_returns=jp_matrix.loc[prior_dates],
        min_obs=max(10, int(cfg.lookback // 2)),
    )
    prior_corr = build_prior_corr_matrix(
        prior_exposure=prior_exposure,
        us_columns=us_matrix.columns.tolist(),
        jp_columns=jp_matrix.columns.tolist(),
    )

    weight_matrix = pd.DataFrame(0.0, index=common_dates, columns=jp_matrix.columns, dtype=float)
    signal_rows = []
    weight_rows = []
    diag_rows = []
    reg_deltas = []

    mode = str(cfg.baseline_mode).lower()
    min_us_required = max(int(cfg.min_us_symbols), int(cfg.n_components))
    min_jp_required = max(int(cfg.min_jp_symbols), 2)

    for i, trade_date in enumerate(common_dates):
        current_us = us_matrix.iloc[i]
        current_jp = jp_matrix.iloc[i]
        diag = {
            "date": pd.Timestamp(trade_date),
            "status": "no_trade",
            "reason": DIAG_INSUFFICIENT_WINDOW,
            "available_us": int(current_us.notna().sum()),
            "available_jp": int(current_jp.notna().sum()),
            "usable_us": 0,
            "usable_jp": 0,
            "window_obs": 0,
            "long_count": 0,
            "short_count": 0,
        }

        if i < int(cfg.lookback):
            diag_rows.append(diag)
            continue

        current_us_symbols = list(current_us[current_us.notna()].index)
        current_jp_symbols = list(current_jp[current_jp.notna()].index)
        if len(current_us_symbols) < min_us_required or len(current_jp_symbols) < min_jp_required:
            diag["reason"] = DIAG_INSUFFICIENT_SYMBOLS
            diag_rows.append(diag)
            continue

        window_us = us_matrix.iloc[i - int(cfg.lookback) : i][current_us_symbols]
        window_jp = jp_matrix.iloc[i - int(cfg.lookback) : i][current_jp_symbols]
        diag["window_obs"] = int(len(window_us))

        if window_us.isna().any().any() or window_jp.isna().any().any():
            diag["reason"] = DIAG_MISSING_DATA
            diag_rows.append(diag)
            continue

        us_std = window_us.std(axis=0, ddof=0)
        jp_std = window_jp.std(axis=0, ddof=0)
        usable_us_symbols = [col for col in window_us.columns if float(us_std.get(col, 0.0)) > _EPS]
        usable_jp_symbols = [col for col in window_jp.columns if float(jp_std.get(col, 0.0)) > _EPS]
        diag["usable_us"] = int(len(usable_us_symbols))
        diag["usable_jp"] = int(len(usable_jp_symbols))
        if len(usable_us_symbols) < min_us_required or len(usable_jp_symbols) < min_jp_required:
            diag["reason"] = DIAG_INSUFFICIENT_SYMBOLS
            diag_rows.append(diag)
            continue

        xw = window_us[usable_us_symbols]
        yw = window_jp[usable_jp_symbols]
        us_pref = ["US::{}".format(sym) for sym in xw.columns]
        jp_pref = ["JP::{}".format(sym) for sym in yw.columns]
        sample_input = pd.concat(
            [xw.set_axis(us_pref, axis=1), yw.set_axis(jp_pref, axis=1)], axis=1
        )
        sample_corr = sample_input.corr().fillna(0.0)

        prior_subset = prior_corr.reindex(index=sample_corr.index, columns=sample_corr.columns).fillna(0.0)
        regularized_corr = compute_regularized_corr(
            sample_corr=sample_corr,
            prior_corr=prior_subset,
            lambda_reg=float(cfg.lambda_reg),
        )
        reg_delta = float(
            np.linalg.norm(
                regularized_corr.to_numpy(dtype=float) - sample_corr.to_numpy(dtype=float),
                ord="fro",
            )
        )
        reg_deltas.append(reg_delta)

        components = extract_block_eigenvectors(
            regularized_corr=regularized_corr,
            us_columns=us_pref,
            jp_columns=jp_pref,
            n_components=int(cfg.n_components),
        )
        current_us_pref = pd.Series(current_us.loc[usable_us_symbols].to_numpy(dtype=float), index=us_pref)
        prev_jp = (
            pd.Series(jp_matrix.iloc[i - 1].loc[usable_jp_symbols].to_numpy(dtype=float), index=jp_pref)
            if i > 0
            else pd.Series(0.0, index=jp_pref)
        )
        mom_start = max(0, i - int(cfg.momentum_lookback))
        mom_window = jp_matrix.iloc[mom_start:i].reindex(columns=usable_jp_symbols)
        if mom_window.empty:
            jp_mom = pd.Series(0.0, index=jp_pref)
        else:
            jp_mom = pd.Series(mom_window.mean(axis=0).to_numpy(dtype=float), index=jp_pref)

        signal_pref = compute_pca_sub_signal(
            us_signal=current_us_pref,
            pca_components=components,
            baseline_mode=mode,
            jp_prev_returns=prev_jp,
            jp_momentum=jp_mom,
        )
        if signal_pref.empty:
            diag["reason"] = DIAG_INSUFFICIENT_SYMBOLS
            diag_rows.append(diag)
            continue

        signal = pd.Series(
            signal_pref.to_numpy(dtype=float),
            index=[str(sym).replace("JP::", "", 1) for sym in signal_pref.index],
            dtype=float,
        )
        weights_today, weight_meta = build_quantile_long_short_weights(
            signal=signal,
            quantile_q=float(cfg.quantile_q),
        )
        if weight_meta.get("reason") != DIAG_TRADABLE:
            diag["reason"] = DIAG_QUANTILE_UNAVAILABLE
            diag_rows.append(diag)
            continue

        aligned_weights = pd.Series(0.0, index=jp_matrix.columns, dtype=float)
        aligned_weights.loc[weights_today.index] = weights_today.to_numpy(dtype=float)
        weight_matrix.loc[trade_date, aligned_weights.index] = aligned_weights.to_numpy(dtype=float)

        for symbol, value in signal.items():
            signal_rows.append(
                {
                    "date": pd.Timestamp(trade_date),
                    "symbol": str(symbol),
                    "signal": float(value),
                    "baseline_mode": mode,
                }
            )
        nonzero = aligned_weights[np.abs(aligned_weights) > _EPS]
        for symbol, value in nonzero.items():
            weight_rows.append(
                {
                    "date": pd.Timestamp(trade_date),
                    "symbol": str(symbol),
                    "weight": float(value),
                    "side": "long" if value > 0 else "short",
                }
            )

        diag["status"] = "trade"
        diag["reason"] = DIAG_TRADABLE
        diag["long_count"] = int(weight_meta.get("long_count", 0))
        diag["short_count"] = int(weight_meta.get("short_count", 0))
        diag_rows.append(diag)

    signals_df = (
        pd.DataFrame(signal_rows).sort_values(["date", "symbol"]).reset_index(drop=True)
        if signal_rows
        else pd.DataFrame(columns=["date", "symbol", "signal", "baseline_mode"])
    )
    weights_df = (
        pd.DataFrame(weight_rows).sort_values(["date", "symbol"]).reset_index(drop=True)
        if weight_rows
        else pd.DataFrame(columns=["date", "symbol", "weight", "side"])
    )
    daily_returns = evaluate_long_short_returns(
        weights=weight_matrix,
        target_returns=jp_matrix,
        one_way_cost_bps=float(cfg.one_way_cost_bps),
    )
    diag_df = pd.DataFrame(diag_rows).sort_values("date").reset_index(drop=True)
    if not daily_returns.empty and not diag_df.empty:
        daily_returns = pd.merge(
            daily_returns,
            diag_df[["date", "reason", "status"]],
            on="date",
            how="left",
        )

    if reg_deltas:
        avg_reg_delta = float(np.mean(reg_deltas))
        nonzero_reg_days = int(np.sum(np.array(reg_deltas) > _EPS))
    else:
        avg_reg_delta = 0.0
        nonzero_reg_days = 0

    summary_df = summarize_leadlag_backtest(
        daily_returns=daily_returns,
        config=cfg,
        avg_regularization_delta=avg_reg_delta,
        nonzero_regularization_days=nonzero_reg_days,
    )

    counts = (
        diag_df["reason"].value_counts(dropna=False).to_dict() if not diag_df.empty else {}
    )
    diagnostics = {
        "counts": counts,
        "daily": diag_df,
        "regularization": {
            "avg_delta": avg_reg_delta,
            "nonzero_days": nonzero_reg_days,
        },
        "config": asdict(cfg),
    }
    return {
        "signals": signals_df,
        "weights": weights_df,
        "daily_returns": daily_returns,
        "summary": summary_df,
        "diagnostics": diagnostics,
    }
