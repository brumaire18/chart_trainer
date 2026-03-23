import unittest

import pandas as pd

from app.data_loader import (
    align_us_signal_and_jp_target,
    compute_jp_open_to_close_returns,
    compute_us_close_to_close_returns,
)
from app.leadlag_pca import (
    DIAG_INSUFFICIENT_SYMBOLS,
    LeadLagConfig,
    build_prior_corr_matrix,
    build_prior_exposure_matrix,
    build_quantile_long_short_weights,
    compute_regularized_corr,
    extract_block_eigenvectors,
    run_leadlag_pca_backtest,
)


class LeadLagCalendarAlignmentTests(unittest.TestCase):
    def _build_us_price_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"date": "2024-01-04", "symbol": "US_A", "market": "US", "close": 100.0, "path_group": "tech"},
                {"date": "2024-01-05", "symbol": "US_A", "market": "US", "close": 102.0, "path_group": "tech"},
                {"date": "2024-01-08", "symbol": "US_A", "market": "US", "close": 101.0, "path_group": "tech"},
                {"date": "2024-01-09", "symbol": "US_A", "market": "US", "close": 103.0, "path_group": "tech"},
                {"date": "2024-01-04", "symbol": "US_B", "market": "US", "close": 200.0, "path_group": "financials"},
                {"date": "2024-01-05", "symbol": "US_B", "market": "US", "close": 198.0, "path_group": "financials"},
                {"date": "2024-01-08", "symbol": "US_B", "market": "US", "close": 201.0, "path_group": "financials"},
                {"date": "2024-01-09", "symbol": "US_B", "market": "US", "close": 205.0, "path_group": "financials"},
            ]
        )

    def _build_jp_price_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"date": "2024-01-05", "symbol": "1301", "market": "JP", "open": 100.0, "close": 101.0, "path_group": "tech"},
                {"date": "2024-01-09", "symbol": "1301", "market": "JP", "open": 101.0, "close": 103.0, "path_group": "tech"},
                {"date": "2024-01-10", "symbol": "1301", "market": "JP", "open": 103.0, "close": 102.0, "path_group": "tech"},
                {"date": "2024-01-05", "symbol": "8306", "market": "JP", "open": 200.0, "close": 199.0, "path_group": "financials"},
                {"date": "2024-01-09", "symbol": "8306", "market": "JP", "open": 199.0, "close": 204.0, "path_group": "financials"},
                {"date": "2024-01-10", "symbol": "8306", "market": "JP", "open": 204.0, "close": 205.0, "path_group": "financials"},
            ]
        )

    def test_aligns_us_signal_dates_to_next_jp_business_day_and_exposes_mapping_df(self):
        us_signal_df = compute_us_close_to_close_returns(self._build_us_price_df())
        jp_target_df = compute_jp_open_to_close_returns(self._build_jp_price_df())

        aligned_df, mapping_df = align_us_signal_and_jp_target(
            us_signal_df=us_signal_df,
            jp_target_df=jp_target_df,
            join_key="path_group",
        )

        self.assertFalse(aligned_df.empty)
        self.assertFalse(mapping_df.empty)

        mapping = mapping_df.set_index("signal_date_us")
        self.assertEqual(pd.Timestamp("2024-01-09"), mapping.loc[pd.Timestamp("2024-01-05"), "trade_date_jp"])
        self.assertEqual(pd.Timestamp("2024-01-09"), mapping.loc[pd.Timestamp("2024-01-08"), "trade_date_jp"])
        self.assertEqual("Friday", mapping.loc[pd.Timestamp("2024-01-05"), "signal_weekday_us"])
        self.assertEqual("Tuesday", mapping.loc[pd.Timestamp("2024-01-05"), "trade_weekday_jp"])
        self.assertEqual(4, int(mapping.loc[pd.Timestamp("2024-01-05"), "calendar_lag_days"]))
        self.assertTrue(bool(mapping.loc[pd.Timestamp("2024-01-05"), "is_signal_friday"]))

        friday_rows = aligned_df[aligned_df["signal_date_us"] == pd.Timestamp("2024-01-05")]
        self.assertTrue((friday_rows["trade_date_jp"] == pd.Timestamp("2024-01-09")).all())
        self.assertEqual({"1301", "8306"}, set(friday_rows["symbol_jp"]))


class LeadLagPcaHelperTests(unittest.TestCase):
    def test_build_quantile_long_short_weights_balances_long_and_short_sides(self):
        signal = pd.Series(
            {
                "1301": -0.9,
                "1302": -0.4,
                "1303": -0.1,
                "1304": 0.2,
                "1305": 0.6,
                "1306": 1.2,
            }
        )

        weights, meta = build_quantile_long_short_weights(signal, quantile_q=0.34)

        self.assertEqual("tradable", meta["reason"])
        self.assertEqual(2, int(meta["long_count"]))
        self.assertEqual(2, int(meta["short_count"]))
        self.assertAlmostEqual(1.0, float(weights[weights > 0].sum()))
        self.assertAlmostEqual(-1.0, float(weights[weights < 0].sum()))
        self.assertEqual({"1305", "1306"}, set(weights[weights > 0].index))
        self.assertEqual({"1301", "1302"}, set(weights[weights < 0].index))

    def test_n_components_three_returns_three_loading_columns(self):
        prior_us = pd.DataFrame(
            {
                "G1": [0.01, 0.02, 0.015, 0.03, 0.01],
                "G2": [0.015, 0.005, 0.02, 0.01, 0.025],
                "G3": [0.005, 0.01, 0.0, 0.015, 0.02],
            },
            index=pd.bdate_range("2024-01-02", periods=5),
        )
        prior_jp = pd.DataFrame(
            {
                "1301": [0.012, 0.021, 0.016, 0.028, 0.013],
                "1302": [0.011, 0.018, 0.014, 0.024, 0.012],
                "1303": [0.016, 0.008, 0.019, 0.011, 0.026],
                "1304": [0.014, 0.006, 0.017, 0.009, 0.021],
                "1305": [0.007, 0.011, 0.002, 0.016, 0.019],
                "1306": [0.005, 0.009, 0.001, 0.013, 0.017],
            },
            index=prior_us.index,
        )

        prior_exposure = build_prior_exposure_matrix(prior_us, prior_jp, min_obs=3)
        prior_corr = build_prior_corr_matrix(prior_exposure, prior_us.columns, prior_jp.columns)
        regularized_corr = compute_regularized_corr(prior_corr, prior_corr, lambda_reg=0.9)
        components = extract_block_eigenvectors(
            regularized_corr=regularized_corr,
            us_columns=["US::G1", "US::G2", "US::G3"],
            jp_columns=["JP::1301", "JP::1302", "JP::1303", "JP::1304", "JP::1305", "JP::1306"],
            n_components=3,
        )

        self.assertEqual((3, 3), components["us_loadings"].shape)
        self.assertEqual((6, 3), components["jp_loadings"].shape)
        self.assertEqual(3, len(components["singular_values"]))


class LeadLagPcaBacktestTests(unittest.TestCase):
    def _build_aligned_df(self) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-02", periods=9)
        us_returns = {
            "tech": [0.010, 0.018, 0.012, 0.026, -0.006, 0.031, 0.009, 0.022, 0.028],
            "financials": [0.020, 0.006, 0.017, -0.008, 0.024, 0.015, -0.010, 0.027, 0.013],
            "industrials": [-0.005, 0.012, 0.008, 0.019, 0.006, -0.011, 0.023, 0.005, 0.018],
        }
        rows = []
        jp_symbols = [
            ("1301", "tech"),
            ("1302", "tech"),
            ("8306", "financials"),
            ("8316", "financials"),
            ("7011", "industrials"),
            ("7201", "industrials"),
        ]

        for idx, trade_date in enumerate(dates):
            regime_flip = -1.0 if idx >= 4 else 1.0
            g1 = us_returns["tech"][idx]
            g2 = us_returns["financials"][idx]
            g3 = us_returns["industrials"][idx]
            jp_returns = {
                "1301": 0.95 * g1 + 0.10 * g2 + (idx * 0.0002),
                "1302": 0.80 * g1 - 0.05 * g3 - (idx * 0.0001),
                "8306": regime_flip * (0.85 * g2) + 0.05 * g1 + 0.0003,
                "8316": regime_flip * (0.65 * g2) - 0.08 * g3 - 0.0002,
                "7011": 0.78 * g3 + 0.08 * g1 + (idx * 0.00015),
                "7201": 0.62 * g3 - 0.04 * g2 - (idx * 0.00005),
            }
            signal_date = trade_date - pd.offsets.BDay(1)
            for path_group, us_ret in us_returns.items():
                for symbol, jp_group in jp_symbols:
                    if jp_group != path_group:
                        continue
                    rows.append(
                        {
                            "signal_date_us": pd.Timestamp(signal_date),
                            "trade_date_jp": pd.Timestamp(trade_date),
                            "path_group": path_group,
                            "symbol_jp": symbol,
                            "us_close_to_close_return": us_ret[idx],
                            "jp_open_to_close_return": jp_returns[symbol],
                        }
                    )
        return pd.DataFrame(rows)

    def _build_config(self, **overrides) -> LeadLagConfig:
        params = {
            "lookback": 3,
            "lambda_reg": 0.9,
            "n_components": 3,
            "quantile_q": 0.34,
            "baseline_mode": "pca_sub",
            "one_way_cost_bps": 0.0,
            "min_jp_symbols": 6,
            "min_us_symbols": 3,
            "momentum_lookback": 2,
        }
        params.update(overrides)
        return LeadLagConfig(**params)

    def test_backtest_prevents_lookahead_for_same_day_jp_returns(self):
        aligned_df = self._build_aligned_df()
        config = self._build_config(lambda_reg=0.0)

        base_result = run_leadlag_pca_backtest(aligned_df, config=config, join_key="path_group")

        target_date = pd.Timestamp("2024-01-12")
        modified_df = aligned_df.copy()
        mask = modified_df["trade_date_jp"] == target_date
        modified_df.loc[mask & (modified_df["symbol_jp"] == "1301"), "jp_open_to_close_return"] = 0.30
        modified_df.loc[mask & (modified_df["symbol_jp"] == "1302"), "jp_open_to_close_return"] = 0.25
        modified_df.loc[mask & (modified_df["symbol_jp"] == "8306"), "jp_open_to_close_return"] = -0.20
        modified_df.loc[mask & (modified_df["symbol_jp"] == "8316"), "jp_open_to_close_return"] = -0.18
        modified_df.loc[mask & (modified_df["symbol_jp"] == "7011"), "jp_open_to_close_return"] = 0.01
        modified_df.loc[mask & (modified_df["symbol_jp"] == "7201"), "jp_open_to_close_return"] = -0.02

        modified_result = run_leadlag_pca_backtest(modified_df, config=config, join_key="path_group")

        base_weights = base_result["weights"]
        changed_weights = modified_result["weights"]
        pd.testing.assert_frame_equal(
            base_weights[base_weights["date"] == target_date].reset_index(drop=True),
            changed_weights[changed_weights["date"] == target_date].reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            base_result["signals"][base_result["signals"]["date"] == target_date].reset_index(drop=True),
            modified_result["signals"][modified_result["signals"]["date"] == target_date].reset_index(drop=True),
        )

        base_daily = base_result["daily_returns"].set_index("date")
        changed_daily = modified_result["daily_returns"].set_index("date")
        self.assertNotEqual(
            float(base_daily.loc[target_date, "gross_return"]),
            float(changed_daily.loc[target_date, "gross_return"]),
        )

    def test_lambda_reg_zero_and_nine_produce_different_regularization_diagnostics(self):
        aligned_df = self._build_aligned_df()

        result_no_reg = run_leadlag_pca_backtest(
            aligned_df,
            config=self._build_config(lambda_reg=0.0),
            join_key="path_group",
        )
        result_with_reg = run_leadlag_pca_backtest(
            aligned_df,
            config=self._build_config(lambda_reg=0.9),
            join_key="path_group",
        )

        summary_no_reg = result_no_reg["summary"].iloc[0]
        summary_with_reg = result_with_reg["summary"].iloc[0]

        self.assertGreater(float(summary_no_reg["avg_regularization_delta"]), 0.0)
        self.assertGreater(float(summary_with_reg["avg_regularization_delta"]), float(summary_no_reg["avg_regularization_delta"]))
        self.assertGreaterEqual(
            int(summary_with_reg["nonzero_regularization_days"]),
            int(summary_no_reg["nonzero_regularization_days"]),
        )

    def test_skips_day_when_available_jp_symbols_are_below_threshold(self):
        aligned_df = self._build_aligned_df()
        skip_date = pd.Timestamp("2024-01-12")
        reduced_df = aligned_df[~((aligned_df["trade_date_jp"] == skip_date) & (aligned_df["symbol_jp"] == "7201"))].copy()

        result = run_leadlag_pca_backtest(reduced_df, config=self._build_config(), join_key="path_group")

        diagnostics_daily = result["diagnostics"]["daily"].set_index("date")
        daily_returns = result["daily_returns"].set_index("date")
        self.assertEqual(DIAG_INSUFFICIENT_SYMBOLS, diagnostics_daily.loc[skip_date, "reason"])
        self.assertTrue(bool(daily_returns.loc[skip_date, "is_empty_position"]))


if __name__ == "__main__":
    unittest.main()
