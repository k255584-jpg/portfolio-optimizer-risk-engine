import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import data_loader
import main
import mandate_parser
import risk_engine
from optimizer import optimize_portfolio
from portfolio_math import (
    calculate_annual_returns,
    calculate_covariance_matrix,
    calculate_log_returns,
    calculate_simple_returns,
    negative_sharpe,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio
)
from risk_engine import calculate_cvar, calculate_var, run_monte_carlo


class TestDataLoader(unittest.TestCase):

    def test_single_stock_extraction(self):
        dates = pd.date_range("2024-01-01", periods=2)
        data = pd.DataFrame(
            {"Adj Close": [100.0, 101.0]},
            index=dates
        )
        data.attrs["requested_tickers"] = ["AAPL"]

        result = data_loader.get_closing_prices(data)

        self.assertEqual(list(result.columns), ["AAPL"])

    def test_reversed_multiindex_extraction(self):
        dates = pd.date_range("2024-01-01", periods=2)
        columns = pd.MultiIndex.from_tuples(
            [
                ("AAPL", "Adj Close"),
                ("MSFT", "Adj Close")
            ]
        )
        data = pd.DataFrame(
            [[100.0, 200.0], [101.0, 201.0]],
            index=dates,
            columns=columns
        )

        result = data_loader.get_closing_prices(data)

        self.assertEqual(list(result.columns), ["AAPL", "MSFT"])

    def test_missing_adjusted_close_raises_error(self):
        data = pd.DataFrame(
            {
                "Close": [100.0, 101.0],
                "Open": [99.0, 100.0]
            }
        )

        with self.assertRaises(ValueError):
            main._require_adjusted_prices(data)

    def test_invalid_prices_raise_error(self):
        data = pd.DataFrame(
            {"Adj Close": [100.0, 0.0]}
        )

        with self.assertRaises(ValueError):
            data_loader.get_closing_prices(data)

    def test_data_quality_metadata_is_recorded(self):
        dates = pd.date_range("2024-01-01", periods=10)
        data = pd.DataFrame(
            {
                "AAPL": [100.0] * 10,
                "MSFT": [200.0, np.nan] + [200.0] * 8
            },
            index=dates
        )
        data.attrs["requested_tickers"] = ["AAPL", "MSFT"]

        result = data_loader.get_closing_prices(
            pd.concat({"Adj Close": data}, axis=1)
        )

        self.assertEqual(result.attrs["total_rows_before_cleaning"], 10)
        self.assertEqual(result.attrs["total_rows_after_cleaning"], 9)
        self.assertEqual(result.attrs["removed_rows"], 1)
        self.assertAlmostEqual(
            result.attrs["removed_rows_percentage"],
            0.10
        )
        self.assertEqual(
            result.attrs["entirely_missing_tickers"],
            []
        )
        self.assertEqual(
            list(result.columns),
            ["AAPL", "MSFT"]
        )

    def test_completely_missing_ticker_data_raises_clear_error(self):
        dates = pd.date_range("2024-01-01", periods=3)
        data = pd.concat(
            {
                "Adj Close": pd.DataFrame(
                    {
                        "AAPL": [100.0, 101.0, 102.0],
                        "MSFT": [np.nan, np.nan, np.nan]
                    },
                    index=dates
                )
            },
            axis=1
        )

        with self.assertRaisesRegex(ValueError, "MSFT"):
            data_loader.get_closing_prices(data)

    def test_missing_more_than_ten_percent_is_reported(self):
        dates = pd.date_range("2024-01-01", periods=300)
        rng = np.random.default_rng(11)
        aapl_prices = 100 * np.exp(
            np.cumsum(rng.normal(0.0002, 0.01, 300))
        )
        msft_prices = 200 * np.exp(
            np.cumsum(rng.normal(0.0002, 0.01, 300))
        )
        aapl_prices[:31] = np.nan
        prices = pd.DataFrame(
            {
                "AAPL": aapl_prices,
                "MSFT": msft_prices
            },
            index=dates
        )
        raw_data = pd.concat({"Adj Close": prices}, axis=1)
        mandate = {
            "tickers": ["AAPL", "MSFT"],
            "max_weight": 0.60,
            "goal": "min_vol"
        }

        with patch.object(
            main,
            "parse_investor_mandate",
            return_value=mandate
        ), patch.object(
            main,
            "download_stock_data",
            return_value=raw_data
        ):
            results = main.analyze_portfolio(
                "test mandate",
                start_date="2020-01-01",
                end_date="2025-01-01",
                num_simulations=10,
                num_days=5
            )

        self.assertEqual(results["removed_rows"], 31)
        self.assertGreater(
            results["data_quality"]["removed_rows_percentage"],
            0.10
        )

        with patch("builtins.print") as print_mock:
            main.print_analysis_report(results, currency_label="USD")

        output = "\n".join(
            str(call.args[0])
            for call in print_mock.call_args_list
            if call.args
        )
        self.assertIn(
            "More than 10% of historical rows were removed",
            output
        )

    def test_missing_requested_ticker_raises_error(self):
        prices = pd.DataFrame({"AAPL": [100.0, 101.0]})

        with self.assertRaises(ValueError):
            main._validate_requested_columns(
                prices,
                ["AAPL", "MSFT"]
            )

    def test_download_data_normalizes_tickers(self):
        fake_data = pd.DataFrame({"Adj Close": [100.0]})

        with patch.object(
            data_loader.yf,
            "download",
            return_value=fake_data
        ) as download:
            data_loader.download_stock_data(
                [" aapl ", "AAPL"],
                "2024-01-01",
                "2024-01-02"
            )

        download.assert_called_once()
        self.assertEqual(
            download.call_args.kwargs["tickers"],
            ["AAPL"]
        )


class TestPortfolioMath(unittest.TestCase):

    def setUp(self):
        self.prices = pd.DataFrame(
            {
                "AAPL": [100.0, 110.0, 99.0, 108.9],
                "MSFT": [200.0, 220.0, 198.0, 217.8]
            }
        )

    def test_simple_returns(self):
        result = calculate_simple_returns(self.prices)
        expected = pd.DataFrame(
            {
                "AAPL": [0.10, -0.10, 0.10],
                "MSFT": [0.10, -0.10, 0.10]
            }
        )
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            expected
        )

    def test_annual_returns_use_simple_returns(self):
        returns = pd.DataFrame(
            {"AAPL": [0.01, 0.02, 0.03]}
        )

        result = calculate_annual_returns(returns)

        self.assertAlmostEqual(
            result["AAPL"],
            returns["AAPL"].mean() * 252
        )

    def test_covariance_uses_simple_returns(self):
        returns = pd.DataFrame(
            {
                "AAPL": [0.01, 0.02, 0.03],
                "MSFT": [0.02, 0.01, 0.04]
            }
        )

        result = calculate_covariance_matrix(returns)
        expected = returns.cov() * 252

        pd.testing.assert_frame_equal(result, expected)

    def test_sharpe_matches_manual_calculation(self):
        expected_returns = pd.Series(
            {"AAPL": 0.10, "MSFT": 0.20}
        )
        covariance = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]],
            index=["AAPL", "MSFT"],
            columns=["AAPL", "MSFT"]
        )
        weights = np.array([0.5, 0.5])
        risk_free_rate = 0.04

        expected_return = np.dot(weights, expected_returns)
        volatility = np.sqrt(weights @ covariance.values @ weights)
        expected_sharpe = (
            expected_return - risk_free_rate
        ) / volatility

        self.assertAlmostEqual(
            sharpe_ratio(
                weights,
                expected_returns,
                covariance,
                risk_free_rate
            ),
            expected_sharpe
        )

    def test_log_returns_remain_available_for_monte_carlo(self):
        log_returns = calculate_log_returns(self.prices)
        self.assertEqual(log_returns.shape, (3, 2))

    def test_portfolio_math_pipeline(self):
        simple_returns = calculate_simple_returns(self.prices)
        annual_returns = calculate_annual_returns(simple_returns)
        covariance = calculate_covariance_matrix(simple_returns)
        weights = np.array([0.5, 0.5])

        expected_return = portfolio_return(weights, annual_returns)
        volatility = portfolio_volatility(weights, covariance)
        sharpe = sharpe_ratio(
            weights,
            annual_returns,
            covariance,
            0.04
        )

        self.assertTrue(np.isfinite(expected_return))
        self.assertGreaterEqual(volatility, 0)
        self.assertTrue(np.isfinite(sharpe))
        self.assertAlmostEqual(
            negative_sharpe(
                weights,
                annual_returns,
                covariance,
                0.04
            ),
            -sharpe
        )

    def test_zero_volatility_sharpe_has_user_friendly_error(self):
        expected_returns = pd.Series({"AAPL": 0.10})
        covariance = pd.DataFrame([[0.0]], index=["AAPL"], columns=["AAPL"])

        with self.assertRaisesRegex(
            ValueError,
            "Sharpe ratio is undefined.*volatility is zero"
        ):
            sharpe_ratio(
                np.array([1.0]),
                expected_returns,
                covariance,
                0.04
            )


class TestOptimizer(unittest.TestCase):

    def setUp(self):
        self.returns = pd.Series(
            {
                "AAPL": 0.10,
                "MSFT": 0.12,
                "NVDA": 0.18,
                "GOOG": 0.09
            }
        )
        self.covariance = pd.DataFrame(
            [
                [0.04, 0.01, 0.015, 0.01],
                [0.01, 0.05, 0.012, 0.01],
                [0.015, 0.012, 0.09, 0.014],
                [0.01, 0.01, 0.014, 0.035]
            ],
            index=self.returns.index,
            columns=self.returns.index
        )

    def test_optimizer_goals_and_constraints(self):
        for goal in ("max_sharpe", "min_vol"):
            weights = optimize_portfolio(
                self.returns,
                self.covariance,
                0.30,
                goal
            )
            self.assertAlmostEqual(weights.sum(), 1.0)
            self.assertTrue((weights >= -1e-8).all())
            self.assertTrue((weights <= 0.30 + 1e-8).all())

    def test_infeasible_three_stock_cap(self):
        with self.assertRaises(ValueError):
            optimize_portfolio(
                self.returns.iloc[:3],
                self.covariance.iloc[:3, :3],
                0.30,
                "min_vol"
            )


class TestMandateParser(unittest.TestCase):

    def test_parser_validation_and_mocked_api(self):
        response = Mock()
        response.text = (
            '{"tickers": ["AAPL", "MSFT", "NVDA", "GOOG", "AAPL"], '
            '"max_weight": 0.30, "goal": "min_vol"}'
        )
        client = Mock()
        client.models.generate_content.return_value = response

        with patch.dict(
            "os.environ",
            {"GEMINI_API_KEY": "test-key"}
        ), patch.object(
            mandate_parser.genai,
            "Client",
            return_value=client
        ):
            result = mandate_parser.parse_investor_mandate(
                "Invest in four companies, cap each at 30%, "
                "and minimize volatility."
            )

        self.assertEqual(
            result["tickers"],
            ["AAPL", "MSFT", "NVDA", "GOOG"]
        )

    def test_missing_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                mandate_parser.parse_investor_mandate(
                    "Invest in AAPL and MSFT."
                )


class TestRiskEngine(unittest.TestCase):

    def setUp(self):
        self.log_returns = pd.DataFrame(
            {
                "AAPL": [0.001, -0.002, 0.003, 0.001, -0.001],
                "MSFT": [0.002, 0.001, -0.001, 0.002, 0.001]
            }
        )
        self.weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40})

    def test_simulation_and_risk_metrics(self):
        first = run_monte_carlo(
            self.log_returns,
            self.weights,
            100000,
            1000,
            10,
            42
        )
        second = run_monte_carlo(
            self.log_returns,
            self.weights,
            100000,
            1000,
            10,
            42
        )

        self.assertEqual(first.shape, (11, 1000))
        self.assertTrue(np.all(first.iloc[0].to_numpy() == 100000))
        self.assertTrue(first.equals(second))

        var = calculate_var(first, 100000)
        cvar = calculate_cvar(first, 100000)
        self.assertGreaterEqual(cvar, var - 1e-8)

    def test_small_simulation_and_zero_volatility(self):
        constant_returns = pd.DataFrame(
            {
                "AAPL": [0.001] * 5,
                "MSFT": [0.001] * 5
            }
        )
        weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})

        result = run_monte_carlo(
            constant_returns,
            weights,
            initial_investment=1000,
            num_simulations=2,
            num_days=3,
            random_seed=7
        )

        self.assertEqual(result.shape, (4, 2))
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertTrue((result.iloc[-1] > result.iloc[0]).all())

    def test_negative_var_is_not_clipped(self):
        simulated_values = pd.DataFrame(
            [[100.0, 100.0], [110.0, 120.0]]
        )

        var = calculate_var(simulated_values, 100.0)
        cvar = calculate_cvar(simulated_values, 100.0)

        self.assertLess(var, 0)
        self.assertLess(cvar, 0)
        self.assertGreaterEqual(cvar, var - 1e-8)

    def test_batching_is_reproducible(self):
        original_batch_size = risk_engine.SIMULATION_BATCH_SIZE
        try:
            risk_engine.SIMULATION_BATCH_SIZE = 2
            first = run_monte_carlo(
                self.log_returns,
                self.weights,
                num_simulations=5,
                num_days=4,
                random_seed=123
            )
            second = run_monte_carlo(
                self.log_returns,
                self.weights,
                num_simulations=5,
                num_days=4,
                random_seed=123
            )
        finally:
            risk_engine.SIMULATION_BATCH_SIZE = original_batch_size

        pd.testing.assert_frame_equal(first, second)

    def test_different_batch_sizes_preserve_output_shape(self):
        original_batch_size = risk_engine.SIMULATION_BATCH_SIZE
        try:
            risk_engine.SIMULATION_BATCH_SIZE = 2
            small_batches = run_monte_carlo(
                self.log_returns,
                self.weights,
                num_simulations=5,
                num_days=4,
                random_seed=123
            )
            risk_engine.SIMULATION_BATCH_SIZE = 4
            larger_batches = run_monte_carlo(
                self.log_returns,
                self.weights,
                num_simulations=5,
                num_days=4,
                random_seed=123
            )
        finally:
            risk_engine.SIMULATION_BATCH_SIZE = original_batch_size

        self.assertEqual(small_batches.shape, (5, 5))
        self.assertEqual(larger_batches.shape, (5, 5))

    def test_unsafe_simulation_request_is_rejected(self):
        with self.assertRaises(ValueError):
            run_monte_carlo(
                self.log_returns,
                self.weights,
                num_simulations=risk_engine.MAX_SIMULATIONS + 1,
                num_days=1
            )

    def test_invalid_initial_investment_is_rejected(self):
        for invalid_value in (float("nan"), float("inf"), -1):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises((TypeError, ValueError)):
                    run_monte_carlo(
                        self.log_returns,
                        self.weights,
                        initial_investment=invalid_value,
                        num_simulations=2,
                        num_days=2
                    )

class TestMainIntegration(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(7)
        self.prices = pd.DataFrame(
            100 * np.exp(
                np.cumsum(
                    rng.normal(0.0002, 0.01, (260, 4)),
                    axis=0
                )
            ),
            columns=["AAPL", "MSFT", "NVDA", "GOOG"]
        )
        self.raw_data = pd.concat(
            {"Adj Close": self.prices},
            axis=1
        )
        self.mandate = {
            "tickers": ["AAPL", "MSFT", "NVDA", "GOOG"],
            "max_weight": 0.30,
            "goal": "min_vol"
        }

    def test_simple_returns_for_optimization_and_log_for_monte_carlo(self):
        original_optimizer = main.optimize_portfolio
        original_monte_carlo = main.run_monte_carlo
        captured = {}

        def capture_optimizer(**kwargs):
            captured["optimizer_returns"] = kwargs["expected_returns"]
            return original_optimizer(**kwargs)

        def capture_monte_carlo(**kwargs):
            captured["monte_carlo_returns"] = kwargs["log_returns"]
            return original_monte_carlo(**kwargs)

        callbacks = []
        with patch.object(
            main,
            "parse_investor_mandate",
            return_value=self.mandate
        ), patch.object(
            main,
            "download_stock_data",
            return_value=self.raw_data
        ), patch.object(
            main,
            "optimize_portfolio",
            side_effect=capture_optimizer
        ), patch.object(
            main,
            "run_monte_carlo",
            side_effect=capture_monte_carlo
        ):
            results = main.analyze_portfolio(
                "test mandate",
                100000,
                "2020-01-01",
                "2025-01-01",
                0.04,
                100,
                10,
                42,
                callbacks.append
            )

        simple_returns = calculate_simple_returns(self.prices)
        log_returns = calculate_log_returns(self.prices)
        expected_annual_returns = calculate_annual_returns(simple_returns)

        pd.testing.assert_series_equal(
            captured["optimizer_returns"],
            expected_annual_returns
        )
        pd.testing.assert_frame_equal(
            captured["monte_carlo_returns"],
            log_returns
        )
        self.assertEqual(
            callbacks,
            [
                "Parsing investment instructions...",
                "Downloading historical stock data...",
                "Calculating financial statistics...",
                "Optimizing portfolio...",
                "Running Monte Carlo simulations...",
                "Generating analysis report..."
            ]
        )
        self.assertEqual(
            results["simulated_values"].shape,
            (11, 100)
        )

    def test_risk_free_rate_reaches_optimizer(self):
        captured = {}

        def capture_optimizer(**kwargs):
            captured["risk_free_rate"] = kwargs["risk_free_rate"]
            return pd.Series(
                0.25,
                index=self.mandate["tickers"]
            )

        with patch.object(
            main,
            "parse_investor_mandate",
            return_value=self.mandate
        ), patch.object(
            main,
            "download_stock_data",
            return_value=self.raw_data
        ), patch.object(
            main,
            "optimize_portfolio",
            side_effect=capture_optimizer
        ):
            main.analyze_portfolio(
                "test mandate",
                100000,
                "2020-01-01",
                "2025-01-01",
                -0.01,
                10,
                5,
                42
            )

        self.assertEqual(captured["risk_free_rate"], -0.01)

    def test_invalid_investment_amount(self):
        with self.assertRaises(ValueError):
            main.analyze_portfolio(
                "test mandate",
                initial_investment=0
            )

    def test_complete_pipeline(self):
        with patch.object(
            main,
            "parse_investor_mandate",
            return_value=self.mandate
        ), patch.object(
            main,
            "download_stock_data",
            return_value=self.raw_data
        ):
            results = main.analyze_portfolio(
                "test mandate",
                initial_investment=100000,
                start_date="2020-01-01",
                end_date="2025-01-01",
                num_simulations=100,
                num_days=10,
                random_seed=42
            )

        self.assertEqual(results["tickers"], self.mandate["tickers"])
        self.assertAlmostEqual(
            results["optimized_weights"].sum(),
            1.0
        )
        self.assertIn("risk_summary", results)

    def test_currency_disclaimers_are_reported(self):
        results = {
            "tickers": ["AAPL"],
            "goal": "min_vol",
            "max_weight": 1.0,
            "initial_investment": 1000.0,
            "actual_start_date": "2020-01-01",
            "actual_end_date": "2020-12-31",
            "removed_rows": 0,
            "data_quality": {
                "total_rows_before_cleaning": 253,
                "total_rows_after_cleaning": 253,
                "removed_rows_percentage": 0.20
            },
            "optimized_weights": pd.Series({"AAPL": 1.0}),
            "expected_return": 0.10,
            "volatility": 0.20,
            "sharpe_ratio": 0.30,
            "risk_free_rate": 0.04,
            "num_simulations": 2,
            "num_days": 1,
            "risk_summary": {
                "mean_final_value": 1000.0,
                "median_final_value": 1000.0,
                "probability_of_loss": 0.0,
                "var": -1.0,
                "cvar": -1.0
            }
        }

        with patch("builtins.print") as print_mock:
            main.print_analysis_report(results, currency_label="PKR")

        output = "\n".join(
            str(call.args[0])
            for call in print_mock.call_args_list
            if call.args
        )
        self.assertIn("conversion is performed", output)
        self.assertIn("exchange-rate movements", output)
        self.assertIn("Transaction fees", output)
        self.assertIn("does not include currency conversion or FX risk", output)
        self.assertIn("More than 10% of historical rows were removed", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
