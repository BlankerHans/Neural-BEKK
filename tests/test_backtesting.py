import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import t as student_t

from backtesting import (
    build_fz_loss_matrix,
    calculate_fz_loss,
    run_backtest_suite,
    run_fz_dm_comparison_plan,
    run_fz_dm_test,
    run_kupiec_test,
    run_modified_dm_test,
    run_dq_test,
)


class BacktestingTests(unittest.TestCase):
    def test_var_hits_include_returns_equal_to_var(self):
        returns = np.array([-1.0, -0.8, -1.2])
        var = np.array([-1.0, -1.0, -1.1])

        result = run_kupiec_test(returns=returns, var=var, alpha=0.1)

        self.assertEqual(result["n_exceptions"], 2)

    def test_dq_uses_segmgarch_result_and_lag_mapping(self):
        calls = []
        returns = np.array([0.2, -1.4, 0.1, -1.2, -1.3, 0.3, 0.0, -1.5, 0.2])
        var = np.array([-1.0, -1.0, -0.9, -1.0, -1.1, -1.0, -0.8, -1.0, -0.9])

        class FakeSegMGarch:
            @staticmethod
            def DQtest(**kwargs):
                calls.append(kwargs)
                return [np.array([[7.5]]), np.array([[0.277]])]

        with patch(
            "backtesting._load_segmgarch",
            return_value=(None, np.asarray, np.asarray, FakeSegMGarch),
        ):
            result = run_dq_test(
                returns=returns,
                var=var,
                alpha=0.2,
                hit_lags=(1, 2),
                var_lags=(0,),
                return_lags=(3,),
            )

        self.assertEqual(result["n_obs"], len(returns) - 3)
        self.assertEqual(result["implementation"], "segMGarch::DQtest")
        self.assertEqual(result["df"], 5)
        self.assertEqual(result["dq_stat"], 7.5)
        self.assertEqual(result["p_value"], 0.277)
        self.assertEqual(
            result["columns"],
            ["intercept", "var_forecast", "hit_lag1", "hit_lag2", "return_sq_lag3"],
        )
        self.assertEqual(
            result["segmgarch_lags"],
            {"lag": 3, "lag_hit": 2, "lag_var": 1},
        )
        self.assertEqual(float(calls[0]["VaR_level"][0]), 0.8)
        self.assertEqual(int(calls[0]["lag"][0]), 3)
        self.assertEqual(int(calls[0]["lag_hit"][0]), 2)
        self.assertEqual(int(calls[0]["lag_var"][0]), 1)

    def test_dq_rejects_samples_shorter_than_requested_lags(self):
        returns = np.array([0.0, -1.0])
        var = np.array([-0.5, -0.5])

        with self.assertRaisesRegex(ValueError, "requires more observations"):
            run_dq_test(returns=returns, var=var, alpha=0.1, hit_lags=(1, 2))

    def test_dq_rejects_specs_not_supported_by_segmgarch(self):
        returns = np.array([0.2, -1.4, 0.1, -1.2, -1.3])
        var = np.full_like(returns, -1.0)

        with self.assertRaisesRegex(ValueError, "consecutive hit lags"):
            run_dq_test(returns=returns, var=var, alpha=0.2, hit_lags=(1, 3))

        with self.assertRaisesRegex(ValueError, "current VaR forecast"):
            run_dq_test(returns=returns, var=var, alpha=0.2, var_lags=())

        with self.assertRaisesRegex(ValueError, "one squared-return lag"):
            run_dq_test(
                returns=returns,
                var=var,
                alpha=0.2,
                hit_lags=(1,),
                return_lags=(1, 2),
            )

    def test_backtest_suite_returns_expected_top_level_keys(self):
        returns = np.array([0.1, -1.4, 0.2, -1.8, -0.3, -1.1])
        var = np.full_like(returns, -1.0)
        es = var - 0.4

        with (
            patch("backtesting.run_dq_test", return_value={"test": "dynamic_quantile"}),
            patch("backtesting.run_cc_backtest", return_value={"test": "cc"}),
            patch("backtesting.run_er_backtest", return_value={"test": "er"}),
            patch("backtesting.run_esr_backtest", return_value={"test": "esr"}),
        ):
            result = run_backtest_suite(returns=returns, var=var, es=es, alpha=0.1)

        self.assertEqual(
            set(result),
            {"meta", "var", "cc", "er", "esr"},
        )

    def test_calculate_fz_loss_matches_patton_ziegel_chen_formula(self):
        returns = np.array([-1.1, -0.4, -1.3])
        var = np.array([-1.0, -0.9, -1.2])
        es = np.array([-1.6, -1.5, -1.7])
        alpha = 0.1

        result = calculate_fz_loss(returns=returns, var=var, es=es, alpha=alpha)

        hits = (returns <= var).astype(float)
        expected = hits * (returns - var) / (alpha * es) + var / es + np.log(-es) - 1.0
        np.testing.assert_allclose(result, expected)

    def test_calculate_fz_loss_rejects_non_negative_es(self):
        with self.assertRaisesRegex(ValueError, "negative ES"):
            calculate_fz_loss(
                returns=np.array([-1.1]),
                var=np.array([-1.0]),
                es=np.array([1.6]),
                alpha=0.1,
            )

    def test_build_fz_loss_matrix_aligns_models_and_drops_nonfinite_rows(self):
        returns = np.array([-1.1, -0.4, -1.3])
        model_forecasts = {
            "a": {
                "returns": returns,
                "var": np.array([-1.0, np.nan, -1.2]),
                "es": np.array([-1.6, -1.5, -1.7]),
            },
            "b": {
                "returns": returns.copy(),
                "var": np.array([-0.9, -0.8, -1.1]),
                "es": np.array([-1.4, -1.3, -1.5]),
            },
        }

        model_names, loss_matrix = build_fz_loss_matrix(
            model_forecasts=model_forecasts,
            alpha=0.1,
        )

        self.assertEqual(model_names, ("a", "b"))
        self.assertEqual(loss_matrix.shape, (2, 2))
        expected_a = calculate_fz_loss(
            returns=returns[[0, 2]],
            var=np.array([-1.0, -1.2]),
            es=np.array([-1.6, -1.7]),
            alpha=0.1,
        )
        expected_b = calculate_fz_loss(
            returns=returns[[0, 2]],
            var=np.array([-0.9, -1.1]),
            es=np.array([-1.4, -1.5]),
            alpha=0.1,
        )
        np.testing.assert_allclose(loss_matrix[:, 0], expected_a)
        np.testing.assert_allclose(loss_matrix[:, 1], expected_b)

    def test_build_fz_loss_matrix_rejects_unaligned_returns(self):
        model_forecasts = {
            "a": {
                "returns": np.array([-1.1, -0.4]),
                "var": np.array([-1.0, -0.9]),
                "es": np.array([-1.6, -1.5]),
            },
            "b": {
                "returns": np.array([-1.1, -0.5]),
                "var": np.array([-0.9, -0.8]),
                "es": np.array([-1.4, -1.3]),
            },
        }

        with self.assertRaisesRegex(ValueError, "not aligned"):
            build_fz_loss_matrix(model_forecasts=model_forecasts, alpha=0.1)

    def test_modified_dm_test_uses_hln_correction_on_loss_differential(self):
        loss_differential = np.array([-0.2, -0.1, 0.0, -0.3])

        result = run_modified_dm_test(
            loss_differential=loss_differential,
            h=1,
            cl=0.05,
        )

        centered = loss_differential - np.mean(loss_differential)
        long_run_variance = float(np.dot(centered, centered) / len(centered))
        dm_stat = float(
            np.mean(loss_differential) / np.sqrt(long_run_variance / len(centered))
        )
        hln_correction = float(np.sqrt((len(centered) - 1.0) / len(centered)))
        modified_dm_stat = dm_stat * hln_correction
        expected_p_value = float(2.0 * student_t.sf(abs(modified_dm_stat), df=3))

        self.assertEqual(result["implementation"], "python_harvey_leybourne_newbold_1997")
        self.assertEqual(result["alternative"], "two-sided")
        self.assertEqual(result["df"], 3)
        self.assertFalse(result["reject"])
        self.assertAlmostEqual(result["long_run_variance"], long_run_variance)
        self.assertAlmostEqual(result["dm_stat"], dm_stat)
        self.assertAlmostEqual(result["hln_correction"], hln_correction)
        self.assertAlmostEqual(result["modified_dm_stat"], modified_dm_stat)
        self.assertAlmostEqual(result["stat"], modified_dm_stat)
        self.assertAlmostEqual(result["p_value"], expected_p_value)
        self.assertAlmostEqual(result["pval"], expected_p_value)

    def test_fz_dm_test_uses_model_1_minus_model_2_loss_difference(self):
        with (
            patch(
                "backtesting.calculate_fz_loss",
                side_effect=[np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 4.0])],
            ),
            patch(
                "backtesting.run_modified_dm_test",
                return_value={
                    "test": "modified_diebold_mariano",
                    "stat": -1.5,
                    "p_value": 0.13,
                },
            ) as dm_mock,
        ):
            result = run_fz_dm_test(
                returns=np.array([-1.0, -0.8, -1.2]),
                var_1=np.array([-0.9, -0.9, -1.1]),
                es_1=np.array([-1.4, -1.4, -1.6]),
                var_2=np.array([-0.8, -0.8, -1.0]),
                es_2=np.array([-1.3, -1.3, -1.5]),
                alpha=0.1,
                model_1="a",
                model_2="b",
            )

        np.testing.assert_allclose(
            dm_mock.call_args.kwargs["loss_differential"],
            np.array([-1.0, 0.0, -1.0]),
        )
        self.assertEqual(
            result["implementation"],
            "python_fz_loss + python_hln_modified_dm",
        )
        self.assertEqual(result["preferred_by_mean_loss"], "a")

    def test_fz_dm_comparison_plan_groups_benchmark_structured_and_pairwise(self):
        model_forecasts = {
            "a": {
                "returns": np.array([0.1, -0.2, 0.0]),
                "var": np.array([-0.3, -0.3, -0.3]),
                "es": np.array([-0.5, -0.5, -0.5]),
            },
            "b": {
                "returns": np.array([0.1, -0.2, 0.0]),
                "var": np.array([-0.4, -0.4, -0.4]),
                "es": np.array([-0.6, -0.6, -0.6]),
            },
            "benchmark": {
                "returns": np.array([0.1, -0.2, 0.0]),
                "var": np.array([-0.2, -0.2, -0.2]),
                "es": np.array([-0.4, -0.4, -0.4]),
            },
        }

        def fake_dm(**kwargs):
            return {
                "model_1": kwargs["model_1"],
                "model_2": kwargs["model_2"],
                "mean_loss_differential": -0.1,
            }

        with patch("backtesting.run_fz_dm_test", side_effect=fake_dm):
            result = run_fz_dm_comparison_plan(
                model_forecasts=model_forecasts,
                alpha=0.1,
                benchmark="benchmark",
                structured_pairs=(("a", "b"),),
                include_pairwise=True,
            )

        self.assertEqual(set(result), {"meta", "benchmark", "structured", "pairwise"})
        self.assertEqual(set(result["benchmark"]), {"a_vs_benchmark", "b_vs_benchmark"})
        self.assertEqual(set(result["structured"]), {"a_vs_b"})
        self.assertEqual(set(result["pairwise"]), {"a_vs_b", "a_vs_benchmark", "b_vs_benchmark"})
        self.assertEqual(result["meta"]["negative_mean_loss_differential_favors"], "model_1")


if __name__ == "__main__":
    unittest.main()
