import unittest

import pandas as pd

from return_transforms import approximate_treasury_return_from_yield


class TreasuryReturnTransformTests(unittest.TestCase):
    def test_constant_yield_returns_daily_carry(self):
        yields = pd.Series([5.0, 5.0])

        returns = approximate_treasury_return_from_yield(yields)

        self.assertAlmostEqual(returns.iloc[1], 0.05 / 252.0, places=12)

    def test_yield_increase_produces_negative_bond_return(self):
        yields = pd.Series([5.0, 5.1])

        returns = approximate_treasury_return_from_yield(yields)

        self.assertLess(returns.iloc[1], 0.0)

    def test_yield_decrease_produces_positive_bond_return(self):
        yields = pd.Series([5.0, 4.9])

        returns = approximate_treasury_return_from_yield(yields)

        self.assertGreater(returns.iloc[1], 0.0)


if __name__ == "__main__":
    unittest.main()
