"""Module for simulating realistic bank transaction data for testing."""

import numpy as np
import pandas as pd
from typing import List


class FinancialSimulator:
    """Generates realistic synthetic bank transactions."""

    def __init__(self, num_accounts: int = 10, days: int = 365) -> None:
        """Initializes the simulator.

        Args:
            num_accounts: Number of unique accounts to simulate.
            days: Number of days of history per account.
        """
        self.num_accounts = num_accounts
        self.days = days
        self.date_range = pd.date_range(start="2023-01-01", periods=days, freq="D")

    def generate(self) -> pd.DataFrame:
        """Creates the synthetic dataset.

        Returns:
            DataFrame with [account_id, date, amount] columns.
        """
        all_data = []

        for acc_id in range(1, self.num_accounts + 1):
            # 1. Monthly Salary (Inflow)
            salary = 3000.0 + np.random.normal(0, 200)
            
            # 2. Fixed Monthly Rent (Outflow)
            rent = -1200.0

            for i, date in enumerate(self.date_range):
                # Add Salary on the 1st
                if date.day == 1:
                    all_data.append([acc_id, date, salary])
                
                # Add Rent on the 5th
                if date.day == 5:
                    all_data.append([acc_id, date, rent])

                # 3. Daily Spending (Noise)
                # 30% chance of a transaction on any given day
                if np.random.rand() < 0.3:
                    spend = -np.random.gamma(shape=2, scale=20) # Gamma distribution for spending
                    all_data.append([acc_id, date, spend])

        df = pd.DataFrame(all_data, columns=["account_id", "date", "amount"])
        return df.sort_values(by=["account_id", "date"]).reset_index(drop=True)
