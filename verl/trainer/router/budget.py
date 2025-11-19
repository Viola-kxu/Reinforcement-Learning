from __future__ import annotations


class RouterBudgetController:
    """Maintains the Lagrange multiplier that enforces the router's tool-use budget."""

    def __init__(self, budget: float, lr: float, init_lambda: float = 0.0):
        if budget < 0:
            raise ValueError(f"router budget must be non-negative, got {budget}")
        if lr <= 0:
            raise ValueError(f"router lagrange lr must be positive, got {lr}")
        self.budget = float(budget)
        self.lr = float(lr)
        self.value = max(0.0, float(init_lambda))

    def update(self, mean_cost: float) -> float:
        """Dual ascent step. Returns the updated lambda value."""
        diff = float(mean_cost) - self.budget
        self.value = max(0.0, self.value + self.lr * diff)
        return self.value


