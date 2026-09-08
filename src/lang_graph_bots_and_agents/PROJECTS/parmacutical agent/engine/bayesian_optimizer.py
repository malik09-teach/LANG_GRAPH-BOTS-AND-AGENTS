import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
import random

class BayesianOptimizer:
    def __init__(self, bounds: list[tuple[float, float]]):
        """
        Initialize the BO engine with the bounds for each variable.
        bounds: List of tuples (min, max) for each variable defined by the user.
        """
        self.bounds = bounds
        self.num_dims = len(bounds)

    def suggest_next_experiment(self, history: list[dict], num_candidates: int = 5) -> list[list[float]]:
        """
        Suggest candidates based on the real lab history provided by the user.
        history: List of dicts, e.g., [{"candidate": [x1, x2], "score": 85.0}, ...]
        """
        if len(history) < 2:
            # Cold start: suggest random points within user-defined bounds
            candidates = []
            for _ in range(num_candidates):
                point = [random.uniform(b[0], b[1]) for b in self.bounds]
                candidates.append(point)
            return candidates

        # Build tensors from history
        X_data = []
        Y_data = []
        for entry in history:
            X_data.append(entry["candidate"])
            Y_data.append([entry["score"]])
            
        train_X = torch.tensor(X_data, dtype=torch.float64)
        train_Y = torch.tensor(Y_data, dtype=torch.float64)

        # Fit GP
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Acquisition function
        best_f = train_Y.max()
        EI = ExpectedImprovement(model, best_f=best_f)

        bounds_tensor = torch.tensor(self.bounds, dtype=torch.float64).T
        
        # Optimize to find best candidates
        candidates, _ = optimize_acqf(
            acq_function=EI,
            bounds=bounds_tensor,
            q=num_candidates, 
            num_restarts=5,
            raw_samples=20,
        )
        
        return candidates.tolist()
