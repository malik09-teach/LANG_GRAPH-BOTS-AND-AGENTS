import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

class BayesianOptimizer:
    def __init__(self):
        # We store past experiments: X (formulation parameters), Y (stability scores)
        self.train_X = None
        self.train_Y = None
        self.model = None

    def update_data(self, x_new: list, y_new: float):
        """Update the dataset with a new experiment result."""
        x_tensor = torch.tensor([x_new], dtype=torch.float64)
        y_tensor = torch.tensor([[y_new]], dtype=torch.float64)

        if self.train_X is None:
            self.train_X = x_tensor
            self.train_Y = y_tensor
        else:
            self.train_X = torch.cat([self.train_X, x_tensor])
            self.train_Y = torch.cat([self.train_Y, y_tensor])
            
        self._fit_model()

    def _fit_model(self):
        """Fit the Gaussian Process surrogate model."""
        if self.train_X.shape[0] < 2:
            return # Need at least 2 points to fit properly
            
        self.model = SingleTaskGP(self.train_X, self.train_Y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def suggest_next_experiment(self, bounds: list, num_candidates: int = 3) -> list:
        """
        Suggest the top 'num_candidates' formulations to test next.
        bounds: List of tuples (min, max) for each continuous parameter.
        """
        if self.train_X is None or self.train_X.shape[0] < 2:
            # Cold start: return random points within bounds
            import random
            candidates = []
            for _ in range(num_candidates):
                point = [random.uniform(b[0], b[1]) for b in bounds]
                candidates.append(point)
            return candidates

        # Use Expected Improvement (EI)
        best_f = self.train_Y.max()
        EI = ExpectedImprovement(self.model, best_f=best_f)

        bounds_tensor = torch.tensor(bounds, dtype=torch.float64).T
        
        candidates, _ = optimize_acqf(
            acq_function=EI,
            bounds=bounds_tensor,
            q=num_candidates, # number of candidates
            num_restarts=5,
            raw_samples=20,
        )
        
        return candidates.tolist()
