from engine.bayesian_optimizer import BayesianOptimizer
from agents.constraint_validator import ConstraintValidatorAgent

class AgentSupervisor:
    def __init__(self, variables: list[dict], constraints: str):
        """
        variables: [{"name": "Excipient A", "symbol": "x1", "min": 0.0, "max": 1.0}, ...]
        constraints: "x1 + x2 <= 1.0"
        """
        self.variables = variables
        self.constraints = constraints
        
        # Build bounds for BO
        bounds = [(float(v["min"]), float(v["max"])) for v in variables]
        self.optimizer = BayesianOptimizer(bounds=bounds)
        self.validator = ConstraintValidatorAgent()

    def get_next_proposal(self, history: list[dict]) -> dict:
        """
        history: [{"candidate": [0.5, 0.3], "score": 85.0}, ...]
        Returns the best candidate that passes constraints.
        """
        candidates = self.optimizer.suggest_next_experiment(history, num_candidates=10)
        
        for cand in candidates:
            is_valid, reason = self.validator.validate(cand, self.variables, self.constraints)
            if is_valid:
                return {"candidate": cand, "reason": reason}
                
        # If all fail, return the first one but flag it
        return {"candidate": candidates[0], "reason": "All candidates failed constraints. Returning top mathematical candidate."}
