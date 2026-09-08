import math
import random

class LabSimulator:
    """
    A deterministic physical simulator acting as a 'wet lab'.
    Evaluates a formulation and returns a stability score (0-100).
    Adds Gaussian noise to simulate experimental error.
    """
    def __init__(self):
        # We define a hidden optimal formulation configuration that the optimizer needs to find.
        self.optimal_ratio_a = 0.4
        self.optimal_ratio_b = 0.3
        self.optimal_temp = 25.0
        
    def execute_experiment(self, excipient_a: float, excipient_b: float, processing_temp: float) -> float:
        """
        Run the physical experiment.
        Parameters are expected in bounds:
        - excipient_a: [0.0, 1.0]
        - excipient_b: [0.0, 1.0]
        - temp: [10.0, 80.0]
        """
        # Physical constraint: If temp > 60 and excipient_a > 0.5, catastrophic failure (phase separation)
        if processing_temp > 60.0 and excipient_a > 0.5:
            return 0.0 # Failed experiment
            
        # Physical constraint: Ratio a + ratio b cannot exceed 1.0
        if excipient_a + excipient_b > 1.0:
            return 0.0 # Failed experiment
            
        # Calculate distance from optimal
        dist_a = (excipient_a - self.optimal_ratio_a) ** 2
        dist_b = (excipient_b - self.optimal_ratio_b) ** 2
        dist_temp = ((processing_temp - self.optimal_temp) / 70.0) ** 2 # Normalize temp
        
        total_dist = math.sqrt(dist_a + dist_b + dist_temp)
        
        # Base score (closer to 0 dist = higher score)
        base_score = max(0.0, 100.0 * (1.0 - total_dist))
        
        # Add experimental noise
        noise = random.gauss(0, 2.0)
        final_score = max(0.0, min(100.0, base_score + noise))
        
        return final_score
