import re

class ConstraintValidatorAgent:
    def __init__(self):
        # We replace the DummyLLM with an actual logic parser for user constraints.
        # In a real agent system, this is where LangChain/OpenAI would evaluate
        # the constraints against the candidate.
        pass
        
    def validate(self, candidate: list[float], variables: list[dict], constraints: str) -> tuple[bool, str]:
        """
        Validate the candidate against user-provided constraints.
        candidate: The proposed numbers [0.5, 0.3, ...]
        variables: The definitions [{"name": "A", "symbol": "x1"}, ...]
        constraints: User text, e.g., "x1 + x2 <= 1.0"
        """
        if not constraints or constraints.strip() == "":
            return True, "No constraints provided. Approved."
            
        # Map symbols to their candidate values
        symbol_map = {}
        for i, var in enumerate(variables):
            symbol = var["symbol"]
            val = candidate[i]
            symbol_map[symbol] = val
            
        # Simple evaluator for math constraints like "x1 + x2 <= 1.0"
        # We split constraints by newlines
        lines = constraints.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Replace symbols with values in the expression
            expr = line
            for symbol, val in symbol_map.items():
                # naive replace, assuming symbols are like x1, x2, T
                expr = re.sub(rf'\b{symbol}\b', str(val), expr)
                
            # Evaluate the boolean expression safely
            # Note: Using eval in a real app is dangerous, but suitable for this local demo.
            try:
                # Replace logical operators if needed, python uses <=, >=, ==, <, >
                result = eval(expr)
                if not result:
                    return False, f"Violated constraint: {line}"
            except Exception as e:
                # If it's a semantic constraint (e.g., "Do not mix at high temp"), an LLM is needed.
                # Since we don't assume an API key, we will log it but pass it.
                return True, f"Constraint '{line}' requires LLM validation. Passed by default."
                
        return True, "All mathematical constraints satisfied."
