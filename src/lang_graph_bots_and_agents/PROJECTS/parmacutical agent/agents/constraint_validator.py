from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Using a dummy model for out-of-the-box execution without API keys. 
# In production, replace with ChatOpenAI or similar.
class DummyLLM:
    def __init__(self):
        pass
        
    def invoke(self, prompt: str) -> str:
        # Simple heuristic matching to simulate LLM reasoning based on RAG knowledge
        if "excipient_a: 0.6" in prompt or "excipient_a: 0.7" in prompt or "excipient_a: 0.8" in prompt or "excipient_a: 0.9" in prompt or "excipient_a: 1.0" in prompt:
            if "processing_temp: 70" in prompt or "processing_temp: 80" in prompt or "processing_temp: 65" in prompt:
                return "REJECT: Phase separation likely at high temp and high excipient A."
        
        # Parse the values
        try:
            a_idx = prompt.find("excipient_a:")
            b_idx = prompt.find("excipient_b:")
            a_val = float(prompt[a_idx+12:a_idx+16])
            b_val = float(prompt[b_idx+12:b_idx+16])
            if a_val + b_val > 1.0:
                return "REJECT: Total excipient ratio exceeds 1.0."
        except:
            pass
            
        return "APPROVE: No physical constraints violated."

class ConstraintValidatorAgent:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.llm = DummyLLM() # Replace with ChatOpenAI(model="gpt-4o")
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "candidate"],
            template="""
            You are a Pharmaceutical Formulation Constraint Validator.
            Using the provided scientific literature and historical failure context, evaluate if the proposed candidate is physically viable.
            
            Context:
            {context}
            
            Candidate:
            {candidate}
            
            Output 'APPROVE' if it is viable. Output 'REJECT: [Reason]' if it violates physical constraints.
            """
        )
        
    def validate(self, candidate_params: list) -> tuple[bool, str]:
        """
        Takes [excipient_a, excipient_b, processing_temp].
        Returns (is_valid, reason).
        """
        # Format candidate
        candidate_str = f"excipient_a: {candidate_params[0]:.2f}, excipient_b: {candidate_params[1]:.2f}, processing_temp: {candidate_params[2]:.2f}"
        
        # Retrieve context
        context = self.knowledge_base.query("physical constraints phase separation excipient ratio")
        
        # Format prompt
        prompt = self.prompt_template.format(context=context, candidate=candidate_str)
        
        # Get LLM response
        response = self.llm.invoke(prompt)
        
        is_valid = "APPROVE" in response.upper()
        return is_valid, response
