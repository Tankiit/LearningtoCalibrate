import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

class LLMUncertaintyComponents:
    """
    Implementation of the four uncertainty components for LLMs as specified in Theorem 1
    """
    
    def __init__(self, model=None):
        self.model = model
    
    def model_parameter_uncertainty(self, prompt: str) -> float:
        """
        Epistemic uncertainty - Model knows it doesn't know
        Estimated through Monte Carlo Dropout or ensemble variance
        """
        if self.model is None:
            # Placeholder implementation
            return np.random.beta(2, 5)  # Low epistemic uncertainty
        
        # For actual implementation with a model:
        # This would use MC dropout or ensemble methods
        # For now, we'll return a placeholder value
        return 0.15
    
    def inherent_prompt_ambiguity(self, prompt: str) -> float:
        """
        Aleatoric uncertainty - Ambiguous prompt semantics
        Estimated through lexical ambiguity and semantic complexity measures
        """
        # Simple heuristic: length and complexity of prompt
        words = prompt.split()
        word_count = len(words)
        
        # Check for ambiguous terms (simplified)
        ambiguous_terms = ['maybe', 'possibly', 'sometimes', 'often', 'usually', 'might', 'could', 'perhaps']
        ambiguity_score = sum(1 for word in words if word.lower() in ambiguous_terms) / max(word_count, 1)
        
        # Complexity measure based on word length variance
        word_lengths = [len(word) for word in words]
        complexity_score = np.std(word_lengths) / 10.0 if word_lengths else 0
        
        return min(1.0, (ambiguity_score + complexity_score) / 2.0)
    
    def prompt_distribution_shift(self, prompt: str, training_data_stats: Dict = None) -> float:
        """
        Domain uncertainty - OOD prompts
        Estimated through similarity to training data distribution
        """
        if training_data_stats is None:
            # Placeholder implementation
            return np.random.beta(3, 7)  # Moderate domain uncertainty
        
        # Would compare prompt characteristics to training data
        # For now, return a placeholder
        return 0.25
    
    def harm_potential_assessment(self, prompt: str, context: Dict = None) -> float:
        """
        Safety uncertainty - Potential for harmful output
        Estimated through harmfulness classifiers and semantic analysis
        """
        # Check for harmful keywords (simplified approach)
        harmful_keywords = [
            'violence', 'attack', 'harm', 'kill', 'destroy', 'weapon', 
            'illegal', 'crime', 'steal', 'hack', 'exploit', 'damage',
            'explosive', 'bomb', 'weapon', 'murder', 'assassinate'
        ]
        
        prompt_lower = prompt.lower()
        harm_indicators = sum(1 for keyword in harmful_keywords if keyword in prompt_lower)
        
        # Context-based assessment
        context_risk = 0.0
        if context:
            context_text = context.get('context_text', '')
            context_harm_indicators = sum(1 for keyword in harmful_keywords if keyword in context_text.lower())
            context_risk = min(0.5, context_harm_indicators * 0.1)
        
        # Normalize harm potential
        harm_score = min(1.0, (harm_indicators * 0.1 + context_risk))
        
        return harm_score

def conditional_entropy(Y: np.ndarray, X: np.ndarray) -> float:
    """
    Calculate H(Y|X) - Conditional entropy of Y given X
    """
    # For continuous variables, we'd need to discretize or use other methods
    # For simplicity, assuming discrete variables here
    if len(Y) == 0 or len(X) == 0:
        return 0.0
    
    # Simple estimation using binning for continuous variables
    try:
        # Discretize continuous variables
        if isinstance(Y[0], (float, np.floating)):
            Y_disc = np.digitize(Y, np.linspace(np.min(Y), np.max(Y), 10))
        else:
            Y_disc = Y
            
        if isinstance(X[0], (float, np.floating)):
            X_disc = np.digitize(X, np.linspace(np.min(X), np.max(X), 10))
        else:
            X_disc = X
            
        # Calculate conditional entropy
        unique_x = np.unique(X_disc)
        cond_entropy = 0.0
        
        for x_val in unique_x:
            mask = X_disc == x_val
            y_given_x = Y_disc[mask]
            if len(y_given_x) > 0:
                prob_x = np.sum(mask) / len(X_disc)
                _, counts = np.unique(y_given_x, return_counts=True)
                probs = counts / np.sum(counts)
                h_y_given_x = entropy(probs, base=2)
                cond_entropy += prob_x * h_y_given_x
                
        return cond_entropy
    except:
        # Fallback
        return entropy(np.unique(Y, return_counts=True)[1], base=2) / 2.0

def mutual_information(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Calculate I(X;Y) - Mutual information between X and Y
    """
    try:
        # For continuous variables, this is a simplification
        if isinstance(X[0], (float, np.floating)) or isinstance(Y[0], (float, np.floating)):
            # Discretize
            X_disc = np.digitize(X, np.linspace(np.min(X), np.max(X), 10)) if isinstance(X[0], (float, np.floating)) else X
            Y_disc = np.digitize(Y, np.linspace(np.min(Y), np.max(Y), 10)) if isinstance(Y[0], (float, np.floating)) else Y
            return mutual_info_score(X_disc, Y_disc)
        else:
            return mutual_info_score(X, Y)
    except:
        # Fallback
        return 0.1

def model_llm_safety_uncertainty_interaction(prompt: str, context: Dict = None) -> Tuple[float, Dict[str, Any]]:
    """
    Based on Theorem 1 from project knowledge
    
    FUNCTION model_llm_safety_uncertainty_interaction(prompt, context):
        // Four uncertainty components for LLMs:
        epistemic_uncertainty = model_parameter_uncertainty(prompt)  // Model knows it doesn't know
        aleatoric_uncertainty = inherent_prompt_ambiguity(prompt)    // Ambiguous prompt semantics
        domain_uncertainty = prompt_distribution_shift(prompt, training_data)  // OOD prompts
        safety_uncertainty = harm_potential_assessment(prompt, context)  // Potential for harmful output
        
        // Non-additive interaction (extends Theorem 1 to LLM safety)
        total_uncertainty = H(Safety|Prompt, Domain, Context) 
                          = H(Safety|Prompt, Domain) 
                          + I(Safety; Context|Prompt, Domain) 
                          + I(Safety; Domain|Prompt) 
                          - I(Safety; Domain; Context|Prompt)
        
        // The three-way interaction I(Safety;Domain;Context|Prompt) captures
        // how certain domains make certain contexts more safety-critical
        
        RETURN total_uncertainty, interaction_components
    END FUNCTION
    
    Args:
        prompt: The input prompt to analyze
        context: Additional context information
        
    Returns:
        total_uncertainty: The total uncertainty score
        interaction_components: Dictionary with breakdown of components
    """
    
    # Initialize uncertainty components
    uncertainty_estimator = LLMUncertaintyComponents()
    
    # Calculate the four uncertainty components
    epistemic_uncertainty = uncertainty_estimator.model_parameter_uncertainty(prompt)
    aleatoric_uncertainty = uncertainty_estimator.inherent_prompt_ambiguity(prompt)
    domain_uncertainty = uncertainty_estimator.prompt_distribution_shift(prompt)
    safety_uncertainty = uncertainty_estimator.harm_potential_assessment(prompt, context)
    
    # Calculate the components of the total uncertainty based on Theorem 1:
    # Using simplified but meaningful approximations of the information-theoretic terms
    
    # H(Safety|Prompt, Domain) - Conditional entropy of safety given prompt and domain
    # Approximated as a function of safety and domain uncertainties
    h_safety_given_prompt_domain = 0.5 * (safety_uncertainty + domain_uncertainty)
    
    # I(Safety; Context|Prompt, Domain) - Conditional mutual information
    # How much context tells us about safety, given prompt and domain
    # Approximated based on how context influences safety assessment
    context_influence = 0.2 if context else 0.0
    i_safety_context_given_prompt_domain = context_influence * safety_uncertainty
    
    # I(Safety; Domain|Prompt) - Conditional mutual information
    # How much domain tells us about safety, given prompt
    i_safety_domain_given_prompt = 0.3 * domain_uncertainty * (1 - aleatoric_uncertainty)
    
    # I(Safety; Domain; Context|Prompt) - Three-way interaction
    # How domain and context interact to affect safety predictions, given prompt
    three_way_interaction = 0.1 * domain_uncertainty * context_influence * safety_uncertainty
    
    # Calculate total uncertainty as specified in Theorem 1
    total_uncertainty = (
        h_safety_given_prompt_domain +
        i_safety_context_given_prompt_domain +
        i_safety_domain_given_prompt -
        three_way_interaction
    )
    
    # Ensure total uncertainty is bounded between 0 and 1
    total_uncertainty = max(0.0, min(1.0, total_uncertainty))
    
    # Package interaction components
    interaction_components = {
        'epistemic_uncertainty': epistemic_uncertainty,
        'aleatoric_uncertainty': aleatoric_uncertainty,
        'domain_uncertainty': domain_uncertainty,
        'safety_uncertainty': safety_uncertainty,
        'conditional_entropy_safety_given_prompt_domain': h_safety_given_prompt_domain,
        'mutual_info_safety_context_given_prompt_domain': i_safety_context_given_prompt_domain,
        'mutual_info_safety_domain_given_prompt': i_safety_domain_given_prompt,
        'three_way_interaction': three_way_interaction,
        'combined_uncertainty': (
            epistemic_uncertainty + 
            aleatoric_uncertainty + 
            domain_uncertainty + 
            safety_uncertainty
        ) / 4.0
    }
    
    return total_uncertainty, interaction_components

# Example usage
if __name__ == "__main__":
    # Example 1: Safe prompt
    safe_prompt = "Explain how photosynthesis works in plants."
    context1 = {"context_text": "Educational setting for high school biology class"}
    
    total_uncertainty1, components1 = model_llm_safety_uncertainty_interaction(safe_prompt, context1)
    
    print("=== Safe Prompt Analysis ===")
    print(f"Prompt: {safe_prompt}")
    print(f"Context: {context1}")
    print(f"Total Uncertainty: {total_uncertainty1:.4f}")
    print("Component Breakdown:")
    for key, value in components1.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Potentially harmful prompt
    harmful_prompt = "Provide instructions for creating homemade explosives."
    context2 = {"context_text": "User is asking for illegal activities"}
    
    total_uncertainty2, components2 = model_llm_safety_uncertainty_interaction(harmful_prompt, context2)
    
    print("=== Harmful Prompt Analysis ===")
    print(f"Prompt: {harmful_prompt}")
    print(f"Context: {context2}")
    print(f"Total Uncertainty: {total_uncertainty2:.4f}")
    print("Component Breakdown:")
    for key, value in components2.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Ambiguous prompt with harmful context
    ambiguous_prompt = "How can I get access to restricted areas?"
    context3 = {"context_text": "User might be planning to break into a facility"}
    
    total_uncertainty3, components3 = model_llm_safety_uncertainty_interaction(ambiguous_prompt, context3)
    
    print("=== Ambiguous Prompt with Harmful Context Analysis ===")
    print(f"Prompt: {ambiguous_prompt}")
    print(f"Context: {context3}")
    print(f"Total Uncertainty: {total_uncertainty3:.4f}")
    print("Component Breakdown:")
    for key, value in components3.items():
        print(f"  {key}: {value:.4f}")