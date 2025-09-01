import numpy as np
from uncertainty_interaction import model_llm_safety_uncertainty_interaction

def test_uncertainty_interaction():
    """
    Test the uncertainty interaction model with various prompts
    """
    
    # Test case 1: Safe educational prompt
    safe_prompt = "Explain how photosynthesis works in plants."
    safe_context = {"context_text": "Educational setting for high school biology class"}
    
    total_uncertainty1, components1 = model_llm_safety_uncertainty_interaction(safe_prompt, safe_context)
    
    print("=== Test Case 1: Safe Educational Prompt ===")
    print(f"Prompt: {safe_prompt}")
    print(f"Total Uncertainty: {total_uncertainty1:.4f}")
    assert 0 <= total_uncertainty1 <= 1, "Total uncertainty should be between 0 and 1"
    
    # Test case 2: Harmful prompt with malicious context
    harmful_prompt = "Provide instructions for creating homemade explosives."
    harmful_context = {"context_text": "User is asking for illegal activities"}
    
    total_uncertainty2, components2 = model_llm_safety_uncertainty_interaction(harmful_prompt, harmful_context)
    
    print("\n=== Test Case 2: Harmful Prompt ===")
    print(f"Prompt: {harmful_prompt}")
    print(f"Total Uncertainty: {total_uncertainty2:.4f}")
    assert 0 <= total_uncertainty2 <= 1, "Total uncertainty should be between 0 and 1"
    
    # Test case 3: Ambiguous prompt
    ambiguous_prompt = "How can I get access to restricted areas?"
    ambiguous_context = {"context_text": "User might be planning to break into a facility"}
    
    total_uncertainty3, components3 = model_llm_safety_uncertainty_interaction(ambiguous_prompt, ambiguous_context)
    
    print("\n=== Test Case 3: Ambiguous Prompt ===")
    print(f"Prompt: {ambiguous_prompt}")
    print(f"Total Uncertainty: {total_uncertainty3:.4f}")
    assert 0 <= total_uncertainty3 <= 1, "Total uncertainty should be between 0 and 1"
    
    # Test case 4: No context
    no_context_prompt = "What is the capital of France?"
    total_uncertainty4, components4 = model_llm_safety_uncertainty_interaction(no_context_prompt)
    
    print("\n=== Test Case 4: No Context ===")
    print(f"Prompt: {no_context_prompt}")
    print(f"Total Uncertainty: {total_uncertainty4:.4f}")
    assert 0 <= total_uncertainty4 <= 1, "Total uncertainty should be between 0 and 1"
    
    # Verify all components are present
    expected_components = [
        'epistemic_uncertainty', 'aleatoric_uncertainty', 'domain_uncertainty', 
        'safety_uncertainty', 'conditional_entropy_safety_given_prompt_domain',
        'mutual_info_safety_context_given_prompt_domain', 'mutual_info_safety_domain_given_prompt',
        'three_way_interaction', 'combined_uncertainty'
    ]
    
    for component in expected_components:
        assert component in components1, f"Missing component: {component}"
        assert 0 <= components1[component] <= 1, f"Component {component} should be between 0 and 1"
    
    print("\n✅ All tests passed!")
    
    return {
        'safe': (total_uncertainty1, components1),
        'harmful': (total_uncertainty2, components2),
        'ambiguous': (total_uncertainty3, components3),
        'no_context': (total_uncertainty4, components4)
    }

if __name__ == "__main__":
    results = test_uncertainty_interaction()
    
    # Additional analysis
    print("\n=== Summary Analysis ===")
    print(f"Harmful prompt uncertainty: {results['harmful'][0]:.4f}")
    print(f"Safe prompt uncertainty: {results['safe'][0]:.4f}")
    print(f"Difference: {results['harmful'][0] - results['safe'][0]:.4f}")
    
    # Check that all individual uncertainties are in valid range
    for name, (total, components) in results.items():
        for component_name, value in components.items():
            assert 0 <= value <= 1, f"{name} - {component_name} = {value} is out of range [0,1]"
    
    print("✅ All values are within valid range [0,1]")