# LLM Safety Uncertainty Interaction Model

This implementation provides a model for calculating the interaction between different types of uncertainty in Large Language Models (LLMs) based on Theorem 1 from the project knowledge.

## Overview

The model implements the function `model_llm_safety_uncertainty_interaction` which calculates a total uncertainty score based on four components:

1. **Epistemic Uncertainty**: Model's uncertainty about its own parameters (what it doesn't know)
2. **Aleatoric Uncertainty**: Uncertainty due to ambiguous or noisy input (inherent prompt ambiguity)
3. **Domain Uncertainty**: Uncertainty due to out-of-distribution inputs (distribution shift)
4. **Safety Uncertainty**: Uncertainty related to potential harmful outputs

## Theoretical Framework

Based on Theorem 1, the total uncertainty is calculated as:

```
total_uncertainty = H(Safety|Prompt, Domain, Context) 
                  = H(Safety|Prompt, Domain) 
                  + I(Safety; Context|Prompt, Domain) 
                  + I(Safety; Domain|Prompt) 
                  - I(Safety; Domain; Context|Prompt)
```

Where:
- H(Safety|Prompt, Domain) is the conditional entropy of safety given prompt and domain
- I(Safety; Context|Prompt, Domain) is the conditional mutual information between safety and context
- I(Safety; Domain|Prompt) is the conditional mutual information between safety and domain
- I(Safety; Domain; Context|Prompt) is the three-way interaction term

## Usage

```python
from uncertainty_interaction import model_llm_safety_uncertainty_interaction

# Example with a safe prompt
safe_prompt = "Explain how photosynthesis works in plants."
context = {"context_text": "Educational setting for high school biology class"}

total_uncertainty, components = model_llm_safety_uncertainty_interaction(safe_prompt, context)

print(f"Total Uncertainty: {total_uncertainty}")
print("Components:", components)
```

## Implementation Details

The implementation includes:

1. **LLM Uncertainty Components Class**: Calculates the four base uncertainty components
2. **Information-Theoretic Terms**: Approximates the conditional entropy and mutual information terms
3. **Interaction Calculation**: Combines all terms according to Theorem 1

## Examples

The implementation includes examples showing:

1. Safe prompts in educational contexts (low uncertainty)
2. Harmful prompts with malicious context (high uncertainty)
3. Ambiguous prompts with potentially harmful context (moderate uncertainty)

## Integration with L2R Framework

This implementation can be integrated with the existing Learning to Reject (L2R) framework to enhance safety mechanisms by providing a theoretically grounded uncertainty measure that accounts for complex interactions between different sources of uncertainty.