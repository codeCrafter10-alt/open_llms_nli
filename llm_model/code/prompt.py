"""
Prompt template for Mistral-7B-Instruct-v0.2 model NLI evaluation
"""

def create_prompt(premise, hypothesis):
    """
    Create an NLI classification prompt.

    Args:
        premise (str): The premise statement.
        hypothesis (str): The hypothesis statement.

    Returns:
        str: A formatted prompt string for the model.
    """
    prompt = f"""
    You are an expert Natural Language Inference (NLI) model. Your task is to determine the relationship between the following premise and hypothesis.

    The possible labels(final answer) are:
    1. Entailment: The hypothesis logically follows from the premise.
    2. Contradiction: The hypothesis logically contradicts the premise.
    3. Neutral: The hypothesis is neither entailed nor contradicted by the premise.

    Analyze the premise and hypothesis carefully before deciding on the relationship. 
    Provide your final answer as ONLY one word and as one of the three labels: Entailment, Contradiction, or Neutral.

    Here are some examples:

    Example 1:
    Premise: A person is riding a bicycle on the street.
    Hypothesis: A person is using a bicycle.
    Label: Entailment

    Example 2:
    Premise: A man is walking to a coffee shop.
    Hypothesis: The man is sleeping in his bed.
    Label: Contradiction

    Example 3:
    Premise: A dad and his son are smiling.
    Hypothesis: Two people are smiling and laughing.
    Label: Neutral

    Now classify the following example, outputting ONLY the label as your final answer:
    Premise: {premise}
    Hypothesis: {hypothesis}
    Label: 
    """

    return prompt.strip()