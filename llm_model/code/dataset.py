"""
Loading and preprocessing of the SNLI dataset for the NLI task.
"""

import pandas as pd

def load_dataset(file_path):
    """
    Load the SNLI test dataset.

    Args:
        file_path (str): The path to the SNLI dataset file.
    
    Returns:
        pd.DataFrame: A DataFrame containing the loaded dataset.
    """

    df = pd.read_csv(file_path, sep='\t', header=0)

    return df

def prepare_data(df):
    """
    Prepare the dataset for NLI evaluation by selecting relevant columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the loaded dataset.
    
    Returns:
        list of dictionaries containing the premise, hypothesis, and label for each example.
    """

    # Remove examples where annotators disagreed on the label
    df = df[df['gold_label'] != '-']
    df = df[:1000] # Limit to first 1,000 examples

    examples = []
    for _, row in df.iterrows():
        examples.append({
            'premise': row['sentence1'],
            'hypothesis': row['sentence2'],
            'label': row['gold_label']
        })

    return examples