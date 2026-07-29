"""
Evaluate accuracy of Mistral-7B-Instruct-v0.2 model on NLI task using SNLI dataset.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import load_dataset, prepare_data
from prompt import create_prompt

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

def load_model():
    """
    Load tokenizer and model
    """
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    model.eval()
    print(model.device)

    load_time = time.time() - start

    return tokenizer, model, load_time

def predict(model, tokenizer, prompt):
    """
    Generate a label prediction from the LLM model
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1, do_sample=False)


    outputs = outputs[0][inputs.input_ids.shape[-1]:] 
    response = tokenizer.decode(outputs, skip_special_tokens=True).strip().lower()

    return response

def main():
    """
    Evaluate the model on the SNLI dataset
    """

    # Load dataset
    start_time = time.time()

    dataset = load_dataset("data/snli_1.0/snli_1.0_test.txt")
    data = prepare_data(dataset)
    dataload_time = time.time() - start_time
    print(f"Dataset loading time: {dataload_time:.2f} seconds")

    # Load model
    tokenizer, model, load_time = load_model()
    print(f"Model loading time: {load_time:.2f} seconds")

    # Evaluate model
    total_examples = len(data)
    correct_predictions = 0

    eval_start_time = time.time()

    for example in data:
        premise = example['premise']
        hypothesis = example['hypothesis']
        true_label = example['label'].strip().lower()

        prompt = create_prompt(premise, hypothesis)
        predicted_label = predict(model, tokenizer, prompt)

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_examples

    eval_time = time.time() - eval_start_time
    print(f"Evaluation time: {eval_time:.2f} seconds\n")
    print(f"Test Accuracy: {100 * accuracy:.2f}%")


    # Save results to results.txt
    with open("results.txt", "w") as f:
        f.write("Model: Mistral-7B-Instruct-v0.2\n\n")
        f.write(f"Model test accuracy: {100*accuracy:.3f}%\n")
        f.write(f"Total test examples: {len(data)}\n")
        f.write(f"Correct predictions: {correct_predictions}\n\n")
        f.write(f"Model loading time: {load_time:.2f}s\n")
        f.write(f"Dataset loading time: {dataload_time:.2f}s\n")
        f.write(f"Evaluation time: {eval_time:.2f}s\n")


if __name__ == "__main__":
    main()