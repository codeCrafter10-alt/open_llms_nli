"""
Evaluate accuracy of Mistral-7B-v0.1 model on NLI task using SNLI dataset.
"""

import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from dataset import load_dataset, prepare_data

model_name = "mistralai/Mistral-7B-v0.1"
adapter_path = "./results/finetuned_model"
test_file = "data/snli_1.0/snli_1.0_test.txt"

def load_model():
    """
    Load tokenizer and model
    """
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, num_labels=3)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print(model.device)

    load_time = time.time() - start

    return tokenizer, model, load_time

def predict(model, tokenizer, premise, hypothesis):
    """
    Generate a label prediction from the LLM model
    """

    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=-1).item()

    return prediction

def main():
    """
    Evaluate the model on the SNLI dataset
    """

    print("Evaluating Model...")

    # Load dataset
    start_time = time.time()

    dataset = load_dataset(test_file)
    test_dataset = prepare_data(dataset, limit=1000)
    dataload_time = time.time() - start_time
    print(f"Dataset loading time: {dataload_time:.2f} seconds")

    # Load model
    tokenizer, model, load_time = load_model()
    print(f"Model loading time: {load_time:.2f} seconds")

    # Evaluate model
    total_examples = len(test_dataset)
    correct_predictions = 0

    eval_start_time = time.time()

    for i, example in enumerate(test_dataset):
        prediction = predict(model, tokenizer, example["premise"], example["hypothesis"])

        if prediction == example["label"]:
            correct_predictions += 1

    accuracy = correct_predictions / total_examples

    eval_time = time.time() - eval_start_time
    print(f"Evaluation time: {eval_time:.2f} seconds\n")
    print(f"Test Accuracy: {100 * accuracy:.2f}%")


    # Save results to results.txt
    with open("results.txt", "w") as f:
        f.write("Model: Mistral-7B-v0.1\n\n")
        f.write(f"Model test accuracy: {100*accuracy:.3f}%\n")
        f.write(f"Total test examples: {len(test_dataset)}\n")
        f.write(f"Correct predictions: {correct_predictions}\n\n")
        f.write(f"Model loading time: {load_time:.2f}s\n")
        f.write(f"Dataset loading time: {dataload_time:.2f}s\n")
        f.write(f"Evaluation time: {eval_time:.2f}s\n")


if __name__ == "__main__":
    main()