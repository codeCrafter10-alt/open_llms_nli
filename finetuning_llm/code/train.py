"""
Fine-tuning a language model for sequence classification using LoRA.
"""

import torch
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType

from dataset import load_dataset, prepare_data

model_name = "mistralai/Mistral-7B-v0.2"
train_file = "data/snli_1.0/snli_1.0_train.txt"
dev_file = "data/snli_1.0/snli_1.0_dev.txt"

def tokenize_data(dataset, tokenizer):
    """
    Tokenizes the dataset using the provided tokenizer.
    """

    def tokenize(batch):
        return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, max_length=256)

    return dataset.map(tokenize, batched=True)


def main():
    print("Finetuning Model...")

    dataset_start = time.time()
    train_df = load_dataset(train_file)
    dev_df = load_dataset(dev_file)
    train_dataset = prepare_data(train_df, limit=100000)
    dev_dataset = prepare_data(dev_df, limit=5000)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  

    train_dataset = tokenize_data(train_dataset, tokenizer)
    dev_dataset = tokenize_data(dev_dataset, tokenizer)

    dataset_time = time.time() - dataset_start
    print(f"Dataset loaded and tokenized in {dataset_time:.2f} seconds.")


    model_start = time.time()
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, quantization_config=bnb_config, device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False 

    model_time = time.time() - model_start
    print(f"Model loaded in {model_time:.2f} seconds.")

    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # Training
    training_args = TrainingArguments(
        output_dir="results/finetuned_model",
        num_train_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        fp16=True,
        logging_steps=50,
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=False,
        metric_for_best_model="eval_loss",
        disable_tqdm=False,
        report_to="none",
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    train_start = time.time()
    print("Starting training...")
    print(model.device)

    trainer.train()
    train_time = time.time() - train_start
    print(f"Training completed in {train_time:.2f} seconds.\n")

    model.save_pretrained("./results/finetuned_model")
    tokenizer.save_pretrained("./results/finetuned_model")

    print("\n\n\n")

if __name__ == "__main__":
    main()