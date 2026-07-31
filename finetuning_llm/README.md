# Finetuning LLMs on NLI
Repo for task 8 of Neural Explanations in LLMs - Finetuning LLMs on NLI

### Dataset
Dataset retrieved from SNLI Corpus. Dataset excluded from version control and must be downloaded separately. Download dataset from https://nlp.stanford.edu/projects/snli/snli_1.0.zip. Inside the zip folder, copy `snli_1.0/snli_1.0_test.txt`, `snli_1.0/snli_1.0_train.txt`, and `snli_1.0/snli_1.0_dev.txt` and place them inside `data/snli_1.0/`

Dataset is preprocessed in `code/dataset.py`

### Finetuning

The model is fine-tuned as a sequence classification model using Hugging Face's `AutoModelForSequenceClassification`. The model is configured with three output labels corresponding to the SNLI classes:

- Entailment
- Neutral
- Contradiction

To make fine-tuning run on limited GPU memory, the model is trained using QLoRA:

- 4-bit NF4 quantization (`BitsAndBytes`)
- LoRA adapters applied to the attention projection layers (`q_proj`, `k_proj`, `v_proj`, and `o_proj`)
- Gradient checkpointing enabled to reduce memory usage
- Mixed precision (FP16) training

The SNLI premise and hypothesis are tokenized as sentence pairs with a maximum sequence length of 256 tokens.

Training configuration:

- **Base model:** Mistral-7B-v0.1
- **Training method:** QLoRA (4-bit quantization with LoRA adapters)
- **Epochs:** 3
- **Learning rate:** 2e-4
- **Effective batch size:** 16 (batch size of 4 with gradient accumulation of 4)
- **Maximum sequence length:** 256 tokens
- **LoRA configuration:** rank = 16, alpha = 32, dropout = 0.05
- **Precision:** FP16 with gradient checkpointing enabled

The best checkpoint is selected based on the validation loss.

See [`code/train.py`](code/train.py) for the exact finetuning code

### Model
Using Mistral-7B-v0.1 with Hugging Face Transformers and `AutoModelForSequenceClassification`. See https://huggingface.co/mistralai/Mistral-7B-v0.1 for more info. 

### Results
The Mistral-7B-v0.1 achieved 88.5% accuracy on the SNLI test data for 1,000 test examples. See [results.txt](results.txt)

The finetuning script records runtime information:
- Model loading time: 46.92 seconds
- Dataset loading and tokenizing time: 36.87 seconds
- Training time: 38683.18 seconds

The evaluation script also records runtime information:
- Model loading time: 48.39 seconds
- Dataset loading time: 3.16 seconds
- Evaluation time: 112.86 seconds

These are stored in `results.txt` and printed in terminal

---

## How to Run

### Setup
```bash
# Clone the repo
git clone https://github.com/codeCrafter10-alt/open_llms_nli
cd open_llms_nli/finetuning_llm

# Install dependencies
pip install -r requirements.txt
```

### Download dataset
**See Dataset and Model**  
Download dataset from https://nlp.stanford.edu/projects/snli/snli_1.0.zip

### Finetune model
```bash
python code/train.py
```
The finetuned model configuration, LoRA adapter, and tokenizer are saved in `results/finetuned_model`

### Run and Compute Accuracy
```bash
python code/eval.py 
```
Accuracy will be printed out in terminal and saved in `results.txt`
