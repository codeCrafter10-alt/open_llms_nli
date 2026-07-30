# Open LLMs on NLI
Repo for task 7 of Neural Explanations in LLMs - Open LLMs on NLI task. Comparing Bowman model accuracy to LLM, specifically the Mistral 7B-Instruct

## LLM Model
Computing accuracy of LLM model on NLI task using the SNLI corpus and prompting strategies.

### Dataset
Dataset retrieved from SNLI Corpus. Dataset excluded from version control and must be downloaded seperately. Download dataset from https://nlp.stanford.edu/projects/snli/snli_1.0.zip. Inside the zip folder, copy `snli_1.0/snli_1.0_test.txt` and place it inside `data/snli_1.0/`

Dataset is preprocessed in `code/dataset.py`

### Prompting Strategy
- Instruction-based prompting: Added clear and stronger instructions for the model
- Few-shot prompting: Added three examples related to NLI task in prompt
- Constrained Output: Prompted model to output exactly one word as the output
- Hidden Reasoning: Made model reason through its logic but still output only the final answer

See [`code/prompt.py`](code/prompt.py) for the exact prompt

### Model
Using Mistral-7B-Instruct-v0.2. See https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 for more info.

### Results
The Mistral-7B-Instruct-v0.2 achieved an accuracy of 68.8% on the SNLI test data for 1,000 test examples. See [results.txt](results.txt)

The evaulation script also records runtime information:
- Model loading time: 44.25 seconds
- Dataset loading time: 0.22 seconds for 1,000 test examples
- Evaluation time: 7705.49 seconds

These are stored in `results.txt` and printed in terminal

---

## How to Run

### Setup
```bash
# Clone the repo
git clone https://github.com/codeCrafter10-alt/open_llms_nli
cd open_llms_nli/llm_model

# Install dependencies
pip install -r requirements.txt
```

### Download dataset
**See Dataset and Model**  
Download dataset from https://nlp.stanford.edu/projects/snli/snli_1.0.zip

### Run and Compute Accuracy
```bash
python code/eval.py 
```
Accuracy will be printed out in terminal and saved in `results.txt`
