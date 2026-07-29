# Open LLMs on NLI
Repo for task 7 of Neural Explanations in LLMs - Open LLMs on NLI task. Comparing Bowman model accuracy to LLM, specifically the Mistral 7B-Instruct

## Bowman/LSTM Model
Original Repository: https://github.com/jayelm/compexp 

Only code required for evaluation copied and all other files and folders have been removed.

### Dataset
Dataset retrieved from SNLI Corpus. Dataset excluded from version control and must be downloaded seperately. Download dataset from https://nlp.stanford.edu/projects/snli/snli_1.0.zip. Inside the zip folder, copy `snli_1.0/snli_1.0_test.txt` and place it inside `data/snli_1.0/`

### Model
Download pretrained epoch 6 of Bowman SNLI model from http://downloads.cs.stanford.edu/nlp/data/muj/bowman_snli/6.pth and store it in `models/bowman_snli/`

### Results
Bowman Model achieved an accuracy of 79.204% on the SNLI test set. See [results.txt](results.txt)

The evaulation script also records runtime information:
- Model loading time: 0.65 seconds
- Dataset loading time: 1.21 seconds for 10,000 test examples
- Evaluation time: 8.16 seconds

These are stored in `results.txt` and printed in terminal

The evaluation script also returns a CSV file containing individual predictions for each test example. See `data/analysis/preds/6_snli_1.0_test.csv`. The file includes:
- `gt`: ground truth
- `pred`: model prediction
- `correct`: boolean value of whether the model predicted correctly

---

## How to Run

### Setup
```bash
# Clone the repo
git clone https://github.com/codeCrafter10-alt/open_llms_nli
cd open_llms_nli/bowman_model

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Download model and dataset
**See Dataset and Model**  
Download dataset from https://nlp.stanford.edu/projects/snli/snli_1.0.zip.  
Download model from http://downloads.cs.stanford.edu/nlp/data/muj/bowman_snli/6.pth

### Run and Compute Accuracy
```bash
python code/snli_eval.py 
```
Accuracy will be printed out in terminal and saved in `results.txt`
