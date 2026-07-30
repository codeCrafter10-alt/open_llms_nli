# Open LLMs on NLI
Repo for task 7 and 8 of Neural Explanations in LLMs - Open LLMs on NLI task and Finetuning LLMs on NLI. Comparing Bowman model accuracy to LLM, specifically the Mistral LLM Model

# Task 7: Open LLMs on NLI Task

**See `/bowman_model/` and `/llm_model/` for their individual readme.**

## Results

**SNLI Bowman Model**: Model achieved accuracy of 79.204% over 10,000 test examples

**Mistral-7B-Instruct-v0.2 LLM Model**: Model achieved accuracy of 68.8% over 1,000 test examples

- Bowman model performed better than the Mistral LLM model with a accuracy that was more than 10% greater than that of the LLM model.
- Bowman model took ~10 seconds for the entire evaluation (including loading model and data)
- Mistral LLM model took ~2 hours for the entire evaluation (including loading model and data)

## Challenges and Solutions
- **Getting test dataset for the NLI task**
    - Found the dataset from the official SNLI corpus
- **Calculating accuracy of the Bowman model using the compexp eval script**
    - Figured out how the script was calculating accuracy, including how and where it was reading the data from, how it was calculating accuracy, and where it was storing the results. Also, I reverse engineered how to run the script with the settings I want, using the dataset to compute accuracy and not custom data through the command line.
- **Copying the relevant files from the compexp repo into the new repo**
    - The eval Python script required other files from the repo. So, I kept running the file, looking at the error message for what file it needed, added that file in its designated location, and kept repeating this process until the eval script ran successfully.
- **Creating a good prompt for the LLM model using prompting strategies**
    - Made use of few-shot prompting and hidden reasoning.
- **Making sure the LLM model only outputs the final answer and no additional text**
    - Set do_sample to False to make model always pick highest probability token, added constrained outputting in the prompt, and processed the final output from the model
- **Processing the SNLI test data for the LLM model so it only retrieves the non-parsed premise and hypothesis**
    - Made use of pandas and converted the text file into a panda dataframe seperated by tabs. Then, I retrieved the columns under sentence1 and sentence2, along with the label
- **Evaluating the Mistral model on 10,000 examples would take too much time on CPU only**
    - Used Nautilus for this task instead of my own workstation and limited the dataset for LLM eval to 1,000 test examples.
