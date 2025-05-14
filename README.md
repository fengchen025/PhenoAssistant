# PhenoAssistant

This is the official code repository for our paper [*"PhenoAssistant: A Conversational Multi-Agent AI System for Automated Plant Phenotyping"*](https://arxiv.org/abs/2504.19818).

---

## Chat logs

- **Case studies' chat logs**  
  Chat logs for the case studies presented in our manuscript are available in:  
  - `[case1.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/main/case1.ipynb)`  
  - `case2.ipynb`  
  - `case3.ipynb`

- **Evaluation's chat logs**  
  Chat logs and results for the evaluations presented in our manuscript are available in:  
  - `eval_tool_selection.ipynb`  
  - `eval_vision_model_selection.ipynb`  
  - `eval_data_analysis.ipynb`
    
---

## Environment setup and demo

To play with a demo, follow these steps:

1. Clone the repository:  
   `git clone https://github.com/fengchen025/PhenoAssistant.git`

2. Navigate into the project directory:  
   `cd PhenoAssistant`

3. Create the conda environment (this may take ~15 minutes):  
   `conda env create -f environment.yml`

4. Activate the environment:  
   `conda activate phenoassistant`

5. - **Demo**  
  A demonstration is available in `demo.ipynb`.  
  **Note**: Running the demo requires a GPU and valid Azure or OpenAI API keys.

Implementation of agents and tools are available at agents.py and ./functions, respectively.

Chat logs of case studies are available at case1.ipynb, case2.ipynb, case3.ipynb.

Chat logs and results of evaluations are available at eval_tool_selection.ipynb, eval_vision_model_selection.ipynb, eval_data_analysis.ipynb.
