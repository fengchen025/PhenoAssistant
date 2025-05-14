# PhenoAssistant

This is the official code repository for our paper [*"PhenoAssistant: A Conversational Multi-Agent AI System for Automated Plant Phenotyping"*](https://arxiv.org/abs/2504.19818).

---

## Chat logs

- **Case studies' chat logs**  
  Chat logs for the case studies presented in our manuscript are available in:  
  - [case1.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/main/case1.ipynb)  
  - [case2.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/main/case2.ipynb)  
  - [case3.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/main/case3.ipynb)  

- **Evaluation's chat logs**  
  Chat logs and results for the evaluations presented in our manuscript are available in:  
  - [eval_tool_selection.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/main/eval_tool_selection.ipynb)  
  - [eval_vision_model_selection.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/main/eval_vision_model_selection.ipynb)  
  - [eval_data_analysis.ipynb](https://github.com/fengchen025/PhenoAssistant/blob/main/eval_data_analysis.ipynb)
    
---

## Key components of PhenoAssistant

- Implementation of agents is available at [agents.py](https://github.com/fengchen025/PhenoAssistant/blob/main/agents.py)
- Implementation of tools is available at [functions](https://github.com/fengchen025/PhenoAssistant/tree/main/functions)

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
