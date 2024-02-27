# LLM Evaluation

## Introduction
This is an attempt at creating a scalable interface to evaluate any large language model on any dataset. The library currently supports evaluation of Gemini Pro on HumanEval.

The library includes a wrapper for large language models from different providers in `models.py` and a wrapper for different evaluations/benchmarks in `evaluation.py`.

## Results
Predictions and evaluation results for Gemini Pro on HumanEval are available in `results/`. Gemini Pro obtains an overall `pass@1 = 54.268`.


## Instructions
### Setup
Create a new conda environment and install required libraries:
```
conda create -n llm-eval python=3.10
conda activate llm-eval
pip install -r requirements.txt
```

### Evaluation
Use the following command for evaluation:
```
python main.py --model {model_name} --dataset {dataset_name} --key {api_key} --data_path {data_path} --out_path {output_path} --n {number_samples}
```

Example command for evaluating Gemini Pro on HumanEval:
```
python main.py --model gemini-pro --dataset humaneval --key {api_key}
```