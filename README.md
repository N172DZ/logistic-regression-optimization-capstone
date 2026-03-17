# Comparison of Optimization Strategies for Logistic Regression

## Overview
This project compares three optimization strategies for Logistic Regression:
- Batch Gradient Descent (BGD)
- Stochastic Gradient Descent (SGD)
- Mini-batch Gradient Descent (MBGD)

The experiments use the Iris dataset and evaluate convergence behavior, classification accuracy, training stability, and computational cost. This directly follows the capstone proposal. 

## Project Goals
- Compare convergence speed across optimization methods
- Compare classification accuracy
- Analyze training stability using loss behavior
- Measure computational cost using training time
- Study the effect of learning rate selection

## Files
- `data_loader.py`: loads and standardizes Iris data
- `model.py`: logistic regression from scratch using softmax
- `train.py`: training loops for BGD, SGD, and MBGD
- `evaluation.py`: result summary functions
- `plots.py`: convergence and accuracy plots
- `main.py`: runs the full experiment

## Environment Setup
```bash
python3.9 -m venv capstone-env
source capstone-env/bin/activate
```

## How to Run
```bash
pip install -r requirements.txt
python main.py
```