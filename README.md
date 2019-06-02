# GPN 19 - Talk on ML Workflow Tools

https://entropia.de/GPN19:ML_Workflow_Tools_Overview

Modern ML workflow requires to run experiments fast at a large scale. 
In order to stay sane and keep an overview of what is going on, there are some tools out there.

We will demonstrate 2 tools in this workshop : ML flow and edflow. These tools focus on different parts of the development workflow : fast model iteration and monitoring (dashboard).

We will provide some basic code and instructions on how to add the tools to basic examples. We will cover a standard classification based problem to demonstrate a simple use-case.


# Environment setup

* create new virtual environment, for example using conda

```bash
conda create --name ml_talk python=3.6
source activate ml_talk
```

* then install requirements
```bash
pip install -r mlflow_excersize/requirements.txt
pip install -r requirements_edflow.txt
```

# Slides 

* https://docs.google.com/presentation/d/10_kPq734hIEYeDczrTDHYUOobqRGwNxZC7qdbSH2VeE/edit#slide=id.g58a2b8ebdc_0_42


# Part 1 - EDFlow

https://github.com/pesser/edflow


# Solutions for Edflow Part

```bash
edflow -t problem1_solution/train.yaml # train model
edflow -t problem1_solution/train.yaml -p project_folder # continue training model


edflow -t problem2_solution/train.yaml -e problem2_solution/validation.yaml # add validation
```

# Part 2 - MLFlow
It consists only of adding some simple logging statements to `mlflow_excersize/linear_model.py` and `mlflow_excersize/linear_model_lasso.py` (compare `mlflow_excersize/linear_model_mlflow.py`) and afterwards running `mlflow ui --filestore mlflow_excersize/mlflow`.

# Part 3 - Project Structure
The template is available under: https://github.com/LeanderK/cookiecutter-ml