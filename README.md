# Sepsis Treatment Optimization using Reinforcement Learning

This repository contains the codebase and final report for the Reinforcement Learning course project at **NOVA Information Management School** (Group C). 

## 🩺 Overview
The project aims to develop and evaluate Reinforcement Learning (RL) agents on the **ICU-Sepsis-v2 benchmark**, an environment built upon the real **MIMIC-III clinical database**. 

Sepsis is a life-threatening condition, and its treatment requires high-stakes, continuous decisions. The objective of our RL agents is to learn optimal policies for administering **vasopressors** and **intravenous (IV) fluids** to ICU patients. The ultimate goal is to maximize patient survival rates while minimizing unnecessary treatment intensity.

## ⚙️ Project Configurations

The environment is approached through two main configurations to explore the transition from classical tabular RL to modern Deep RL:

### 1. Configuration A: Discrete State Space
* **State Space:** 716 discrete clinical states.
* **Action Space:** 25 possible actions (combinations of vasopressor and IV fluid doses).
* **Methods:** Tabular RL methods (e.g., Q-Learning, SARSA, Monte Carlo) and model-based approaches (Policy Iteration) leveraging the fully accessible MDP model.

### 2. Configuration B: Continuous Observation Space
* **State Space:** 47-dimensional continuous feature vector containing normalized physiological measurements (e.g., SOFA score, heart rate, lactate, blood pressure).
* **Methods:** Deep Reinforcement Learning algorithms (e.g., PPO, DDPG) utilizing neural networks for function approximation since tabular methods are no longer viable.

## 🚀 Creative Extensions & Advanced Analysis
Beyond the baseline requirements, this project implements several creative extensions:
* **Clinical Reward Shaping:** Adjusting the baseline reward function to penalize excessive treatment intensity, making the policies safer and more clinically viable (`Clinical_Rewarding_Shaping_A.ipynb`, `Clinical_Rewarding_Shaping_B.ipynb`).
* **Context-Aware Ensemble Policy:** An advanced policy architecture designed to improve the robustness of treatment decisions (`Context-Aware Ensemble Policy.ipynb`).
* **SOFA-Stratified Performance Analysis:** A granular evaluation of agent performance across different patient severity levels, categorized by the Sequential Organ Failure Assessment (SOFA) score (`SOFA-Stratified Performance Analysis.ipynb`).

## 📁 Repository Structure

```text
├── RL Group C Project.ipynb                   # Main project report and implementation (Methodology, Evaluation, Conclusion)
├── Clinical_Rewarding_Shaping_A.ipynb         # Reward shaping extension for Tabular RL
├── Clinical_Rewarding_Shaping_B.ipynb         # Reward shaping extension for Deep RL
├── Context-Aware Ensemble Policy.ipynb        # Implementation of the ensemble policy extension
├── SOFA-Stratified Performance Analysis.ipynb # Severity-based evaluation
├── envs/                                      # Custom Gymnasium environments and wrappers
│   ├── continuous_sepsis_env.py               # Continuous observation wrapper for Config B
│   ├── configa_modelfree.py                   # Environment setup for Config A
│   └── wrappers.py                            # Additional Gym wrappers
├── best_params_configA/                       # Tuned hyperparameters for Config A
├── best_params_configB/                       # Tuned hyperparameters for Config B
├── models_configB/                            # Saved trained models for Deep RL
├── logs_configB/                              # TensorBoard logs for training progress
└── plots/                                     # Saved visualizations and learning curves
```

## 🛠️ Setup & Installation

To run the notebooks and reproduce the results, you will need to install the required dependencies. It is recommended to use a virtual environment.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-directory>

# 2. Install dependencies
pip install gymnasium stable-baselines3 torch numpy pandas jupyter icu-sepsis
```

Once installed, you can launch Jupyter Notebook to interact with the code:
```bash
jupyter notebook
```

## 👥 Authors
* **Group C**
* Reinforcement Learning Course - NOVA Information Management School
