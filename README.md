# Heart Attack Risk Prediction - Coder Interactive Interface

Terraform template using Coder's `coder_parameter` to build an interactive workspace creation form for heart attack risk prediction, sharing results visually in provisioned cloud environments.

## Coder Workspace Template

This repo provides a Terraform template (`main.tf`) that uses Coder's `coder_parameter` to create an interactive UI for heart attack risk prediction. The template provisions a workspace that runs the prediction based on user inputs and displays the result, fulfilling the hackathon requirement for incorporating Coder's product suite as a visual way to share research results.

## Usage

1. Deploy `main.tf` in your Coder instance.
2. Users create workspaces by filling the interactive form with patient data (age, blood pressure, etc.).
3. The workspace provisions and runs the prediction, outputting the risk probability and classification.

## Setup Requirements

- Coder instance with Terraform provider support.
- The template clones this repo and installs dependencies to train the model and run predictions.

## Notes

- `model.joblib` is intentionally gitignored. The workspace auto-trains the model on provision.
- Data: `Datasets/Heart Attack/heart_processed.csv`
- Training script: `train_model.py`
- Prediction script: `predict_from_env.py`
