terraform {
  required_providers {
    coder = {
      source  = "coder/coder"
      version = "~> 0.17"
    }
  }
}

data "coder_parameter" "age" {
  name         = "age"
  display_name = "Age"
  description  = "Patient age in years"
  type         = "number"
  default      = 50
  validation {
    min = 1
    max = 120
  }
  order = 1
}

data "coder_parameter" "resting_bp" {
  name         = "resting_bp"
  display_name = "Resting Blood Pressure (mmHg)"
  description  = "Systolic resting blood pressure"
  type         = "number"
  default      = 130
  validation {
    min = 50
    max = 250
  }
  order = 2
}

data "coder_parameter" "cholesterol" {
  name         = "cholesterol"
  display_name = "Cholesterol (mg/dL)"
  description  = "Serum cholesterol"
  type         = "number"
  default      = 200
  validation {
    min = 0
    max = 800
  }
  order = 3
}

data "coder_parameter" "fasting_bs" {
  name         = "fasting_bs"
  display_name = "Fasting Blood Sugar > 120 mg/dL?"
  description  = "1 if fasting blood sugar > 120 mg/dL, 0 otherwise"
  type         = "number"
  default      = 0
  validation {
    min = 0
    max = 1
  }
  order = 4
}

data "coder_parameter" "max_hr" {
  name         = "max_hr"
  display_name = "Max Heart Rate"
  description  = "Maximum heart rate achieved"
  type         = "number"
  default      = 150
  validation {
    min = 40
    max = 250
  }
  order = 5
}

data "coder_parameter" "oldpeak" {
  name         = "oldpeak"
  display_name = "Oldpeak"
  description  = "ST depression induced by exercise relative to rest"
  type         = "number"
  default      = 1.0
  validation {
    min = 0.0
    max = 10.0
  }
  order = 6
}

data "coder_parameter" "sex" {
  name         = "sex"
  display_name = "Sex"
  description  = "Patient sex"
  type         = "string"
  default      = "Female"
  option {
    name  = "Female"
    value = "Female"
  }
  option {
    name  = "Male"
    value = "Male"
  }
  order = 7
}

data "coder_parameter" "chest_pain_type" {
  name         = "chest_pain_type"
  display_name = "Chest Pain Type"
  description  = "Type of chest pain"
  type         = "string"
  default      = "ASY"
  option {
    name  = "ASY"
    value = "ASY"
  }
  option {
    name  = "ATA"
    value = "ATA"
  }
  option {
    name  = "NAP"
    value = "NAP"
  }
  option {
    name  = "TA"
    value = "TA"
  }
  order = 8
}

data "coder_parameter" "resting_ecg" {
  name         = "resting_ecg"
  display_name = "Resting ECG"
  description  = "Resting electrocardiogram results"
  type         = "string"
  default      = "Normal"
  option {
    name  = "LVH"
    value = "LVH"
  }
  option {
    name  = "Normal"
    value = "Normal"
  }
  option {
    name  = "ST"
    value = "ST"
  }
  order = 9
}

data "coder_parameter" "exercise_angina" {
  name         = "exercise_angina"
  display_name = "Exercise Angina"
  description  = "Exercise-induced angina"
  type         = "string"
  default      = "No"
  option {
    name  = "No"
    value = "No"
  }
  option {
    name  = "Yes"
    value = "Yes"
  }
  order = 10
}

data "coder_parameter" "st_slope" {
  name         = "st_slope"
  display_name = "ST Slope"
  description  = "Slope of the peak exercise ST segment"
  type         = "string"
  default      = "Flat"
  option {
    name  = "Down"
    value = "Down"
  }
  option {
    name  = "Flat"
    value = "Flat"
  }
  option {
    name  = "Up"
    value = "Up"
  }
  order = 11
}

data "coder_parameter" "threshold" {
  name         = "threshold"
  display_name = "Decision Threshold"
  description  = "Probability threshold for high risk classification"
  type         = "number"
  default      = 0.5
  validation {
    min = 0.05
    max = 0.95
  }
  order = 12
}

resource "coder_workspace" "harpa" {
  name        = "harpa-workspace-${data.coder_workspace_owner.me.name}"
  display_name = "Heart Attack Risk Predictor Workspace"

  template = "harpa-template"

  owner_id = data.coder_workspace_owner.me.id

  parameters = {
    age               = data.coder_parameter.age.value
    resting_bp        = data.coder_parameter.resting_bp.value
    cholesterol       = data.coder_parameter.cholesterol.value
    fasting_bs        = data.coder_parameter.fasting_bs.value
    max_hr            = data.coder_parameter.max_hr.value
    oldpeak           = data.coder_parameter.oldpeak.value
    sex               = data.coder_parameter.sex.value
    chest_pain_type   = data.coder_parameter.chest_pain_type.value
    resting_ecg       = data.coder_parameter.resting_ecg.value
    exercise_angina   = data.coder_parameter.exercise_angina.value
    st_slope          = data.coder_parameter.st_slope.value
    threshold         = data.coder_parameter.threshold.value
  }

  startup_script = <<EOF
#!/bin/bash
set -e

# Install Python and pip if not present (assuming Ubuntu/Debian base)
sudo apt-get update && sudo apt-get install -y python3 python3-pip git

# Clone the repo (or copy files; for demo, assume repo is accessible)
git clone https://github.com/cutewizzy11/Heart-Attack-Risk-Predictor.git /tmp/harpa
cd /tmp/harpa

# Install requirements
pip3 install -r requirements.txt

# Run training (creates model.joblib)
python3 train_model.py

# Run prediction with parameters as env vars
export AGE=${data.coder_parameter.age.value}
export RESTING_BP=${data.coder_parameter.resting_bp.value}
export CHOLESTEROL=${data.coder_parameter.cholesterol.value}
export FASTING_BS=${data.coder_parameter.fasting_bs.value}
export MAX_HR=${data.coder_parameter.max_hr.value}
export OLDPEAK=${data.coder_parameter.oldpeak.value}
export SEX=${data.coder_parameter.sex.value}
export CHEST_PAIN_TYPE=${data.coder_parameter.chest_pain_type.value}
export RESTING_ECG=${data.coder_parameter.resting_ecg.value}
export EXERCISE_ANGINA=${data.coder_parameter.exercise_angina.value}
export ST_SLOPE=${data.coder_parameter.st_slope.value}
export THRESHOLD=${data.coder_parameter.threshold.value}

python3 predict_from_env.py

EOF
}

data "coder_workspace_owner" "me" {}

resource "coder_agent" "main" {
  arch           = "amd64"
  os             = "linux"
  startup_script = resource.coder_workspace.harpa.startup_script
}

resource "docker_container" "workspace" {
  name  = "coder-${data.coder_workspace_owner.me.name}-${lower(data.coder_workspace_owner.me.name)}"
  image = "codercom/enterprise-base:ubuntu"

  command = ["sh", "-c", coder_agent.main.init_script]
  env     = ["CODER_AGENT_TOKEN=${coder_agent.main.token}"]

  host {
    host = "host.docker.internal"
    ip   = "host-gateway"
  }
}
