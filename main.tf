###############################################################################
# HARPA – Heart Attack Risk Prediction App
# Coder Workspace Template (Terraform)
#
# This template creates an interactive workspace creation form using
# coder_parameter.  When a developer provisions a workspace from this
# template they fill in patient clinical values via the Coder dashboard;
# the workspace then trains the model, runs the prediction, AND launches
# the full Streamlit web app so the user can keep experimenting.
###############################################################################

terraform {
  required_providers {
    coder = {
      source  = "coder/coder"
      version = "~> 0.17"
    }
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

# ── Data sources ─────────────────────────────────────────────────────────────
data "coder_workspace" "me" {}
data "coder_workspace_owner" "me" {}

# ── Interactive parameters (rendered as a form in the Coder dashboard) ───────

data "coder_parameter" "age" {
  name         = "age"
  display_name = "Age"
  description  = "Patient age in years"
  type         = "number"
  default      = 50
  mutable      = true
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
  mutable      = true
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
  mutable      = true
  validation {
    min = 0
    max = 800
  }
  order = 3
}

data "coder_parameter" "fasting_bs" {
  name         = "fasting_bs"
  display_name = "Fasting Blood Sugar > 120 mg/dL?"
  description  = "Select Yes (1) or No (0)"
  type         = "string"
  default      = "0"
  mutable      = true
  option {
    name  = "No"
    value = "0"
  }
  option {
    name  = "Yes"
    value = "1"
  }
  order = 4
}

data "coder_parameter" "max_hr" {
  name         = "max_hr"
  display_name = "Max Heart Rate"
  description  = "Maximum heart rate achieved"
  type         = "number"
  default      = 150
  mutable      = true
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
  mutable      = true
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
  mutable      = true
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
  description  = "Type of chest pain experienced"
  type         = "string"
  default      = "ASY"
  mutable      = true
  option {
    name  = "ASY (Asymptomatic)"
    value = "ASY"
  }
  option {
    name  = "ATA (Atypical Angina)"
    value = "ATA"
  }
  option {
    name  = "NAP (Non-Anginal Pain)"
    value = "NAP"
  }
  option {
    name  = "TA (Typical Angina)"
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
  mutable      = true
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
  mutable      = true
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
  mutable      = true
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
  description  = "Probability threshold for high-risk classification (0.05–0.95)"
  type         = "number"
  default      = 0.5
  mutable      = true
  validation {
    min = 0.05
    max = 0.95
  }
  order = 12
}

# ── Agent (runs inside the container) ────────────────────────────────────────

resource "coder_agent" "main" {
  arch = "amd64"
  os   = "linux"

  startup_script = <<-EOF
    #!/bin/bash
    set -e

    # ── 1. System deps ───────────────────────────────────────────────
    sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-pip git > /dev/null

    # ── 2. Clone repo ────────────────────────────────────────────────
    git clone --depth 1 https://github.com/cutewizzy11/Heart-Attack-Risk-Predictor.git /home/coder/harpa
    cd /home/coder/harpa

    # ── 3. Install Python deps ───────────────────────────────────────
    pip3 install --quiet -r requirements.txt

    # ── 4. Train model (creates model.joblib) ────────────────────────
    python3 train_model.py

    # ── 5. Run one-shot prediction from workspace parameters ─────────
    export AGE="${data.coder_parameter.age.value}"
    export RESTING_BP="${data.coder_parameter.resting_bp.value}"
    export CHOLESTEROL="${data.coder_parameter.cholesterol.value}"
    export FASTING_BS="${data.coder_parameter.fasting_bs.value}"
    export MAX_HR="${data.coder_parameter.max_hr.value}"
    export OLDPEAK="${data.coder_parameter.oldpeak.value}"
    export SEX="${data.coder_parameter.sex.value}"
    export CHEST_PAIN_TYPE="${data.coder_parameter.chest_pain_type.value}"
    export RESTING_ECG="${data.coder_parameter.resting_ecg.value}"
    export EXERCISE_ANGINA="${data.coder_parameter.exercise_angina.value}"
    export ST_SLOPE="${data.coder_parameter.st_slope.value}"
    export THRESHOLD="${data.coder_parameter.threshold.value}"

    python3 predict_from_env.py | tee /home/coder/harpa/prediction_result.txt

    # ── 6. Launch Streamlit app for interactive use ──────────────────
    nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
  EOF
}

# Expose Streamlit inside the Coder dashboard
resource "coder_app" "streamlit" {
  agent_id     = coder_agent.main.id
  slug         = "streamlit"
  display_name = "HARPA Streamlit App"
  icon         = "/icon/database.svg"
  url          = "http://localhost:8501"
  subdomain    = true

  healthcheck {
    url       = "http://localhost:8501/_stcore/health"
    interval  = 5
    threshold = 10
  }
}

# ── Docker infrastructure ────────────────────────────────────────────────────

resource "docker_image" "workspace" {
  name = "codercom/enterprise-base:ubuntu"
}

resource "docker_container" "workspace" {
  count = data.coder_workspace.me.start_count
  name  = "coder-${data.coder_workspace_owner.me.name}-harpa"
  image = docker_image.workspace.image_id

  command = ["sh", "-c", coder_agent.main.init_script]

  env = [
    "CODER_AGENT_TOKEN=${coder_agent.main.token}",
  ]

  host {
    host = "host.docker.internal"
    ip   = "host-gateway"
  }
}
