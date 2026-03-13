# Homeowners Risk Scoring – GLM + GAM Underwriting Intelligence

## Overview

This project implements a **three-tier underwriting intelligence system** for **U.S. homeowners insurance risk scoring** using a hybrid **GLM + GAM architecture**.

The system simulates how modern insurers enhance traditional actuarial pricing with **explainable machine learning**.

The model predicts:

* Risk Score (0–100)
* Expected Loss
* Risk Segment
* Indicative Premium
* Underwriting Flags
* Key Risk Drivers

The application is deployed as an **interactive Streamlit underwriting dashboard**.

---

# Insurance Problem

Traditional homeowners pricing relies heavily on:

* ISO rating plans
* Generalized Linear Models (GLMs)

These models are **transparent and regulator-friendly**, but they often leave **significant unexplained variance**.

This project introduces an **Explainable ML intelligence layer** that captures:

* nonlinear risk relationships
* feature interactions
* geographical hazard effects

without sacrificing interpretability.

---

# Model Architecture

The system follows a **three-tier actuarial modeling framework**.

```
Final Prediction

η_final(x) = η_GLM(x) + δ_GAM(x)
```

Where

```
η_GLM(x)  = GLM prediction (Tier 1 + Tier 2)
δ_GAM(x)  = GAM nonlinear correction (Tier 3)
```

---

# Tier Structure

## Tier 1 – Structural Foundation (GLM)

Core property characteristics.

Features include:

* roof age
* roof material
* home age
* construction type
* prior water claim
* replacement cost ratio
* coverage amount
* location zone

These represent **traditional underwriting variables**.

---

## Tier 2 – Lifestyle & Behavioral Risk (GLM)

Adds behavioral risk factors.

Examples:

* prior claims history
* insurance lapses
* crime index
* pool / trampoline
* home business
* hydrant distance
* building code compliance
* monitored alarm
* gated community
* recent renovation

---

## Tier 3 – Geographical + GAM Residual Layer

Captures nonlinear environmental hazards.

Examples:

* wildfire score
* canopy density
* flood depth
* slope steepness
* hail zone
* burn history
* foundation type

Includes interaction effects such as:

* Roof Age × Wildfire Zone
* Water Claim × Tree Canopy
* Flood Zone × Foundation Type
* Slope × Burn History
* Roof Age × Hail Zone

---

# Project Structure

```
Homeowners_Risk_Scoring_Platform
│
├── app.py
├── generate_data.py
├── models.py
├── scoring.py
│
├── data
│   ├── raw
│   │   └── homeowners_portfolio.csv
│   │
│   ├── processed
│   │   └── features_processed.csv
│   │
│   └── models
│       ├── glm_t1.pkl
│       ├── glm_full.pkl
│       ├── gam_residual.pkl
│       └── model_metrics.json
│
├── requirements.txt
└── README.md
```

---

# Key Files

### app.py

Streamlit underwriting dashboard.

Features:

* interactive risk scoring form
* GLM + GAM model visualization
* risk score gauge
* waterfall contribution chart
* portfolio analytics dashboard
* underwriting decision engine

The application loads a portfolio dataset and scores policies in real time. 

---

### scoring.py

Core scoring engine.

Contains the full actuarial computation pipeline including:

* GLM score calculation
* GAM nonlinear adjustments
* interaction detection
* expected loss calibration
* premium calculation
* underwriting flags

This file contains the function:

```
score_policy()
```

which produces the final risk output. 

---

### models.py

Model training pipeline.

Implements:

* Tier 1 GLM
* Tier 2 GLM
* Tier 3 GAM residual model

Models are trained on processed features and saved to disk.

```
η_final = η_GLM + δ_GAM
```

Training output:

* glm_t1.pkl
* glm_full.pkl
* gam_residual.pkl
* model_metrics.json 

---

### generate_data.py

Creates a **synthetic homeowners portfolio dataset** used for training and demo.

Outputs:

```
data/raw/homeowners_portfolio.csv
data/processed/features_processed.csv
```

---

# Risk Score Interpretation

| Score  | Segment   | Underwriting Action |
| ------ | --------- | ------------------- |
| 0–30   | Preferred | Auto-bind eligible  |
| 31–60  | Standard  | Bind with review    |
| 61–80  | Rated     | Apply surcharge     |
| 81–100 | Decline   | Refer to E&S market |

---

# Expected Loss Calculation

Expected loss is derived from the final model output:

```
Loss = exp(αη + β)
```

where

```
η = final log-risk score
α = calibration slope
β = calibration intercept
```

Loss is constrained between:

```
$800 – $20,000
```

to reflect realistic homeowners insurance losses.

---

# Indicative Premium

Premium is derived using a target loss ratio.

```
Premium = Expected Loss / Target Loss Ratio
```

Typical loss ratio assumptions:

```
Preferred → 68%
Standard → 65%
Rated → 60%
```

---

# Dashboard Features

The Streamlit application provides:

### Risk Contribution Waterfall

Shows how each tier contributes to the final score.

```
Base → Tier1 → Tier2 → Tier3 → Final Score
```

### Feature Contribution Chart

Displays the largest positive and negative risk drivers.

### Underwriting Flags

Highlights:

* inspection triggers
* fraud indicators
* catastrophe exposures
* interaction risks

### Portfolio Analytics

Displays statistics across the entire portfolio dataset.

---

# Installation

Clone the repository.

```
git clone <repository-url>
cd Final_Risk_Scoring
```

Install dependencies.

```
pip install -r requirements.txt
```

---

# Running the Application

Start the Streamlit dashboard.

```
streamlit run app.py
```

The application will launch at:

```
http://localhost:8501
```

---

# Example Workflow

1. Generate synthetic data

```
python generate_data.py
```

2. Train models

```
python models.py
```

3. Launch dashboard

```
streamlit run app.py
```

4. Enter property attributes

5. View risk score and underwriting decision

---

# Example Output

```
Risk Score: 72
Segment: Rated

Expected Loss: $4,280
Indicative Premium: $7,100
```

Top Drivers:

```
Roof Age
Wildfire Exposure
Prior Claims
Slope Risk
```

---

# Business Value

This project demonstrates how insurers can augment traditional pricing with explainable machine learning.

Benefits include:

* improved risk segmentation
* more accurate pricing
* interpretable AI models
* underwriter decision support

The approach remains **regulator-friendly** because GLM remains the core pricing model while GAM adds transparent nonlinear adjustments.

---

# Tech Stack

Python

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Plotly

---

# Author

Tamanna Vaikkath
