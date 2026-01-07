# ICU Patient Care Analysis
## Overview
Regional Medical Center (RMC) is a 400-bed acadmeic medical center serving a diverse metropolitan population. Its 36-bed Intensive Care Unit (ICU) treates critically ill patients with complex and resource-intensive needs. Hopsital leadership and the ICU Directory seek to better understand whether patients characteristics, treatment patterns, and care processes influence:
* ICU length of stay (LOS)
* Resource Utilization
* Patient recovery and mortality

Using MIMIC-IV ICU and hospital data, this project applies exploratory data analysis, rigorous data cleaning, and predicitve modeling to identify key drivers of outcomes and inform operational and clincial decision-making.

# Data Sources
All data are derived from the MIMIC-IV database.

# Directory Structure
* Data
  * hosp/: Hospital-level datasets extracted from MIMIC-IV.
  * icu/: ICU-specific datasets extrated from MIMIC-IV (primary datasets: icustays.csv).
  * Final Data/: Cleaned and merged datasets exported from the data cleaning pipeline, including/train/validation/test splits.
  * Clean example/: Intermediate datasets used to demonstrate and document the data cleaning process.

# Additional Documents
ICU_AGG_DS : Aggregated ICU dataset used to view ICU-related variables in a single table.
Data sets overview: Text file describing ICU variables and their definitions.

# Jupyter Notebook
EDA.ipynb: Exploratory data analysis (univariate and bivariate statistics)
Data_Cleaning.ipynb: Data cleaning, transformation, and feature engineering
Data_Modeling.ipynb: Regression and classification modeling


Requirements.txt:
list of libraries required to run my analysis

README.md # Project documentation

## Setup and depedenencies
1:Clone this repositiory
2: Create a virtual environment
3: Install dependencies
pip install -r requirements.txt

## How to run the Analysis
1: Open the jupyter notebook
2: Load the datasets from the hosp/ and icu folders
3: run all cells to reproduce data analysis.

# Data Engineering and Cleaning Approach
* Hospital and ICU datasets follow a star schema and are merged using appropriate keys (e.g., icd_code, icd_version, subject_id)
* Datetime fields (e.g., charttime) are converted from object types to proper datetime formats
* Non-analytic identifiers (e.g., caregiver_id) are removed to maintain a focus on departmental and system-level patterns rather than individual performance
* Categorical values are standardized (e.g., lowercasing and trimming care unit names)
* Missingness is assessed and handled prior to modeling

# ICD Code Grouping
ICD diagnosis and procedure codes are mapped into clinically meaningful categories to improve interpretability. This allows diagnoses and procedures to be analyzed at a system level rather than as sparse individual codes.

# Feature Engineering and aggregation
* High-dimensional event tables are reduced through aggregation at the patient level
* Mediaction and order information are grouped into unique lists per patients
* Cleaned dataset are merged into a single analytic table

# Dataset splits
Data are stratified by mortality and split into:
* 60% training
* 20% Validation
* 20% Test
These datasets are exported to the Final Data/ directory for reproducibility.

# Modeling Strategy
Regression (Length of stay)
Target:
* log_los (Log-transformed length of stay)
Models:
* Ordinary Least Squares (OLS)
* Lasso Regression
* Ridge Regression
* Random Forest Regressor

# Classification (Mortality)
Target:
* death_flag
Models:
* Logistic Regression (With Class weighting)
* Random Forest Classifier
Models performance is evaluated using:
* R² and residual plots (regression)
* ROC-AUC, Confusion matrices, and odds ratios (classification)

# Final Results and insights
Key finidings from the analysis include:
* Number of medications administered is a statistically significant predictor of both ICU length of stay and mortality
* Age is a significant predictor of mortality but not length of stay
These results suggest that treatment complexit and medication burden play a critical role in patients outcomes.

# Operational Recommendations
Based on the finidings, two practical interventions are proposed:
# Clinical Prescription Recommendation System
A decision-support tool that:
* Accepts patient conditions and physcians-prescribed medications
* Recommends alternative or complementary prescriptions with comparable or improved outcomes.
* Flags drug compatability and overlapping therapeutic effects.
Potential benefits:
* Reduced length of stay through optimized concurrent treatments
* Lower medication burden without compromising care quality
* Improved prescribing efficiency
# Mortality Risk Idenfitication System
A color-coded ICU wristband system indicating mortality risk:
* Red: High risk (Older Patients, high comorbidity burden)
* Blue : Lower risk
This system enables rapid visual identification of patients who may require increased monitoring or intervention.

# Conclusion
By combining robust data engineering with interpretable machine learning models, this project demonstrates how ICU data can be leveraged to improve patient outcomes while maintaining cost efficiency. The proposed solutions are scalable, clinically actionable, and aligned with real-world hospital workflows.
