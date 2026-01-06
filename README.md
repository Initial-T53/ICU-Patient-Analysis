# ICU Patient Care Analysis
## Overview
Regional Medical Center (RMC) is a 400-bed acadmeic medical center serving a diverse metropolitan population. The hospital has 36 intensive care unit patients (ICU) that treat critically ill patients. The ICU Director and Hopital administration would lke to know if there are patient characteristics, treatment patterns, and care process that influnence length of stay (LOS) , resource utilization and patient recovery.

# Files
Data: Contains the data used for exploratory analysis
hosp: inside this file contains all the hospital data sets extracted from the MIMIC IV Website.
icu: Inside this file contains all the ICU patients data extracted from the MIMIC IV website. The key data set used in this analysis is in this folder. icustays.csv contains information about the patients ICU visit.
Final Data: Inside this file contains datasets that are exported from the Dataclean notebook. This data is the clean ICU & Hospital data merged together and also contains the test/train/validation data.
Clean example: inside this file contains datasets that helped walk through the data cleaning process

# Additional Documents
ICU_AGG_DS : ICU data aggregated, used to view the ICU data all on one file to look at all the different ICU data
Data sets overview: An txt file breaking down ICU variables

# Jupyter Notebook
EDA: Jupyternotebook of exploratory data analysis
Data Cleaning: Jupyternotebook of data cleaning process
Data modeling: Jupyternotebook of data modeling process

Requirements.txt:
list of libraries required to run my analysis

README.md # Project documentation

## Setup and depedenencies
Clone this repositiory
Create a virtual environment
Install dependencies
pip install -r requirements.txt

## How to run
Open the jupyter notebook
Load the datasets from the hosp/ and icu folders
run all cells to reproduce data analysis.

The notebook flow is Data discover -> Univariate statistics -> Bivariate statistics.


# Code explaination 
# Merges datasets. Since both of the datasets, hosp and icu are broken up into star schema. Inner joins on the same variable in this case (icd_code and icd_version) is required for data analysis.
# Merging the datasets together help connect variables so they can be used in data analysis.
mh_procedure=procedure.merge(procedure_code,on=['icd_code', 'icd_version'], how='inner')

# The code block below changes a variable in the dataframe to a datetime. 
# This is seen through out as there are variables like charttime, that are objects when they should be date time.
chart_events['charttime']=pd.to_datetime(chart_events['charttime'])

# Code block below drops the column caregiver_id. Caregiver ID was dropped because it was a unique identifier for caregiver. Based on the scope, the goal is to look at departmental issues. Caregiver change the scope to individual performance issues.
chart_events=chart_events.drop(['caregiver_id'], axis=1)

# Code block changes the name of columns.
date_time_events = date_time_events.rename(columns={'charttime':'chart_time',
                            'storetime':'store_time'})

# Changes data of death into a categorical variable. This makes it so the patient is classified as as alive or dead. 
patients['death_status'] = patients['date_of_death'].apply(lambda x: 'Dead' if pd.notna(x) else 'Alive')

# ICD codes for both procedures and Diagonises don't provide enough context. you can merge the datasets but you only get the long_title which is a description of the the diagnosis/procedure.
# This code block uses the ICD code to create diagnoses and procedures together.
def icd_group(icd):
    code = str(icd)
    
    if code.startswith('V'):
        return 'Supplemental/V-code'
    
    try:
        prefix = int(code[:3])
    except ValueError:
        return 'Unknown'
    
    if 1 <= prefix <= 139:
        return 'Infectious Diseases'
    elif 140 <= prefix <= 239:
        return 'Neoplasms'
    elif 240 <= prefix <= 279:
        return 'Endocrine/Metabolic'
    elif 280 <= prefix <= 289:
        return 'Blood Disorders'
    elif 290 <= prefix <= 319:
        return 'Mental Disorders'
    elif 320 <= prefix <= 389:
        return 'Neurological'
    elif 390 <= prefix <= 459:
        return 'Cardiovascular'
    elif 460 <= prefix <= 519:
        return 'Respiratory'
    elif 520 <= prefix <= 579:
        return 'Digestive'
    elif 580 <= prefix <= 629:
        return 'Genitourinary'
    elif 630 <= prefix <= 679:
        return 'Pregnancy/Perinatal'
    elif 680 <= prefix <= 709:
        return 'Skin/Subcutaneous'
    elif 710 <= prefix <= 739:
        return 'Musculoskeletal'
    elif 740 <= prefix <= 759:
        return 'Congenital Anomalies'
    elif 760 <= prefix <= 779:
        return 'Perinatal Conditions'
    elif 780 <= prefix <= 799:
        return 'Symptoms/Signs'
    elif 800 <= prefix <= 999:
        return 'Injury/Poisoning'
    else:
        return 'Other'

# Procedure
def icd_proc_group(icd):
    try:
        code = int(str(icd))
    except ValueError:
        return 'Unknown'

    if 0 <= code <= 399:
        return 'Miscellaneous Procedures'
    elif 400 <= code <= 499:
        return 'Nervous System Procedures'
    elif 500 <= code <= 599:
        return 'Respiratory System Procedures'
    elif 600 <= code <= 699:
        return 'Cardiovascular Procedures'
    elif 700 <= code <= 799:
        return 'Digestive System Procedures'
    elif 800 <= code <= 899:
        return 'Genitourinary Procedures'
    elif 900 <= code <= 999:
        return 'Musculoskeletal Procedures'
    elif 1000 <= code <= 1099:
        return 'Integumentary Procedures'
    elif 1100 <= code <= 1199:
        return 'Endocrine Procedures'
    elif 1200 <= code <= 1299:
        return 'Eye Procedures'
    elif 1300 <= code <= 1399:
        return 'Ear Procedures'
    elif 1400 <= code <= 1499:
        return 'Obstetric Procedures'
    elif 1500 <= code <= 1599:
        return 'Reproductive Procedures'
    elif 1600 <= code <= 1699:
        return 'Other Therapeutic Procedures'
    elif 1700 <= code <= 1799:
        return 'Diagnostic Imaging'
    elif 1800 <= code <= 1999:
        return 'Other Diagnostic Procedures'
    elif 2000 <= code <= 2999:
        return 'Special Procedures and Devices'
    elif 3000 <= code <= 3999:
        return 'Respiratory and Circulatory Support'
    else:
        return 'Other'

The data cleaning notebook starts with the main datasets and slowly chips away and removes datasets that wont be used.
After that, the dataset is then clean and later merged together. 

# Normalizes data
icu_stays['first_careunit']= icu_stays['first_careunit'].str.strip().str.lower()

# Checks if the dataset has an missings
input_events.isna().sum()

# Drops columns

input_events=input_events.drop(columns=['hadm_id','caregiver_id','starttime','endtime','storetime','amount','amountuom','rate',
                                        'rateuom', 'orderid','linkorderid','ordercategoryname','secondaryordercategoryname',
                                        'patientweight','totalamount','totalamountuom','isopenbag','continueinnextdept',
                                        'originalamount','originalrate','statusdescription'])

# Aggregates variables (Import for merging)
inputs_agg = input_events.groupby('subject_id').agg({
    'ordercomponenttypedescription': lambda x: list(set(x)),
    'ordercategorydescription': lambda x: list(set(x))
}).reset_index()

# Joins datasets together on subject_id
patient_admin=patients.merge(adm_agg,on=['subject_id'],how='inner')

# Creates the 60/20/20 Train/Test/Validation datasets
train_df,temp_df=train_test_split(demo_care_clean,
                                 test_size=0.4,
                                 stratify=demo_care['death_flag'],
                                 random_state=42)

val_df, test_df=train_test_split(temp_df,
                                test_size=0.5,
                                stratify=temp_df['death_flag'],
                                random_state=42)

# creating folder for exported data
final_data_path = os.path.join("..", "Data", "Final Data")
os.makedirs(final_data_path, exist_ok=True)

# Export dataset
demo_care.to_csv(os.path.join(final_data_path, "demo_care.csv"), index=False)
demo_care_clean.to_csv(os.path.join(final_data_path, "demo_care_clean.csv"), index=False)
train_df.to_csv(os.path.join(final_data_path, "train_dataset.csv"), index=False)
val_df.to_csv(os.path.join(final_data_path, "validation_dataset.csv"), index=False)
test_df.to_csv(os.path.join(final_data_path, "test_dataset.csv"), index=False)

Data modeling starting with the importing data then models the data using OLS, Lasso, Ridge, Random forest regression, Logistic regression and Random forest classifier.

Classification and regression targets
# Regression Target
target_los='log_los'

# Classification target:
target_mort='death_flag'


drop_cols=['los_hours','los','long_stay_flag','death_within_24h_flag',target_los,target_mort]

# Dropping columns in each dataset

#Training
X_train=train.drop(columns=drop_cols,errors='ignore')
y_train_los=train[target_los]
y_train_mort=train[target_mort]

# Val

X_val=validation.drop(columns=drop_cols,errors='ignore')
y_val_los=validation[target_los]
y_val_mort=validation[target_mort]

#Test
X_test=test.drop(columns=drop_cols,errors='ignore')
y_test_los=test[target_los]
y_test_mort=test[target_mort]

# OLS 
ols=LinearRegression()
ols.fit(X_train,y_train_los)

y_pred_val_ols=ols.predict(X_val)
y_pred_test_ols=ols.predict(X_test)

# Lasso
lasso= Lasso(max_iter=5000)

lasso_params={'alpha':np.logspace(-3,1,20)}
grid_lasso=GridSearchCV(lasso, lasso_params, cv=5, scoring='r2')
grid_lasso.fit(X_train, y_train_los)

best_lasso=grid_lasso.best_estimator_

# Plotting residual plot
plt.scatter(y_val_los, y_pred_val_lasso, alpha=0.6)
plt.xlabel("Actual LOS")
plt.ylabel("Predicted LOS")
plt.title("Residual Plot - Lasso Regression")
plt.show()

# Ridge

ridge=Ridge(max_iter=5000)
ridge_params={'alpha':np.logspace(-3,3,20)}

grid_ridge=GridSearchCV(ridge, ridge_params, cv=5, scoring='r2')
grid_ridge.fit(X_train, y_train_los)

best_ridge= grid_ridge.best_estimator_

# Random forest regression

rf_reg = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    max_depth=None
)
rf_reg.fit(X_train, y_train_los)

y_pred_val_rf = rf_reg.predict(X_val)
y_pred_test_rf = rf_reg.predict(X_test)

# Logistic regression
log_model = LogisticRegression(max_iter=5000, class_weight='balanced')
log_model.fit(X_train, y_train_mort)

y_pred_proba=log_model.predict_proba(X_val)[:,1]
roc_auc=roc_auc_score(y_val_mort,y_pred_proba)

# Odds ratio
odds_ratios=np.ratios=np.exp(log_model.coef_[0])
coeff_table=pd.DataFrame({'Feature':X_train.columns,
                         'Odds_ratio':odds_ratios}).sort_values(by='Odds_ratio', ascending=False)
coeff_table.head(10)

# ROC Curve

fpr, tpr, _ = roc_curve(y_val_mort, y_pred_proba)
plt.plot([0,1], [0,1], '--', color='blue', label='Random Chance')
plt.plot(fpr, tpr, color='red', label='ROC Curve') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.show

# Random Forest classifier
rf_clf = RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced',
    random_state=42
)
rf_clf.fit(X_train, y_train_mort)

y_pred_proba_rf = rf_clf.predict_proba(X_val)[:,1]
roc_auc_rf=roc_auc_score(y_val_mort,y_pred_proba_rf)

# Confusion matrix

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['Survive', 'Death'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Random Forest')
plt.show()


# Final results summary
After completing my data analysis, I found that the number of drugs administer to a patient was a statistically significant variable that affects both the length of stay and mortality rate of a patient. While I found that age was statistically significant for only mortality rate. Based on these findings, I have came up with two solutions based on the variables. For the number of drugs administer, the solution to reduce this is creating a recommendation system that helps recommend the doctors prescriptions that best meet the patients need. The recommendation system will accept the patients conditions along with the doctors recommended prescription. With these two pieces of information inputted, the recommendation system will then go ahead and give doctors additional prescriptions that have comparable results. If there is a better prescription then what the doctor assigned it will let the doctor know. It will also let the doctor know if the prescription can be administered with other prescriptions. By implementing this recommendation system, you reduce the length of stay as patients can have multiple prescriptions at once, reducing the time spent on waiting for a prescription to be fully absorbed by a patient. It will also reduce the number of prescriptions because if there is a prescription that is like the doctors recommendation then it would recommend it and list the additional conditions the prescription helps treat. The other solution is introducing a higher mortality risk wrist band. This would be adminstered by a hot to cold scale.  Red being higher risk (Older patients) and Blue being lower risk patient. This will let hospital staff know which patients might need additional care. By implementing these solutions, the hospital can increase the care that their patients are receiving and maintain relatively low costs.