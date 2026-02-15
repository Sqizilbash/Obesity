# Obesity
Predict Obesity
Problem Statement: Obesity is a major public health concern that increases the risk of chronic diseases such as diabetes, cardiovascular diseases, and hypertension. Early detection and intervention are critical to reduce health risks. The objective of this project is to predict obesity levels of individuals based on their eating habits, physical condition, and lifestyle factors using various machine learning models. 
By building predictive models, healthcare providers and individuals can identify potential obesity risks and take preventive measures.
Dataset Description: - 
Dataset: Estimation of Obesity Levels Based on Eating Habits and Physical Condition (UCI Repository) – 
Number of Instances: 2111 – 
Number of Features: 16
Features: 
Gender (Categorical) - Gender of the individual 
Age (Continuous) - Age of the individual 
Height (Continuous) - Height in meters 
Weight (Continuous) - Weight in kg 
family_history_with_overweight (Binary) - Family member with overweight? (yes/no) 
FAVC (Binary) - Frequently eats high caloric food? (yes/no) 
FCVC (Integer) - Frequency of vegetable consumption (1-3) 
NCP (Continuous) - Number of main meals per day 
CAEC (Categorical) - Eats between meals? (Never/Sometimes/Frequently/Always) 
SMOKE (Binary) - Smokes? (yes/no) 
CH2O (Continuous) - Daily water consumption (liters) 
SCC (Binary) - Monitors calories? (yes/no) 
FAF (Continuous) - Frequency of physical activity 
TUE (Integer) - Time using technological devices (0-2) 
CALC (Categorical) - Alcohol consumption (Never/Sometimes/Frequently/Always) 
MTRANS (Categorical) - Transportation method (Walking/Public Transport/Automobile/Bike/Motorbike) 
NObeyesdad (Target) - Obesity level (7 classes: Insufficient Weight, Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, II, III)
Models Used and Evaluation Metrics:
ML Model Name	Accuracy	AUC Score	Precision	Recall	F1 Score	MCC
Logistic Regression	0.8794	0.9846	0.8779	0.879	0.8781	0.859
Decision Tree	0.9125	0.9487	0.9157	0.913	0.9134	0.898
kNN	0.8369	0.9653	0.8342	0.837	0.8297	0.811
Naive Bayes	0.4941	0.8336	0.5131	0.494	0.4435	0.433
Random Forest	0.9433	0.995	0.9479	0.943	0.9443	0.934
XGBoost	0.9551	0.9977	0.9572	0.955	0.9555	0.948

Observations on Model Performance:
ML Model Name	Observation about Model Performance
Logistic Regression	Performs moderately; captures linear relationships well but may miss complex feature interactions.
Decision Tree	Shows good accuracy and is interpretable; may overfit without ensemble methods.
kNN	Performs reasonably; sensitive to feature scaling and distance metrics.
Naive Bayes	Fast and simple; low accuracy on this dataset due to feature dependencies.
Random Forest	Very good accuracy; ensemble reduces overfitting and captures feature interactions effectively.
XGBoost	Highest performance; robust to overfitting; handles non-linear relationships and feature importance well.


