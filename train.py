import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import joblib


dataset = pd.read_csv('train.csv')
numerical_cols = dataset.select_dtypes(include=['int64','float']).columns.tolist()
categorical_cols = dataset.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('Loan_Status')
categorical_cols.remove('Loan_ID')

# filling categorical columns with mode

for col in categorical_cols:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

for col in numerical_cols:
    dataset[col].fillna(dataset[col].median(),inplace=True)

# OUTLIER IPUTATION

dataset[numerical_cols] = dataset[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

# log transformation
dataset['LoanAmount'] = np.log(dataset['LoanAmount']).copy()
dataset['Total_income'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['Total_income'] = np.log(dataset['Total_income']).copy()

# Dropping ApplicantIncome and CoapplicantIncome
dataset = dataset.drop(columns=['ApplicantIncome','CoapplicantIncome'])

# Label encoding categorical variables

for col in categorical_cols:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])

# Encode the target variable

dataset['Loan_Status'] = le.fit_transform(dataset['Loan_Status'])

# TRAIN TEST train_test_split

X = dataset.drop(columns=['Loan_Status','Loan_ID'],axis=1)
y = dataset['Loan_Status']

rf = RandomForestClassifier(random_state=0)
param_grid_forest = {
    'n_estimators': [200,400,700],
    'max_depth': [10,20,30],
    'criterion': ["gini","entropy"],
    'max_leaf_nodes':[50,100]
}

grid_forest = GridSearchCV(
    estimator = rf,
    param_grid=param_grid_forest,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

model_forest = grid_forest.fit(X,y)

joblib.dump(model_forest, 'RF_loan_model.joblib')

loaded_model = joblib.load('RF_Loan_model.joblib')

data = [[
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.98745,
                360.0,
                1.0,
                2.0,
                8.698
            ]]
print(f"Prediction is : {loaded_model.predict(pd.DataFrame(data))}")