import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import joblib



numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = [
        'gender',
        'seniorcitizen',
        'partner',
        'dependents',
        'phoneservice',
        'multiplelines',
        'internetservice',
        'onlinesecurity',
        'onlinebackup',
        'deviceprotection',
        'techsupport',
        'streamingtv',
        'streamingmovies',
        'contract',
        'paperlessbilling',
        'paymentmethod',
    ]


def createDataframe():
    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    df = pd.read_csv(data_url)

    df.columns = df.columns.str.lower().str.replace(' ', '_')
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')
    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)
    df.churn = (df.churn == 'yes').astype(int)
  
    return df


def trainModel(df):
    y_train = df.churn
    model = LogisticRegression(solver='liblinear')
    X_train = df[numerical + categorical]  

    cat_pipe = Pipeline([
        ('encode', OneHotEncoder(handle_unknown ="ignore", sparse_output=False))
    ]

    )
    preprocessor = ColumnTransformer(transformers=[
        ("cat",cat_pipe, categorical)
    ])
    final_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model",model)
        ]
    )
    return final_pipeline.fit(X_train, y_train)

def saveModel(pipeline):
    joblib.dump(pipeline, 'model.pkl')
    print("model.pkl created")


df = createDataframe()
pipeline = trainModel(df)
saveModel(pipeline)


 


