import joblib
import pandas as pd

pipeline=joblib.load("model/pipeline.pkl")

def predict(data: dict):
    df=pd.DataFrame([data])
    prediction=pipeline.predict(df)[0]
    probablity=pipeline.predict_proba(df)[0].tolist()
    
    return{
          "prediction": int(prediction),
          "probablity": {
             "No Survival": probablity[0],"Survival":probablity[1]}
        }
