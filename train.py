from sklearn import preprocessing
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

df=sns.load_dataset("titanic")
df=df[["pclass","sex","age","fare","survived"]]
x=df.drop("survived",axis=1)
y=df["survived"]
numeric_features=["age","fare"]
categorial_features=["sex","pclass"]
numerical_pipeline=Pipeline([("Imputer",SimpleImputer(strategy="median")),("Scaler",StandardScaler())]) # Instantiated StandardScaler
categorical_pipeline=Pipeline([("Imputer",SimpleImputer(strategy="most_frequent")),("Encoder",OrdinalEncoder())]) # Corrected Pipeline syntax, strategy typo, and instantiated OrdinalEncoder
preprocessor=ColumnTransformer([("num",numerical_pipeline,numeric_features),
                                ("cat",categorical_pipeline,categorial_features)]) # Renamed variable to 'preprocessor' to avoid shadowing and match usage
pipeline=Pipeline([("Preprocessor",preprocessor),
                   ("Classifier",LogisticRegression())])
pipeline.fit(x,y)
os.makedirs("model",exist_ok=True)
joblib.dump(pipeline,"model/pipeline.pkl")
print("Pipeline saved successfully.")