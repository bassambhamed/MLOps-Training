#from dotenv import load_dotenv
import os
from data_preprocessing_training import transform_data
import pandas as pd
from imblearn.over_sampling import SMOTE
import mlflow
from treatement import RandomForestModel
from treatement import XGBoostModel
from treatement import logisticRegressionModel

#Load environement variable (Dagshub credentials)
# from dotenv import load_dotenv
# import os
# load_dotenv("../backend/src/.env")

# DagsHub_username = os.getenv("DagsHub_username")
# DagsHub_token=os.getenv("DagsHub_token")

#Get Dagshub credentials
# os.environ['MLFLOW_TRACKING_USERNAME']= DagsHub_username
# os.environ["MLFLOW_TRACKING_PASSWORD"] = DagsHub_token

#Affect Daghsub credentials 

os.environ['MLFLOW_TRACKING_USERNAME']= "..."
os.environ["MLFLOW_TRACKING_PASSWORD"] = "..."

#setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/.../....mlflow') #your mlfow tracking uri
mlflow.set_experiment("fraud-detector-experiment")

#Data Url and version
version = "v2.0"
data_url = "../../dataGit/fraud2.csv"

#read the data
df = pd.read_csv(data_url)

#cleaning and preprocessing
X_train,X_test,y_train,y_test = transform_data(df)
method= SMOTE()
X_resampled, y_resampled = method.fit_resample(X_train, y_train)

#Execute the models with new version of data:
logisticRegressionModel(data_url,version,df,X_resampled,y_resampled,X_test,y_test)
XGBoostModel(data_url,version,df,X_resampled,y_resampled,X_test,y_test)
RandomForestModel(data_url,version,df,X_resampled,y_resampled,X_test,y_test)


