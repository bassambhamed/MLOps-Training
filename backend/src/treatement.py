import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support as score
import mlflow
import warnings
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

def logisticRegressionModel(data_url,version,df,X_resampled,y_resampled,X_test,y_test):
    # disable autologging
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='LogisticRegression'):
        mlflow.log_param("data_url",data_url)
        mlflow.log_param("data_version",version)
        mlflow.log_param("input_rows",df.shape[0])
        mlflow.log_param("input_cols",df.shape[1])
        #model fitting and training
        lr=LogisticRegression()
        mlflow.set_tag(key= "model",value="LogisticRegression")
        params = lr.get_params()
        mlflow.log_params(params)
        lr.fit(X_resampled,y_resampled)
        train_features_name = f'{X_resampled=}'.split('=')[0]
        train_label_name = f'{y_resampled=}'.split('=')[0]
        mlflow.set_tag(key="train_features_name",value= train_features_name)
        mlflow.set_tag(key= "train_label_name",value=train_label_name)
        predicted=lr.predict(X_test)
        precision,recall,fscore,support=score(y_test,predicted,average='macro')
        mlflow.log_metric("Precision_test",precision)
        mlflow.log_metric("Recall_test",recall)
        mlflow.log_metric("F1_score_test",fscore)
        mlflow.sklearn.log_model(lr,artifact_path="ML_models")

def RandomForestModel(data_url,version,df,X_resampled,y_resampled,X_test,y_test):
    # disable autologging
    mlflow.sklearn.autolog(disable=True)
    with mlflow.start_run(run_name='RandomForest'):
        mlflow.log_param("data_url",data_url)
        mlflow.log_param("data_version",version)
        mlflow.log_param("input_rows",df.shape[0])
        mlflow.log_param("input_cols",df.shape[1])
        rf = RandomForestClassifier(random_state=5)
        mlflow.set_tag(key="model", value = "RandomForest")
        params = rf.get_params()
        mlflow.log_params(params)
        rf.fit(X_resampled,y_resampled)
        train_features_name = f'{X_resampled=}'.split('=')[0]
        train_label_name = f'{y_resampled=}'.split('=')[0]
        mlflow.set_tag(key="train_features_name",value= train_features_name)
        mlflow.set_tag(key= "train_label_name",value=train_label_name)
        predicted=rf.predict(X_test)
        precision,recall,fscore,support=score(y_test,predicted,average='macro')
        mlflow.log_metric("Precision_test",precision)
        mlflow.log_metric("Recall_test",recall)
        mlflow.log_metric("F1_score_test",fscore)
        mlflow.sklearn.log_model(rf,artifact_path="ML_models")

def XGBoostModel(data_url,version,df,X_train,y_train,X_test,y_test):
    # disable autologging
    mlflow.xgboost.autolog(disable=True)
    with mlflow.start_run(run_name='XGBoost'):
        mlflow.log_param("data_url",data_url)
        mlflow.log_param("data_version",version)
        mlflow.log_param("input_rows",df.shape[0])
        mlflow.log_param("input_cols",df.shape[1])
        xg = XGBClassifier()
        params = xg.get_params()
        mlflow.set_tag(key= "model", value="XGBClassifier")
        mlflow.log_params(params)
        xg.fit(X_train,y_train)
        train_features_name = f'{X_train=}'.split('=')[0]
        train_label_name = f'{y_train=}'.split('=')[0]
        mlflow.set_tag(key="train_features_name",value= train_features_name)
        mlflow.set_tag(key= "train_label_name",value=train_label_name)
        predicted=xg.predict(X_test)
        precision,recall,fscore,support=score(y_test,predicted,average='macro')
        mlflow.log_metric("Precision_test",precision)
        mlflow.log_metric("Recall_test",recall)
        mlflow.log_metric("F1_score_test",fscore)
        mlflow.xgboost.log_model(xg,artifact_path="ML_models")