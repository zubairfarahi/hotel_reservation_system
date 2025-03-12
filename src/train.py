import os
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from config.model_params import *
from utils.common_function import read_yaml,load_data
from scipy.stats import randint

import mlflow
import mlflow.sklearn

from zlogger.logger import ZLogger
import configparser

path_file = "config/logging.ini"
config = configparser.ConfigParser()
config.read(path_file)
log = ZLogger("hotel_reservation_system", config)

class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path =model_output_path

        self.params_dist = LIGHTGM_PARAMS 
        self.random_search_params = RANDOM_SEARCH_PARAMS
    
    def load_and_split_data(self):

        try:
            
            log.info(f'Loding train data from {self.train_path}')
            train_df = load_data(self.train_path) 

            log.info(f'Loding test data from {self.test_path}')
            test_df = load_data(self.test_path) 

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status'] 

            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df ['booking_status'] 

            log.info('Data splitted successfully for model training')
            return X_train,y_train,X_test,y_test
        except Exception as e:
            log.error(f'Error while loading  data {e}')
    
    def train_lgbm(self, X_train, y_train):
        try:
            log.info('Inializing our model')

            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])

            log.info('Starting our Hyperparamter tuning')

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                cv=self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
                
            )

            log.info('Starting model training')

            random_search.fit(X_train, y_train)

            log.info('End model training')

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            log.info(f'Best Param are: {best_params}')

            return best_lgbm_model
        except Exception as e:
            log.error(f'Error while training  data {e}')
    
    def evaluate_model(self , model , X_test , y_test):
        try:
            log.info("Evaluating our model")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)

            log.info(f"Accuracy Score : {accuracy}")
            log.info(f"Precision Score : {precision}")
            log.info(f"Recall Score : {recall}")
            log.info(f"F1 Score : {f1}")

            return {
                "accuracy" : accuracy,
                "precison" : precision,
                "recall" : recall,
                "f1" : f1
            }
        except Exception as e:
            log.error(f"Error while evaluating model {e}")
    
    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)

            log.info("saving the model")
            joblib.dump(model , self.model_output_path)
            log.info(f"Model saved to {self.model_output_path}")

        except Exception as e:
            log.error(f"Error while saving model {e}")
    
    def run(self):
        try:
            with mlflow.start_run():
                log.info("Starting our Model Training pipeline")

                log.info("Starting our MLFLOW experimentation")

                log.info("Logging the training and testing datset to MLFLOW")
                mlflow.log_artifact(self.train_path , artifact_path="datasets")
                mlflow.log_artifact(self.test_path , artifact_path="datasets")

                X_train,y_train,X_test,y_test =self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train,y_train)
                metrics = self.evaluate_model(best_lgbm_model ,X_test , y_test)
                self.save_model(best_lgbm_model)

                log.info("Logging the model into MLFLOW")
                mlflow.log_artifact(self.model_output_path)

                log.info("Logging Params and metrics to MLFLOW")
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                log.info("Model Training sucesfullly completed")

        except Exception as e:
            log.error(f"Error in model training pipeline {e}")


if __name__=="__main__":
    
    trainer = ModelTraining('data/train.csv','data/test.csv','model/lgbm_model.pkl')
    trainer.run()