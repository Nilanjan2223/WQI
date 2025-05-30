import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models
from src.utils import save_object

import os
import sys

@dataclass
class ModelTrainingConfig:
    train_model_path_file = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            model = {
                RandomForestRegressor: RandomForestRegressor(),
                AdaBoostRegressor: AdaBoostRegressor(),
                LinearRegression: LinearRegression(),
                KNeighborsRegressor: KNeighborsRegressor(),
                DecisionTreeRegressor: DecisionTreeRegressor(),
                XGBRegressor: XGBRegressor(),
                CatBoostRegressor: CatBoostRegressor(verbose=False)
            }
            model_report:dict= evaluate_models(X_train, y_train, X_test, y_test, model)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", None)
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(file_path=self.model_trainer_config.train_model_path_file,
                        obj=best_model_name)
            best_model_name = max(model_report, key=model_report.get)
            best_model = model[best_model_name]

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test, predicted)

            return r2
        except Exception as e:
            raise CustomException(e, sys)
        