import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def calc_wqi(self, df):
        # Standards and weights
        nitrate_Si, nitrate_Vi = 45, 0
        fc_Si, fc_Vi = 2500, 0
        tc_Si, tc_Vi = 5000, 0
        bod_Si, bod_Vi = 3, 0
        ph_Vi = 7.0
        ph_Si_upper, ph_Si_lower = 8.5, 6.5
        do_Si, do_Vi = 5.0, 14.6

        weights = {
            'nitrate': 0.2,
            'fc': 0.2,
            'tc': 0.2,
            'bod': 0.15,
            'ph': 0.15,
            'do': 0.1
        }

        def calc_wqi_row(row):
            nitrate = (row['N_Min'] + row['N_Max']) / 2
            fc = (row['FC_Min'] + row['FC_Max']) / 2
            tc = (row['TC_Min'] + row['TC_Max']) / 2
            bod = (row['BOD_Min'] + row['BOD_Max']) / 2
            ph = (row['pH_Min'] + row['pH_Max']) / 2
            do = (row['Oxy_Min'] + row['Oxy_Max']) / 2

            nitrate_qi = ((nitrate - nitrate_Vi) / (nitrate_Si - nitrate_Vi)) * 100
            fc_qi = ((fc - fc_Vi) / (fc_Si - fc_Vi)) * 100
            tc_qi = ((tc - tc_Vi) / (tc_Si - tc_Vi)) * 100
            bod_qi = ((bod - bod_Vi) / (bod_Si - bod_Vi)) * 100
            ph_qi = ((ph_Vi - ph) / (ph_Vi - ph_Si_lower) * 100) if ph < ph_Vi else ((ph - ph_Vi) / (ph_Si_upper - ph_Vi)) * 100
            do_qi = ((do_Si - do) / (do_Si - do_Vi)) * 100

            return (
                nitrate_qi * weights['nitrate'] +
                fc_qi * weights['fc'] +
                tc_qi * weights['tc'] +
                bod_qi * weights['bod'] +
                ph_qi * weights['ph'] +
                do_qi * weights['do']
            )

        df['WQI'] = df.apply(calc_wqi_row, axis=1)
        df['WQI_log'] = np.log1p(df['WQI'])
        return df

    def remove_outliers(self, df):
        features = [
            'Temp_Min', 'Temp_Max', 'Oxy_Min', 'Oxy_Max', 'N_Min', 'N_Max',
            'FC_Min', 'FC_Max', 'TC_Min', 'TC_Max', 'Con_Min', 'Con_Max',
            'BOD_Min', 'BOD_Max', 'pH_Min', 'pH_Max'
        ]
        iso = IsolationForest(contamination=0.01, random_state=42)
        df['is_outlier'] = iso.fit_predict(df[features]) == -1
        return df[~df['is_outlier']].drop(columns='is_outlier').reset_index(drop=True)

    def get_preprocessor(self):
        categorical_features = ['State']
        numerical_features = [
            'Temp_Min', 'Temp_Max', 'Oxy_Min', 'Oxy_Max', 'N_Min', 'N_Max',
            'FC_Min', 'FC_Max', 'TC_Min', 'TC_Max', 'Con_Min', 'Con_Max',
            'BOD_Min', 'BOD_Max', 'pH_Min', 'pH_Max'
        ]
        num_pipeline = Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler",StandardScaler())
        ])
        cat_pipeline  = Pipeline(steps=[
            ("onehot", OneHotEncoder())
            ])
        logging.info("creating numerical and categorical feature pipelines")
        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, numerical_features),
                ("cat_pipeline", cat_pipeline, categorical_features)
            ]
        )
        return preprocessor
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data loaded successfully")

            train_df = self.calc_wqi(train_df)
            test_df = self.calc_wqi(test_df)
            train_df = self.remove_outliers(train_df)

            features = [
                'Temp_Min', 'Temp_Max', 'Oxy_Min', 'Oxy_Max', 'N_Min', 'N_Max',
                'FC_Min', 'FC_Max', 'TC_Min', 'TC_Max', 'Con_Min', 'Con_Max',
                'BOD_Min', 'BOD_Max', 'pH_Min', 'pH_Max', 'State'
            ]
            target = 'WQI_log'

            X_train = train_df[features]
            y_train = train_df[target]
            X_test = test_df[features]
            y_test = test_df[target]
            logging.info("Features and target variable separated")


            prepocessor_obj = self.get_preprocessor()
            X_train_transformed = prepocessor_obj.fit_transform(X_train)
            X_test_transformed = prepocessor_obj.transform(X_test)

            logging.info("Data transformation completed")

            train_arr = np.c_[X_train_transformed,np.array(y_train)]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            logging.info("Saving preprocessor object")

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=prepocessor_obj
            )
            logging.info("Preprocessor object saved successfully")
            return (
                train_arr, test_arr, self.config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)
