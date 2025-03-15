import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from notebook.outliers.multi_columns import train_isolation_forest
from notebook.outliers.multi_columns import get_anomaly_and_score, get_outliers_index, plot_anomaly


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def clean_df(self,df):
        try:
            df.columns = df.columns.str.strip().str.lower().str.replace(":", "").str.replace(" ", "_")
            df['model'] = df['model'].astype(str).str.strip()
            df = df.drop_duplicates().reset_index(drop=True)
            return df
        
        except Exception as e:
            raise CustomException(e,sys)

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["odometer",'age']
            categorical_columns = [
                "vin",
                "condition",
                "fuel",
                "type",
                "paint_color",
                "title_status",
                "transmission",
                "cylinders",
                "drive",
                "model"
            ]


            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("featureCreation",FeatureCreation(year_column='year', model_column='model')),
                ("outliers", OutlierHandler(features=numerical_columns))

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("one_hot_encoder",OneHotEncoder())
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    
    class featureCreation(BaseEstimator,TransformerMixin):
        def __init__(self, year_column='year', model_column='model'):
            self.year_column = year_column
            self.model_column = model_column
            self.current_year = datetime.now().year
        def fit(self, X, y=None):
        # No fitting is required for this transformer, so we return self
            return self
    
        def transform(self, X):
            # Ensure the input is a pandas DataFrame
            X = X.copy()
            
            # Feature 1: Create 'age' from the year column
            X['age'] = self.current_year - X[self.year_column]

            # Feature 2: Odometer - Create a new feature that might be a transformation of the original
            # For example, you could apply a log transformation, if it makes sense for the data
            X['odometer_log'] = X[self.odometer_column].apply(lambda x: x if x == 0 else (x + 1).log())
            
            # Return the transformed DataFrame with the new features
            return X
            

    class OutlierHandler(BaseEstimator, TransformerMixin):
        def __init__(self, features=None, threshold=-0.55):
            self.features = features  # List of numerical features to check
            self.threshold = threshold  # Threshold for outlier detection
            self.clf = None  # Placeholder for trained model

        def fit(self, X, y=None):
            """Trains the Isolation Forest on specified features."""
            if self.features is None:
                self.features = X.columns.tolist()  # Use all columns if not specified
            self.clf = train_isolation_forest(X, self.features)
            return self

        def transform(self, X):
            """Detects and removes outliers."""
            new_data = get_anomaly_and_score(X, self.features, self.clf)
            outlier_index, clean_index = get_outliers_index(new_data, mode='threshold', threshold=self.threshold)
            
            return X.loc[clean_index].reset_index(drop=True)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            train_df = self.clean_df(train_df)
            test_df = self.clean_df(test_df)

            logging.info("Data cleaning completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="price"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)