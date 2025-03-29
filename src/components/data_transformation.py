import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin
from src.outlier_detection import train_isolation_forest
from src.outlier_detection import get_anomaly_and_score, get_outliers_index



from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
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
                "make"
            ]
            
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("outliers_handler", OutlierHandler(columns=numerical_columns)),
                ("z_score_method",ZScoreTransformer(columns = ['odometer'])),
                ("OutlierRemoval",OutlierRemover()),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("encoding", Encoder(columns=categorical_columns))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns),
                ("drop",DropColumn(['post_id']),['post_id'])

                ],
                remainder="passthrough"
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            clean_obj = CleanData()
            feature_obj = FeatureCreation()

            train_df = clean_obj.fit_transform(train_df)
            test_df = clean_obj.fit_transform(test_df)

            train_df = feature_obj.fit_transform(train_df)
            test_df = feature_obj.fit_transform(test_df)


            logging.info("Cleaned/Created features to train and test data")

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

class CleanData(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X.columns = X.columns.str.strip().str.lower().str.replace(":", "").str.replace(" ", "_")
        X = X.drop_duplicates().reset_index(drop=True)
        return X

        
class FeatureCreation(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        from datetime import datetime 
        self.current_year = datetime.now().year
        return self

    def transform(self, X):
        X = X.copy()
        X['age'] = self.current_year - X['year']
        X = X[X['age']>=0]

        X['make'] = X['model'].fillna('other').str.split().str[0].str.lower()

        
        corrections = {
            'chevolet': 'chevrolet',
            'chev': 'chevrolet',
            'chevy': 'chevrolet',
            'caddilac': 'cadillac',
            'infinity': 'infiniti',
            'land': 'land rover',
            'range': 'land rover',
            'corvette':'other',
            'freightliner':'other',
            'vw':'volkswagen',
            'mercedes-benz':'mercedes'
        }
        X['make'] = X['make'].replace(corrections)
        make_count = X['make'].value_counts()
        X['make'] = X['make'].map(lambda x: x if make_count.get(x, 0) > 3 else 'other')

        return X
        

class OutlierHandler(BaseEstimator, TransformerMixin): #Removes 435 instances
    def __init__(self, threshold=-0.55, columns = None):
        self.threshold = threshold
        self.columns = columns
        self.clf = None 

    def fit(self, X, y=None):
        """Trains the Isolation Forest on specified features."""

        self.clf = train_isolation_forest(X)
        return self

    def transform(self, X):
        """Detects and removes outliers."""
        data = X.copy()
        X = pd.DataFrame(X, columns = self.columns)
       
        new_data = get_anomaly_and_score(data, self.clf)
        outlier_index, clean_index = get_outliers_index(new_data, mode='threshold', threshold=self.threshold)
        #print(f"SHAPE BEFORE: {X.shape}")
        

        X = X.loc[clean_index].reset_index(drop=True)
        #print(f"SHAPE after: {X.shape}")

        return X

class ZScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=2.5, columns = None):
        self.threshold = threshold
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        log_X = np.log1p(X)

        z_scores = np.abs(stats.zscore(log_X, nan_policy='omit'))
        X["outlier_flag"] = 0
        X["outlier_flag"] = (z_scores > self.threshold).astype(int).max(axis=1)  
        # X[np.all(z_scores < self.threshold, axis=1)]
    
        return X


class Encoder(BaseEstimator,TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns)
        
        X["vin"] = (X["vin"] != "Missing").astype(int)
        categorical_features = ["fuel", "transmission", "cylinders", "drive"]
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
        def OHE_top_x(X, feature, top_x_labels):
            for label in top_x_labels:
                X[feature+'_'+label] = np.where(X[feature]==label,1,0)
        features = ['paint_color','type','make']
        for feature in features:
            top_10 = [x for x in X[feature].value_counts().sort_values(ascending=False).head(10).index]
            OHE_top_x(X, feature, top_10)
        return X
    
class OutlierRemover(BaseEstimator, TransformerMixin): #Removes 196 instances
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X = X[X["outlier_flag"] == 0].drop(columns=["outlier_flag"]).reset_index(drop=True)
        X['odometer'], parameters = stats.boxcox(X['odometer'])
        #print(f"SHAPE OF X AFTER: {X.shape}")
        return X

class DropColumn(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        return X.drop(columns=self.cols, axis=1, errors='ignore')