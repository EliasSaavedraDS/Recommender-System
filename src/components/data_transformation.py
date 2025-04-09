import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from scipy import stats

from sklearn import set_config
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

set_config(transform_output="pandas")

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
                ("outliers_handler", OutlierHandler()),
                ("z_score_method",ZScoreTransformer()),
                #("OutlierRemoval",OutlierRemover()),
                #("scaler",StandardScaler())
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


            preprocessor = Pipeline(
                steps=[
                    ("transformations", ColumnTransformer(
                        transformers=[
                            ("num_pipeline", num_pipeline, numerical_columns),
                            ("cat_pipeline", cat_pipeline, categorical_columns)
                        ],
                        remainder="drop"
                    )),
                    ("outlier_remover", OutlierRemover())
                ]
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
            test_df = clean_obj.transform(test_df)

            train_df = feature_obj.fit_transform(train_df)
            test_df = feature_obj.transform(test_df)


            logging.info("Cleaned/Created features to train and test data")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="price"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            # train_indices = input_feature_train_df.index
            # test_indices = input_feature_test_df.index
            preprocessing_obj.set_output(transform="pandas")

            target_feature_train_df = target_feature_train_df.fillna(target_feature_train_df.median())
            target_feature_test_df = target_feature_test_df.fillna(target_feature_test_df.median())

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            logging.info(f"Training data shape before fit_transform: {input_feature_train_df.shape}")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            logging.info(f"Index sample: {input_feature_train_arr.index[:5].tolist()}")
            logging.info(f"Columns: {input_feature_train_arr.columns.tolist()}")

            # target_feature_train_df, input_feature_train_df = target_feature_train_df.align(input_feature_train_df, join='inner')
            # target_feature_test_df, input_feature_test_df = target_feature_test_df.align(input_feature_test_df, join='inner')
            input_feature_train_arr = input_feature_train_arr.reset_index(drop=True)
            target_feature_train_df = target_feature_train_df.reset_index(drop=True)
            input_feature_test_arr = input_feature_train_arr.reset_index(drop=True)
            target_feature_test_df = target_feature_train_df.reset_index(drop=True)


            logging.info(f"Training data shape after fit_transform: {input_feature_train_arr.shape}")
            logging.info(f"Test data shape after transform: {input_feature_test_arr.shape}")

            target_feature_train_df = target_feature_train_df.loc[input_feature_train_arr.index]
            target_feature_test_df = target_feature_test_df.loc[input_feature_test_arr.index]
            # target_feature_train_df = target_feature_train_df.iloc[:input_feature_train_arr.shape[0]]
            # target_feature_test_df = target_feature_test_df.iloc[:input_feature_test_arr.shape[0]]

            logging.info(f"Target data shape after alignement: {target_feature_train_df.shape}")
            logging.info(f"Target data shape after alignement: {target_feature_test_df.shape}")

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
    def __init__(self, threshold=-0.55):#, columns = None):
        self.threshold = threshold
        #self.columns = columns
        self.clf = None 

    def fit(self, X, y=None):
        """Trains the Isolation Forest on specified features."""
        self.columns_ = X.columns.tolist()
        self.clf = train_isolation_forest(X)
        return self

    def transform(self, X):
        """Detects and removes outliers."""
        data = X.copy()
        #X = pd.DataFrame(X, columns = self.columns)
        
       
        new_data = get_anomaly_and_score(data, self.clf)
        outlier_index, clean_index = get_outliers_index(new_data, mode='threshold', threshold=self.threshold)
        
        X["outlier_flag"] = 'non-outlier'
        X.loc[outlier_index, "outlier_flag"] = 'outlier'
        #print(f'count {X[X['outlier_flag'] == "outlier"].shape[0]}')

        #X = X.loc[clean_index].reset_index(drop=True)
        self.columns_ = X.columns.tolist()
        #print(f'after iso:{X.shape}')
        return X
    def get_feature_names_out(self, input_features=None):
        return self.columns_

class ZScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=2.5):
        self.threshold = threshold
        

    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self
    
    def transform(self, X):
        
        data = X.copy()
        clean_data = data[data["outlier_flag"] == 'non-outlier'].copy()

        log_X = np.log1p(clean_data['odometer'])

        z_scores = np.abs(stats.zscore(log_X, nan_policy='omit'))

        new_outlier_index = np.where((z_scores > self.threshold))[0]
        clean_data.loc[clean_data.index[new_outlier_index], "outlier_flag"] = 'outlier'
        data.loc[new_outlier_index, "outlier_flag"] = "outlier"

        #clean_data = data[data["outlier_flag"] == 'non-outlier'].copy()
        clean_data['odometer'], parameters = stats.boxcox(clean_data['odometer'])
        data.loc[clean_data.index, 'odometer'] = clean_data['odometer']
        #print(f'after z score:{data.shape}')
        return data
    def get_feature_names_out(self, input_features=None):
        return self.columns_


class Encoder(BaseEstimator,TransformerMixin):
    def __init__(self, columns = None):
        self.columns = columns
        self.top_10_labels = {}
    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=self.columns)
        # for feature in ['paint_color', 'type', 'make']:
        #     self.top_10_labels[feature] = (
        #         X[feature].value_counts().nlargest(10).index.tolist()
        #     )
        X_transformed = self._transform_logic(X.copy())
        self.feature_names_out_ = X_transformed.columns.tolist()
        return self
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns)
    
        return self._transform_logic(X, use_saved_top_labels=True)
    def _transform_logic(self, X, use_saved_top_labels=False):
        
       
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
            X.drop(feature, axis=1, inplace=True)
       
        title_rank = {
            'clean': 0,
            'rebuilt': 1,
            'salvage': 2,
            'lien': 3,
            'parts only': 4,
            'missing': 5
        }
        X['title_status'] = X['title_status'].map(title_rank)
        
        condition_rank = {
            'new':1,
            'like new':2,
            'excellent':3,
            'good':4,
            'fair':5,
            'salvage':6,
            'Missing':7
        }
        X['condition'] = X['condition'].map(condition_rank)
        return X
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
    
class OutlierRemover(BaseEstimator, TransformerMixin): #Removes 196 instances
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        # print(f'shape of outlierremover input: {X.shape}')
        # outlier_count = (X["num_pipeline__outlier_flag"] == 'outlier').sum()
        # print(f"Number of outliers: {outlier_count}")
        X = X[X["num_pipeline__outlier_flag"] != 'outlier'].drop(columns=["num_pipeline__outlier_flag"]).reset_index(drop=True)

        self.columns_ = X.columns.tolist()
        #X['odometer'], parameters = stats.boxcox(X['odometer'])

        # df = pd.DataFrame(X)
        # cleaned_df = df[~df.isin(['outlier']).any(axis=1)].reset_index(drop=True)
        # cleaned_df = cleaned_df.drop(columns=[col for col in cleaned_df.columns if 'non-outlier' in cleaned_df[col].values])
        #print(X.shape)
        return X#cleaned_df.values
    def get_feature_names_out(self, input_features=None):
        return self.columns_

class DropColumn(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        return X.drop(columns=self.cols, axis=1, errors='ignore')
    