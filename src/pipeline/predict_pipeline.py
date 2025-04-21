import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from datetime import datetime


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,df):
        try:
            
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")


            data_scaled=preprocessor.transform(df)
            preds=model.predict(data_scaled)

            df = df.loc[data_scaled.index]
            df['pred_price'] = preds

            
            # df['residual'] = df['pred_price'] - df['price']
            # df = df[df['residual'] > 0].sort_values(by='residual', ascending=False)
            df = df[df['pred_price'] > 0].sort_values(by='pred_price', ascending=False)

            return df[["model", "pred_price", "url"]].to_dict(orient="records")
        
        except Exception as e:
            raise CustomException(e,sys)
class EditData:
    def __init__(self):
        pass
    def changes(self,path):
        try:
            X = pd.read_csv(path)
            current_year = datetime.now().year

            X['age'] = current_year - X['year']
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
        except Exception as e:
            raise CustomException(e,sys)

