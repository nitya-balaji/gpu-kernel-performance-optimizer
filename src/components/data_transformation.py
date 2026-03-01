import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging 
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    #paths/inputs that we will need 
    preprocessor_object_file_path=os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
       #create all .pkl files (responsible for feature engineering, feature selection, and data cleaning) 
       try:
           #state numerical features
           numerical_columns=[
               "MWG", "NWG", "KWG", "MDIMC",
               "NDIMC", "MDIMA", "NDIMB", "KWI",
               "VWM", "VWN", "STRM", "STRN", "SA", "SB"   
           ]
           #create pipeline
           num_pipeline = Pipeline(
            steps=[
            ("scaler", StandardScaler()) 
            ]  
           )
           
           logging.info(f"Numerical columns: {numerical_columns}")
           
           #bring the num_pipeline and numerical_columns together through ColumnTransformer (doesn't execute - just blueprint -> fit_transform() does the actual execution and returns an array)
           preprocessor= ColumnTransformer(
               [
               ("num_pipeline", num_pipeline, numerical_columns)
              ]                             
            )
           
           
           return preprocessor
       except Exception as e:
           raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path): 
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Train and test data have been read")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="Runtime"
            numerical_columns=[
               "MWG", "NWG", "KWG", "MDIMC",
               "NDIMC", "MDIMA", "NDIMB", "KWI",
               "VWM", "VWN", "STRM", "STRN", "SA", "SB"   
           ]
            
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

                file_path=self.data_transformation_config.preprocessor_object_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)