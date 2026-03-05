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
    """
    Configuration class to define the destination for the saved preprocessor.
    We save this as a .pkl file so our web app can use the exact same scaling later.
    """ 
    preprocessor_object_file_path=os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        #initialize the config to have access to the file path
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
       """
       Builds the 'Blueprint' for data transformation. 
       This defines WHICH columns to transform and WHAT math to apply (Scaling).
       """
       try:
           #state numerical features that need to be standardized
           numerical_columns=[
               "MWG", "NWG", "KWG", "MDIMC",
               "NDIMC", "MDIMA", "NDIMB", "KWI",
               "VWM", "VWN", "STRM", "STRN", "SA", "SB"   
           ]
           #create pipeline to automate scaling process
           num_pipeline = Pipeline(
            steps=[
            ("scaler", StandardScaler()) 
            ]  
           )
           
           logging.info(f"Numerical columns: {numerical_columns}")
           
           #ColumnTransformer applies the num_pipeline ONLY to the numerical_columns (doesn't execute yet - these are just instructions for the model)
           preprocessor= ColumnTransformer(
               [
               ("num_pipeline", num_pipeline, numerical_columns)
              ]                             
            )
           return preprocessor
       except Exception as e:
           raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path): 
        """
        Executes the transformation. This reads the raw CSVs, applies the scaling,
        and bundles the data into numpy arrays for the Model Trainer.
        """
        try:
            #load the split datasets from the artifacts folder
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Train and test data have been read")
            
            logging.info("Obtaining preprocessing object")
            
            #get the 'Blueprint' we built in get_data_transformer_object
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="Runtime"
            numerical_columns=[
               "MWG", "NWG", "KWG", "MDIMC",
               "NDIMC", "MDIMA", "NDIMB", "KWI",
               "VWM", "VWN", "STRM", "STRN", "SA", "SB"   
           ]
            #separate Features (X) from Target (y)
            #we drop the answer ('Runtime') from X bc we shouldn't scale the target value and also prevents data leakage (which can influence scaling).
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            #perform the Scaling
            #fit_transform: Learns the Mean/StdDev from training data AND scales it.
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            #transform: Scales the test data using the stats learned from the training data.
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df) #scaled training array
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] #in model_trainer we should slice the features and runtime (to avoid cheating and memorization)

            logging.info(f"Saved preprocessing object.")

            #save the "Preprocessor Machine"
            #we save the 'obj' as a .pkl file so we don't lose the learned scaling rules.
            save_object(

                file_path=self.data_transformation_config.preprocessor_object_file_path,
                obj=preprocessing_obj

            )
            #return the ready-to-use arrays and the path to the saved pkl file
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)