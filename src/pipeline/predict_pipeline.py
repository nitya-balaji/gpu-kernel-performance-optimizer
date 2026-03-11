import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object 

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features): #load respective .pkl files
        try:
            model_path="artifacts/model.pkl"
            preprocessor_path="artifacts/preprocessor.pkl"
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            #scaling the data inputted by user
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled) #model looks at scaled values from above line and provides a value for runtime
            return preds #return the answer back to the website
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 MWG:float, NWG:float, KWG:float, MDIMC:float,
               NDIMC:float, MDIMA:float, NDIMB:float, KWI:float,
               VWM:float, VWN:float, STRM:float, STRN:float, SA:float, SB:float):
        self.MWG=MWG
        self.NWG = NWG
        self.KWG=KWG
        self.MDIMC=MDIMC
        self.NDIMC=NDIMC
        self.MDIMA=MDIMA
        self.NDIMB=NDIMB
        self.KWI=KWI
        self.VWM=VWM
        self.VWN=VWN
        self.STRM=STRM
        self.STRN=STRN
        self.SA=SA
        self.SB=SB
    
    
    def get_data_as_data_frame(self):
        try:
            #maps variables to exact column names the model saw during training
            custom_data_input_dict = {
                "MWG": [self.MWG],
                "NWG": [self.NWG],
                "KWG": [self.KWG],
                "MDIMC":[self.MDIMC],
                "NDIMC":[self.NDIMC],
                "MDIMA":[self.MDIMA],
                "NDIMB":[self.NDIMB],
                "KWI":[self.KWI],
                "VWM":[self.VWM],
                "VWN":[self.VWN],
                "STRM":[self.STRM],
                "STRN":[self.STRN],
                "SA":[self.SA],
                "SB":[self.SB],
            }
            return pd.DataFrame(custom_data_input_dict) #turns dictionary into 1-row dataframe (aka a table)
        
        except Exception as e:
            raise CustomException(e,sys)
            
        