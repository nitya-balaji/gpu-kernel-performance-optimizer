import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], #taking every column except last one (X_train)
                train_array[:, -1], #taking last column (y_train)
                test_array[:, :-1], #same applies for X_test
                test_array[:, -1] #same applies for y_test
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, allow_writing_files=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1],
                    'subsample': [0.6],
                    'n_estimators': [8, 16]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1],
                    'n_estimators': [8, 16]
                },
                "CatBoosting Regressor": {
                    'depth': [6],
                    'learning_rate': [0.1],
                    'iterations': [30]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1],
                    'n_estimators': [8, 16]
                }
            }
            #this will give us am R^2 score for each model (using the evaluate_models function)
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            #get best model score and name
            best_model_score = max(model_report.values())
            #to get the name of the best model, we find the index (position) of the highest 
            #score in the values list and use that same position to grab the corresponding 
            #name from the keys list.
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            #grab the specific model instance that was just trained and achieved the highest score (this isn't just a score anymore - it is a trained object)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")

            logging.info(f"Best model found: {best_model_name} with R2 Score: {best_model_score}")

            #save best_model object as .pkl file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model 
            )

            predicted = best_model.predict(X_test) #list of all predictions made by the model for the X_test data set
            r2_square = r2_score(y_test, predicted) #comparing the actual runtimes with the predicted ones (and returning that comparison score - in next line)
            
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)