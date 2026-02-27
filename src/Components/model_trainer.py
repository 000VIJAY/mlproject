from dataclasses import dataclass
import os
import sys

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

from src.Components.data_ingestion import DataIngestion
from src.Components.data_transformation import DataTransformation
from src.logger import logger
from src.exception import CustomException

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("[ModelTrainer] Starting model trainer execution.")
            logger.info("Splitting training and testing input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            logger.info(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
            logger.info("Splitting training and testing input data completed")

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "K-Nearest Neighbors": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor()
            }
            logger.info(f"Models initialized: {list(models.keys())}")
            logger.info("Calling evaluate_models function.")
            model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            logger.info(f"Model report: {model_report}")
            # Find the best model based on the highest Test Score
            best_model_name, best_model_scores = max(model_report.items(), key=lambda item: item[1]['Test Score'])
            base_model_score = best_model_scores['Test Score']
            logger.info(f"Best model score: {base_model_score}")
            logger.info(f"Best model name: {best_model_name}")
            best_model = models[best_model_name]

            if base_model_score < 0.6:
                logger.error("No best model found. Score below threshold.")
                raise CustomException("No best model found", sys)

            logger.info(f"Saving best model to {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logger.info("Predicting with best model.")
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logger.info(f"R2 Score: {r2_square}")
            return r2_square

        except Exception as e:
            logger.error(f"Exception occurred in ModelTrainer: {e}")
            raise CustomException(e, sys)   


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data, test_data) 
    model_trainer = ModelTrainer()
    r2_square = model_trainer.initiate_model_trainer(train_arr, test_arr)