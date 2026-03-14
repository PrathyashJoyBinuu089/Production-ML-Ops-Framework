"""
Training Pipeline for MLOps Production Framework.
Author: Senior ML Engineer
"""
import os
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple

from loguru import logger
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class TrainingPipeline:
    """
    A modular training pipeline that handles data loading, preprocessing, 
    training, evaluation, and model serialization.
    """

    def __init__(self, model_name: str = "iris_classifier", model_dir: str = "models"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Initialized training pipeline. Model will be saved to {self.model_dir}")

    def load_data(self) -> Tuple[Any, Any]:
        """Loads the Iris dataset."""
        logger.info("Loading dataset...")
        iris = load_iris()
        return iris.data, iris.target

    def split_data(self, X: Any, y: Any) -> Tuple[Any, Any, Any, Any]:
        """Splits the dataset into training and testing sets."""
        logger.info("Splitting data into train and test sets...")
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train: Any, y_train: Any) -> RandomForestClassifier:
        """Trains the Random Forest model."""
        logger.info("Training Random Forest Classifier...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        return clf

    def evaluate_model(self, model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
        """Evaluates the model on the test set."""
        logger.info("Evaluating model performance...")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy, "report": report}

    def save_artifact(self, model: Any, metrics: Dict[str, Any]) -> str:
        """Saves the model artifact and metadata."""
        model_filename = f"{self.model_name}_{self.timestamp}.joblib"
        latest_filename = f"{self.model_name}_latest.joblib"
        
        model_path = os.path.join(self.model_dir, model_filename)
        latest_path = os.path.join(self.model_dir, latest_filename)
        
        # Save timestamped model
        joblib.dump(model, model_path)
        logger.info(f"Model artifact saved to {model_path}")
        
        # Symlink or copy to 'latest'
        joblib.dump(model, latest_path)
        logger.info(f"Latest model updated at {latest_path}")
        
        # In a real enterprise scenario, we'd log metrics to MLflow or similar
        logger.info(f"Final Metrics: {metrics['accuracy']}")
        
        return model_path

    def run(self):
        """Executes the full training pipeline."""
        try:
            X, y = self.load_data()
            X_train, X_test, y_train, y_test = self.split_data(X, y)
            
            model = self.train_model(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            
            artifact_path = self.save_artifact(model, metrics)
            logger.info("Training pipeline completed successfully.")
            return artifact_path
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Define paths relative to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(base_dir, "models")
    
    pipeline = TrainingPipeline(model_dir=models_dir)
    pipeline.run()
