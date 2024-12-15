import xgboost as xgb
from api_keys import XGBOOST_MODEL_PATH


class XGBoostPredictor:
    def __init__(self, model_path=XGBOOST_MODEL_PATH):
        self.model = self.load_model(model_path)

    def load_model(self, path):
        model = xgb.Booster()
        model.load_model(path)
        return model

    def preprocess_data(self, data):
        # data is expected to be in a numeric format (list or numpy array)
        dmatrix = xgb.DMatrix(data)
        return dmatrix

    def predict(self, data):
        preprocessed_data = self.preprocess_data(data)
        predictions = self.model.predict(preprocessed_data)
        return predictions
