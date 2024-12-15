# model_trainer.py
# This class handles generating training data, training the model, and saving it.
import random
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

class ModelTrainer:
    def __init__(self, data_preprocessor, sentiment_fetcher, model_path):
        self.data_preprocessor = data_preprocessor
        self.sentiment_fetcher = sentiment_fetcher
        self.model_path = model_path

    def generate_dummy_data(self, n_samples=50):
        # Generate synthetic historical matches.
        data = []
        for _ in range(n_samples):
            team1, team2 = random.sample(self.data_preprocessor.teams, 2)
            sentiment_t1 = self.sentiment_fetcher.get_team_sentiment("nba", team1, limit=5)
            sentiment_t2 = self.sentiment_fetcher.get_team_sentiment("nba", team2, limit=5)

            # Probability of team1 winning influenced by sentiment difference
            p_team1_win = 0.5 + (sentiment_t1 - sentiment_t2) * 0.2
            p_team1_win = max(0, min(1, p_team1_win))
            winner = 1 if random.random() < p_team1_win else 0

            features = self.data_preprocessor.create_feature_vector(team1, team2, sentiment_t1, sentiment_t2)
            data.append((features[0], winner))  # features is (1, n), we store as 1D

        X = np.array([row[0] for row in data])
        y = np.array([row[1] for row in data])
        return X, y

    def train_model(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.1,
            "max_depth": 3,
            "seed": 42
        }

        evals = [(dtrain, "train"), (dval, "eval")]
        model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10)

        y_val_pred = model.predict(dval)
        val_logloss = log_loss(y_val, y_val_pred)
        print(f"Validation Logloss: {val_logloss:.4f}")

        model.save_model(self.model_path)
        return model
