# main.py
from reddit_sentiment_analysis import RedditSentimentAnalyzer
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from api_keys import XGBOOST_MODEL_PATH

if __name__ == "__main__":
    teams = ["lakers", "warriors", "celtics", "nets", "bucks", "heat", "suns", "clippers"]

    sentiment_fetcher = RedditSentimentAnalyzer()
    preprocessor = DataPreprocessor(teams)
    trainer = ModelTrainer(preprocessor, sentiment_fetcher, XGBOOST_MODEL_PATH)

    # Generate dummy data and train the model
    X, y = trainer.generate_dummy_data(n_samples=20)  # Smaller for demo
    model = trainer.train_model(X, y)

    print("Model training complete. Model saved to:", XGBOOST_MODEL_PATH)
