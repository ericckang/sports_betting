# Reddit Sentiment-Based Betting Predictor

This project provides a workflow and codebase for pulling sentiment data from Reddit posts about NBA teams, using that sentiment to train and serve predictions via a Flask web API. The model predicts a team's probability of winning based on sentiment and other features. It also integrates with OpenAI to summarize results and provide a recommended bet.

## Features

- Fetch and analyze team sentiment from Reddit using PRAW and TextBlob.
- Process features such as sentiment, team rankings, and win streaks.
- Train an XGBoost model to predict a team's winning probability.
- Use OpenAI's API to generate a summary and suggestion.
- Serve predictions as a JSON API via Flask.

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`:
  - `flask`
  - `praw`
  - `textblob`
  - `xgboost`
  - `scikit-learn`
  - `openai`
  - `numpy`
  - `concurrent.futures` (part of Python standard library)
  
Install these using:

```pip install -r requirements.txt```


## Setup

1. Copy `api_keys.py` and update with your own credentials:
   - Reddit API keys (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, etc.)
   - OpenAI API key (`OPENAI_API_KEY`)
   - The path to the XGBoost model file (`XGBOOST_MODEL_PATH`), for example: `xgboost_model.bin`

2. Run `main.py` to:
   - Generate dummy training data.
   - Train the XGBoost model.
   - Save the model to `XGBOOST_MODEL_PATH`.


## Running the App

Once the model is trained and the `xgboost_model.bin` file is generated, run:

```python app.py```

The Flask app will start on `http://0.0.0.0:5000`.

## Making Predictions

Send a GET request to:

```http://localhost:5000/predict?team1=x&team2=y```

This returns a JSON response containing:
- Team 1 and Team 2 sentiments
- Predicted probabilities
- A summarized betting suggestion from OpenAI

## Notes

- The sentiment data comes from the last 3 months of Reddit posts mentioning the team.
- The model is trained on dummy data for demonstration. For real accuracy, integrate historical match outcomes and more robust training data.

## Contributing

Feel free to modify and enhance:
- Add more features to `DataPreprocessor`.
- Integrate real historical data for training.
- Experiment with other ML models or parameter tuning.

