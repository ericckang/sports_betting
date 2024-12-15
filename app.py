from flask import Flask, request, jsonify
from reddit_sentiment_analysis import RedditSentimentAnalyzer
from xgboost_predictor import XGBoostPredictor
from openai_summarizer import OpenAISummarizer
from data_preprocessor import DataPreprocessor

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    team1 = request.args.get('team1', default="lakers", type=str)
    team2 = request.args.get('team2', default="warriors", type=str)

    # Sentiment Analysis
    analyzer = RedditSentimentAnalyzer()
    sentiment_team1 = analyzer.get_team_sentiment("nba", team1, limit=100)
    sentiment_team2 = analyzer.get_team_sentiment("nba", team2, limit=100)

    # If no sentiment found, default to 0.0
    sentiment_team1 = sentiment_team1 if sentiment_team1 is not None else 0.0
    sentiment_team2 = sentiment_team2 if sentiment_team2 is not None else 0.0

    # Prepare data for prediction
    all_teams = ["lakers", "warriors", "celtics", "nets", "bucks", "heat", "suns", "clippers"]
    preprocessor = DataPreprocessor(all_teams)
    input_data = preprocessor.create_feature_vector(team1, team2, sentiment_team1, sentiment_team2)

    # Predict probability of team1 winning
    predictor = XGBoostPredictor()
    prediction = predictor.predict(input_data)
    team1_prob = float(prediction[0])
    team2_prob = 1.0 - team1_prob

    # Summarize results
    summarizer = OpenAISummarizer()
    summary = summarizer.summarize(team1, team2, sentiment_team1, sentiment_team2, [team1_prob, team2_prob])

    # Return JSON response
    response = {
        "team1": team1,
        "team2": team2,
        "team1_sentiment": sentiment_team1,
        "team2_sentiment": sentiment_team2,
        "predicted_probability_team1": team1_prob,
        "predicted_probability_team2": team2_prob,
        "suggested_bet_summary": summary
    }

    return jsonify(response)

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
