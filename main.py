from reddit_sentiment_analysis import RedditSentimentAnalyzer
#from xgboost_predictor import XGBoostPredictor
#from openai_summarizer import OpenAISummarizer


def main():
    # Step 1: Sentiment Analysis
    analyzer = RedditSentimentAnalyzer()
    overall_sentiment = analyzer.get_sentiment_scores("nba", "warriors", limit=100)
    if overall_sentiment is not None:
        print(f"Overall Sentiment Score: {overall_sentiment}")
    else:
        print("No relevant posts found in the specified timeframe.")

    # Step 2: XGBoost Prediction (assuming sentiment_score affects predictions)
    #xgb_predictor = XGBoostPredictor('path/to/model.bin')
    #prediction_results = xgb_predictor.predict(sentiment_score)

    # Step 3: Summarization with OpenAI
    #summarizer = OpenAISummarizer(api_key="your_openai_api_key")
    #summary = summarizer.summarize(sentiment_score, prediction_results)

    # Output the best bet or summary
    #print("Summary:", summary)


if __name__ == "__main__":
    main()