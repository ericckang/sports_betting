import praw
from textblob import TextBlob
from datetime import datetime, timedelta
import concurrent.futures


class RedditSentimentAnalyzer:
    def __init__(self):
        # Store API credentials as class attributes
     

        # Initialize the Reddit instance within the class
        self.reddit = praw.Reddit(client_id=self.client_id,
                                  client_secret=self.client_secret,
                                  user_agent=self.user_agent,
                                  username=self.username,
                                  password=self.password)

    def analyze_post_sentiment(self, submission, three_months_ago):
        submission_time = datetime.fromtimestamp(submission.created_utc)
        if submission_time > three_months_ago:
            analysis = TextBlob(submission.title).sentiment.polarity
            return analysis
        return 0

    def get_sentiment_scores(self, subreddit_name, team_name, limit=None):
        subreddit = self.reddit.subreddit(subreddit_name)
        three_months_ago = datetime.utcnow() - timedelta(days=90)
        submissions = [submission for submission in subreddit.search(team_name, limit=limit)
                       if datetime.fromtimestamp(submission.created_utc) > three_months_ago]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            sentiment_scores = list(
                executor.map(lambda submission: self.analyze_post_sentiment(submission, three_months_ago), submissions))

        sentiment_scores = [score for score in sentiment_scores if score != 0]
        if sentiment_scores:
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        else:
            overall_sentiment = None

        return overall_sentiment
