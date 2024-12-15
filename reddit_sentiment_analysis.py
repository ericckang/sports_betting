import praw
from textblob import TextBlob
from datetime import datetime, timedelta
import concurrent.futures
from api_keys import (REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
                      REDDIT_USER_AGENT, REDDIT_USERNAME, REDDIT_PASSWORD)

class RedditSentimentAnalyzer:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            username=REDDIT_USERNAME,
            password=REDDIT_PASSWORD
        )

    def _analyze_post_sentiment(self, submission, three_months_ago):
        submission_time = datetime.fromtimestamp(submission.created_utc)
        if submission_time > three_months_ago:
            score = TextBlob(submission.title).sentiment.polarity
            return score
        return 0

    def get_team_sentiment(self, subreddit_name, team_name, limit=50):
        three_months_ago = datetime.utcnow() - timedelta(days=90)
        subreddit = self.reddit.subreddit(subreddit_name)
        submissions = [s for s in subreddit.search(team_name, limit=limit)
                       if datetime.fromtimestamp(s.created_utc) > three_months_ago]

        if not submissions:
            return 0.0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            scores = list(executor.map(lambda s: self._analyze_post_sentiment(s, three_months_ago), submissions))

        scores = [sc for sc in scores if sc != 0]
        if len(scores) == 0:
            return 0.0
        return sum(scores) / len(scores)
