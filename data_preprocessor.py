# data_preprocessor.py
# In this class, we’ll create features from the raw inputs. 
# More features can be added as needed: 
# For example: sentiment scores, team rankings, win streaks, sentiment differences, etc.
import random
import numpy as np

class DataPreprocessor:
    def __init__(self, teams):
        # We might store references like team rankings in a dictionary.
        # In a real scenario, load these from a database or a file.
        self.team_rankings = {team: random.randint(1,30) for team in teams}
        self.team_win_streaks = {team: random.randint(0,10) for team in teams}
        self.teams = teams

    def create_feature_vector(self, team1, team2, sentiment_t1, sentiment_t2):
        # Additional features
        ranking_t1 = self.team_rankings.get(team1, 15)  # default
        ranking_t2 = self.team_rankings.get(team2, 15)
        win_streak_t1 = self.team_win_streaks.get(team1, 0)
        win_streak_t2 = self.team_win_streaks.get(team2, 0)

        # Example features:
        # [sentiment_t1, sentiment_t2, sentiment_diff, ranking_t1, ranking_t2, win_streak_t1, win_streak_t2]
        sentiment_diff = sentiment_t1 - sentiment_t2
        features = np.array([sentiment_t1, sentiment_t2, sentiment_diff, ranking_t1, ranking_t2, win_streak_t1, win_streak_t2])
        return features.reshape(1, -1)
