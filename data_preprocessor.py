import random
import numpy as np

class DataPreprocessor:
    def __init__(self, teams):
        self.team_rankings = {team: random.randint(1,30) for team in teams}
        self.team_win_streaks = {team: random.randint(0,10) for team in teams}
        self.teams = teams

    def create_feature_vector(self, team1, team2, sentiment_t1, sentiment_t2):
        ranking_t1 = self.team_rankings.get(team1, 15)
        ranking_t2 = self.team_rankings.get(team2, 15)
        win_streak_t1 = self.team_win_streaks.get(team1, 0)
        win_streak_t2 = self.team_win_streaks.get(team2, 0)
        sentiment_diff = sentiment_t1 - sentiment_t2
        features = np.array([sentiment_t1, sentiment_t2, sentiment_diff, ranking_t1, ranking_t2, win_streak_t1, win_streak_t2])
        return features.reshape(1, -1)
