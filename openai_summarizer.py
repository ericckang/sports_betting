import openai
from api_keys import OPENAI_API_KEY

class OpenAISummarizer:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def summarize(self, team1, team2, sentiment_team1, sentiment_team2, prediction):
        # prediction is now a list: [team1_prob, team2_prob]
        prompt = f"""
Teams: {team1} vs {team2}
Team 1 Sentiment: {sentiment_team1}
Team 2 Sentiment: {sentiment_team2}
Predicted Probability of {team1} winning: {prediction[0]*100:.2f}%
Predicted Probability of {team2} winning: {prediction[1]*100:.2f}%

Provide a concise summary and which team seems like the best bet:
"""
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        return response.choices[0].text.strip()
