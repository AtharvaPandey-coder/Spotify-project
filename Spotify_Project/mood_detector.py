from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)


analyzer=SentimentIntensityAnalyzer()


def mood_detect_vader(user_text,df,top_n=5):
    scores=analyzer.polarity_scores(user_text)
    compound=scores['compound']
    pos=scores['pos']
    neg=scores['neg']

    print(f"scores:{scores}")

    # Map to mood
    if compound >= 0.3:
        mood='Happy'
    elif compound <= -0.3 and pos < 0.1:
        mood='Sad'
    elif compound <= -0.1 and pos > 0.1:
        mood='Angry'
    else:
        mood='Calm'
    
    confidence = round(abs(compound) * 100,1)

    filtered=df[df['mood'] == mood]
    if len(filtered) < top_n:
        filtered=df
    results=filtered.sample(
        min(top_n,len(filtered))
    )[['track_name','track_artist','mood']]
    return results,mood,confidence

