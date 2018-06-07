
# Observable Trends

- Most tweets are concentrated at a polarity of 0, meaning that media sources aim to report in a neutral fashion.
- Per the data on 06/07/2018, all media sources were overall tweeting negatively.
- Per the data on 06/07/2018, CNN was tweeting most neutral while CBSNews was tweeting most negatively.


```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Account
target_users = ("BBCWorld","CBSNews", "CNN", "FoxNews", "nytimes")

sentiments = []

for target in target_users:
    
    # Counter
    counter = 1
    
    # Variable for max_id
    oldest_tweet = None

    # Loop through 5 pages of tweets (total 100 tweets)
    for x in range(5):

            # Get all tweets from home feed
            public_tweets = api.user_timeline(target,  max_id = oldest_tweet)

            # Loop through all tweets
            for tweet in public_tweets:

                # Run Vader Analysis on each tweet
                results = analyzer.polarity_scores(tweet["text"])
                compound = results["compound"]
                pos = results["pos"]
                neu = results["neu"]
                neg = results["neg"]
                tweets_ago = counter
                
                # Get Tweet ID, subtract 1, and assign to oldest_tweet
                oldest_tweet = tweet['id'] - 1

                # Add sentiments for each tweet into a list
                sentiments.append({"Source Account": target,
                           "Text": tweet["text"],
                           "Date": tweet["created_at"],
                           "Positive": pos,
                           "Neutral": neu,
                           "Negative": neg,
                           "Compound": compound,
                            "Tweets Ago": counter})
                
                # Add to counter 
                counter += 1
```


```python
# Convert sentiments to DataFrame
sentiments_pd = pd.DataFrame(sentiments)
sentiments_pd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Source Account</th>
      <th>Text</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.7650</td>
      <td>Thu Jun 07 15:32:15 +0000 2018</td>
      <td>0.524</td>
      <td>0.476</td>
      <td>0.000</td>
      <td>BBCWorld</td>
      <td>Kate Spade death: Mental illness 'doesn't disc...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.4404</td>
      <td>Thu Jun 07 15:17:30 +0000 2018</td>
      <td>0.131</td>
      <td>0.611</td>
      <td>0.258</td>
      <td>BBCWorld</td>
      <td>Silent alert system for women who are being at...</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>Thu Jun 07 15:11:02 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBCWorld</td>
      <td>Korean woman survives six days in Australian w...</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.7783</td>
      <td>Thu Jun 07 15:11:00 +0000 2018</td>
      <td>0.531</td>
      <td>0.469</td>
      <td>0.000</td>
      <td>BBCWorld</td>
      <td>Israel blames Iran for Gaza border violence ht...</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.7003</td>
      <td>Thu Jun 07 14:45:00 +0000 2018</td>
      <td>0.492</td>
      <td>0.508</td>
      <td>0.000</td>
      <td>BBCWorld</td>
      <td>Ship hack 'risks chaos in English Channel' htt...</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Reorganize columns
# source acount, its text, its date, and its compound, positive, neutral, and negative sentiment scores
sentiments_pd_reorg = sentiments_pd[["Source Account", "Text", "Date", "Compound", "Positive", "Neutral",
                                    "Negative", "Tweets Ago"]]
sentiments_pd_reorg.head()

# Export to csv
sentiments_pd_reorg.to_csv("output/Twitter_sentiments_news.csv",
                     encoding="utf-8")
```


```python
sns.set(style="darkgrid")
scatter_plot = sns.lmplot(x = "Tweets Ago", y = "Compound", data = sentiments_pd_reorg,
                         hue= "Source Account", palette=dict(BBCWorld="lightcoral", 
                            CBSNews="green", CNN="red", FoxNews='blue', nytimes='yellow'), 
                         fit_reg=False, legend=False)
scatter_plot = (scatter_plot.set(xlim=(100, 0), ylim=(-1.0, 1.0)))

# Incorporate the other graph properties
now = datetime.now()
now = now.strftime("%m/%d/%Y")
plt.title(f"Sentiment Analysis of Media Tweets ({now})")
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.grid(True)

# Create a legend
lgnd = plt.legend(fontsize="medium", loc='upper center', bbox_to_anchor=(1.2, 0.8), title="Media Sources")

# Save Figure
plt.savefig("output/SentimentAnalysisScatter.png")

# Show plot
plt.show()
```


![png](output_7_0.png)



```python
account_compound_mean = sentiments_pd_reorg.groupby(["Source Account"]).mean()["Compound"]
account_compound_mean
```




    Source Account
    BBCWorld   -0.100531
    CBSNews    -0.149098
    CNN        -0.021505
    FoxNews    -0.048664
    nytimes    -0.089586
    Name: Compound, dtype: float64




```python
plt.bar(target_users, account_compound_mean, color = ['lightcoral','g','r','b','y'], alpha=1, align="center")

# Tell matplotlib where to place each of the x-axis labels
tick_locations = [value for value in target_users]
plt.xticks(tick_locations, target_users)

# Set the x-limits
plt.xlim(-0.75, len(target_users)-0.25)

# Set the y-limits
plt.ylim(min(account_compound_mean)-.025), (max(account_compound_mean)+.05)

# Give our chart some labels and a tile
plt.title(f"Overall Media Sentiment based on Twitter ({now})")
plt.ylabel("Tweet Polarity")

# Save Figure
plt.savefig("output/SentimentAnalysisOverall.png")

# Print our chart to the screen
plt.show()
```


![png](output_9_0.png)

