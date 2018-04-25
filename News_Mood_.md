

```python
#Dependencies
import tweepy
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
```


```python
#Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Twitter API Keys
from config import consumer_key, consumer_secret, access_token, access_token_secret
consumer_key = consumer_key
consumer_secret = consumer_secret
access_token = access_token
access_token_secret = access_token_secret
```


```python
#Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Search Term
news_orgs = ["@BBC", "@CBS", "@CNN", "@FoxNews", "@nytimes"]
results_list = []

for org in news_orgs:
    # Create empty sentiment lists
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    index = []
    count = 0 
    
    # Grab the most recent 100 tweets
    public_tweets = api.search(org, count=100, result_type="recent")
    
        # Loop
    for tweet in public_tweets["statuses"]:

        # Enable Vader Analyzer
        compound = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
    
        # Adding to the sentiment list
        compound_list.append(compound)
        positive_list.append(pos)
        negative_list.append(neg)
        neutral_list.append(neu)
        index.append(count)
        count = count + 1
        
    lgnd = plt.legend(news_orgs, title= 'News Organizations', bbox_to_anchor=(1, 0.75))
    plt.scatter(index, compound_list, facecolors=['red','blue','purple','green','yellow'], edgecolors="black", linewidth=1, marker='o', alpha=0.8)
    plt.title("Sentiment Analysis of Media Tweets")
    plt.xlabel("Tweets Ago")
    plt.ylabel("Tweet Polarity")
    plt.legend
    plt.xlim(len(compound_list), 0)
    plt.yticks([-1, -.5, 0, .5, 1])
        
# Store the Average Sentiments
    sentiment = {"News Organization": org,
                 "Compound": np.mean(compound_list),
                 "Positive": np.mean(positive_list),
                 "Negative": np.mean(negative_list),
                 "Neutral": np.mean(neutral_list)}
    results_list.append(sentiment)

# Save the plot
plt.savefig("plot1.png")
#Show the plot
plt.show()

```


![png](output_4_0.png)



```python
compounds = []
for sentiment in results_list:
    compounds.append(sentiment["Compound"])
    
plt.title("Overall Media Sentiment Based on Twitter")
plt.ylabel("Tweet Polarity")
    
plt.bar(news_orgs, compounds)

# Save the plot
plt.savefig("plot2.png")
#Show the plot
plt.show()
```


![png](output_5_0.png)



```python
news_orgs_df = pd.DataFrame(results_list).set_index('News Organization').round(3).reset_index()
news_orgs_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>News Organization</th>
      <th>Compound</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBC</td>
      <td>0.006</td>
      <td>0.067</td>
      <td>0.856</td>
      <td>0.077</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@CBS</td>
      <td>0.158</td>
      <td>0.057</td>
      <td>0.845</td>
      <td>0.098</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@CNN</td>
      <td>0.069</td>
      <td>0.059</td>
      <td>0.842</td>
      <td>0.098</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@FoxNews</td>
      <td>0.002</td>
      <td>0.067</td>
      <td>0.864</td>
      <td>0.069</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@nytimes</td>
      <td>-0.112</td>
      <td>0.104</td>
      <td>0.823</td>
      <td>0.073</td>
    </tr>
  </tbody>
</table>
</div>




```python
news_orgs_df.to_csv("NewsMood.csv", encoding='utf-8', index=False)

```
