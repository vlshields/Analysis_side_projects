
# Twitter Sentiment Analysis

The file twitter_sentiment.py uses python's tweepy library to search
twitter for a key word, and then uses python's textblob library to rank
the tweet's polarity and subjectivity. Laslty, matplotlib is used to create
pie chart based on the positivity, negativity, and neutrality of the tweets
returned by the search word.


## How to use the program

1. Install the required libraries.

  ```
  pip install -r requirements.txt
  ```

  Note: On my ubuntu laptop, I had to install `python3-tk`.

  ```
  sudo apt install python3-tk
  ```

2. Add your twitter API credentials by modifying `CONSUMER_KEY`, `CONSUMER_SECRET`, `ACCESS_TOKEN` and `ACCESS_TOKEN_SECRET` in `twitter_sentiment.py`. If you don't have any credentials, you can create them by [creating a twitter application](https://apps.twitter.com/app/new).

3. Change the search_word in twitter_sentiment.py variable to any word/phrase that you want a sentiment analysis of.

 ```
  search_word = "word"
```


3. Run the program.
  ```
  python3 twitter_sentiment.py
  ```
