import tweepy
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt

# Connect to twitter
consumer_key = 	''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#set up search word
search_word = "Charlottesville"
public_tweets = api.search(search_word)

# lists to store sentiment analysis results
polarity = []
subjectivity = []

def get_sentiment():
	"""Append polarity and subjectivity to lists"""
	for tweet in public_tweets:
	    analysis = TextBlob(tweet.text)
	    polarity.append(float(analysis.sentiment.polarity))
	    subjectivity.append(float(analysis.sentiment.subjectivity))
	return 
#Lists to store the results of rank_sentiment function
positive_list = []
negative_list = []
neutral_list = []

def rank_sentiment(sentiment_list):
	"""rank the results of polarity list"""
	for i in sentiment_list:
		if i > 0:
			positive_list.append(i)
		elif i == 0:
			neutral_list.append(i)
		else:
			negative_list.append(i)
		
get_sentiment()
rank_sentiment(polarity)

#print mean subjectivity and polarity for reference
print(np.mean(polarity), np.mean(subjectivity))

#create pie chart
slices = [len(positive_list),len(negative_list),len(neutral_list)]
activities = ['Positive','negative', 'neutral']
cols = ['c', 'm','r',]
plt.pie(slices, 
	labels= activities, 
	colors = cols, 
	shadow=True,
	explode=(0,0.1,0),
	autopct='%1.1f%%')
plt.title('%s Sentiment Analysis' % search_word)
plt.show()
