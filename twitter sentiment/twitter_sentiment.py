import tweepy, csv
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


consumer_key = 	''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# lists to store sentiment analysis results


def get_sentiment(tweets):
	"""Return polarity and subjectivity from a list of tweets"""
	subjectivity = []
	polarity = []
    

	for tweet in tweets:
	    analysis = TextBlob(tweet.text)
	    polarity.append(float(analysis.sentiment.polarity))
	    subjectivity.append(float(analysis.sentiment.subjectivity))
	return polarity, subjectivity
#Lists to store the results of rank_sentiment function


def rank_sentiment(sentiment_list):
	"""rank the results of polarity list"""
	neutral_list = []
	negative_list = []
	positive_list = []
    

	for i in sentiment_list:
		if i > 0:
			positive_list.append(i)
		elif i == 0:
			neutral_list.append(i)
		else:
			negative_list.append(i)

	return positive_list, negative_list, neutral_list

def write_to_csv(tweets):
	"""Write the subjectivity and polarity results to csv, along with the tweet"""
	columntitles = ["Polarity", "Subjectivity", "Tweet"]
	with open('twittersentiment.csv', 'w', newline='') as csvfile:
		tweetwriter = csv.writer(csvfile)
		tweetwriter.writerow(columntitles)
		for tweet in tweets:
			analysis = TextBlob(tweet.text)
			pol = analysis.sentiment.polarity
			sub = analysis.sentiment.subjectivity
			tweetwriter.writerow([pol,sub, analysis])
	return
		

def show_pie_chart(search_word, positive_list, neutral_list, negative_list):
    """Show a pie chart from the polarity results."""
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

# Connect to twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

search_word = "MAGA"
public_tweets = api.search(search_word)

polarity, subjectivity = get_sentiment(public_tweets)
positive_list, negative_list, neutral_list = rank_sentiment(polarity)

# Print mean subjectivity and polarity for reference
print(np.mean(polarity), np.mean(subjectivity))

show_pie_chart(search_word, positive_list, negative_list, neutral_list)
write_to_csv(public_tweets)


