'''
    Using Twitter API to get live tweets on a specific topic
'''
import tweepy
from time import sleep
import settings

#The time to sleep when the API limit is reached in seconds
SLEEP_TIME          = 60*15

def auth():
    '''
        Authenticate to Twitter
    '''
    auth = tweepy.OAuthHandler(settings.CONSUMER_KEY, settings.CONSUMER_SECRET)
    auth.set_access_token(settings.ACCESS_TOKEN, settings.ACCESS_SECRET)
    return tweepy.API(auth, parser=tweepy.parsers.JSONParser())

API = auth()

def make_call(func,*args,**kwargs):
    '''
        Make an API call, waits if the limit is reached
    '''
    while True :
        try :
            return func(*args,**kwargs)
        except Exception as e :
            if 'that page does not exist' in str(e):
                return []
            print ('Sleeping for {} seconds'.format(SLEEP_TIME))
            sleep(SLEEP_TIME)

def get_last_tweets(tag):
    '''
        Get the last tweets corresponding the given tag
    '''
    return [x['retweeted_status']['full_text'] if x.get('retweeted_status') else x['full_text'] for x in  make_call(API.search,'#'+tag,count=settings.TWEETS_LIM,tweet_mode='extended',result_type='recent')['statuses']]
