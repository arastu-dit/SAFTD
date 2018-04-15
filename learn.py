"""
    This class contains the logic for Machine Learning
"""
from sklearn.feature_extraction.text import CountVectorizer
import json
import os
import codecs
import pandas as pd
import re
import nltk
from collections import Counter
import settings
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from time import time

#Contains regex expressions for cleaning

CLEAN_REGEX = {
    'url'       : 'http.?://[^\s]+[\s]?'                    ,
    'username'  : '@[^\s]+[\s]?'                            ,
    'tag'       : '#[^\s]+[\s]?'                            ,
    'empty'     : 'Not Available'                           ,
    'number'    : '\s?\d+\.?\d*'                            ,
    'special'   : '[^\w+\s]'                                ,
    }

def clean_tweet(tweet_str):
    """
        Cleans a tweet from Urls, Usernames, Empty tweets, Special Characters, Numbers and Hashtags.
    """
    try :
        #Create the combined expression to eliminate all unwanted strings
        clean_ex    = '|'.join(['(?:{})'.format(x) for x in CLEAN_REGEX.values()])
        result      = re.sub(clean_ex, '', tweet_str).strip()

        #Remove character sequences
        return re.sub('([a-z])\\1+', '\\1\\1', result).strip()
    except :
        #Some times the DataFrame contains NAN
        return ''

def stem_tweet(tokens):
    """
        Stemming the process of reducing a derived word to it's original word.
    """
    #Using SnowballStemmer for english
    stemmer     = nltk.SnowballStemmer('english')
    return [stemmer.stem(x) for x in tokens]

def tokenize_tweet(tweet_str):
    """
        Tokenization.
    """
    #Using SnowballStemmer for english
    return nltk.word_tokenize(tweet_str,'english')

def init():
    """
        - Initialize the training and testing sets.
        - Cleans the data frames
        - Tokenization
        - Stemming
    """
    pol_path    = os.path.join(os.getcwd(),settings.DATA_FOLDER, 'pol.csv'      )
    if not( os.path.isfile(pol_path) and settings.LAOD_POL_CSV) :
        train_path  = os.path.join(os.getcwd(),settings.DATA_FOLDER, 'train.csv'     )

        train_df    = pd.DataFrame.from_csv(train_path  )

        #Clean the data frames
        print('Initiliazing cleaning of the datasets!')
        train_df['Tweet']    = train_df['Tweet'].apply(clean_tweet)
        #Remove empty entries
        train_df            = train_df[train_df['Tweet']!='']
        print('Datasets have been successfully cleaned!')

        #Stemming and tokenization
        print('Initializing Tokenization and Stemming of the datasets!')
        train_df['tokens']  = train_df['Tweet' ].apply(tokenize_tweet   )
        train_df['stem']    = train_df['tokens'].apply(stem_tweet       )
        print('Stemming and Tokenization successful!')

        #Counting the words occurences and calculating the polarization
        count_df            = words_occ(train_df)
    else :
        train_df,count_df   = None, None
    words_df,pol_df     = words_pol(train_df,count_df)

    train_args                                          = get_train_data(words_df)
    precision, recall, accuracy, f1,model               = train(*train_args)
    tokens                                              = words_df.columns[1:]
	#Shows prediction accuracy of classifier at 57%
    print ('Prediction Accuracy : {}'.format(accuracy))
    return model,tokens

def words_occ(df,column = 'stem'):
    """
        Get the occurences of words in a given data frame
    """
    print ('Counting tokens started!')
    #Create the counter
    words_counter   = Counter()
    for x in df[column]:
        words_counter.update(x)

    #Download the stop wrods
    try :
        nltk.download('stopwords')
    except :
        pass

    #Remove the stop words, excluding not
    stop_words      = nltk.corpus.stopwords.words('english')
    stop_words.remove('not')
    for x in stop_words :
        if 'n\'t' not in x :
            del words_counter[x]

    #Return the df indexed by words, sorted buy count
    df = pd.DataFrame([[k,words_counter[k]] for k in words_counter],columns=['word','count']).set_index('word').sort_values(by = 'count',ascending = False)
    #Remove words with less than three occurences
    df = df[df['count']>3]
    print ('Counting tokens done!')

    return df

def words_pol(df_tweets,df_words):
    """
        Creates a data frame contining the occurence of each word in df_words for every tweet in df_tweets
    """
    print ('Calculating words polarization started!')
    loaded      = False
    pol_path    = os.path.join(os.getcwd(),settings.DATA_FOLDER, 'pol.csv'      )
    if settings.LAOD_POL_CSV :
        try :
            df          = pd.DataFrame.from_csv(pol_path)
            loaded      = True
        except Exception as e  :
            print(e)
            pass

    if not loaded :
        df = pd.DataFrame(
            [
                [row['Category']]+[row['stem'].count(word) for word in df_words.index] for id_,row in df_tweets.iterrows()],
            columns = ['Category']+list(df_words.index)
        )
        df.to_csv(pol_path)

    #Group the rows by sentiment, and sum the groups
    #This will produce the total occurences of a word for each sentiment
    gdf = df.groupby(['Category']).sum()

    print ('Calculating words polarization done!')
    #Flip and return the DataFrame
    return df,gdf.T

def train(train_x,test_x,train_y,test_y,classifier=BernoulliNB()):
    '''
        Train the classifier and test against the test data
    '''
    print('Training started!')
    model       = classifier.fit(train_x, train_y)
    predictions = model.predict(test_x)
    labels      = sorted(list(set(train_y)))
    precision   = precision_score(test_y, predictions, average=None, pos_label=None, labels=labels)
    recall      = recall_score(test_y, predictions, average=None, pos_label=None, labels=labels)
    accuracy    = accuracy_score(test_y, predictions)
    f1          = f1_score(test_y, predictions, average=None, pos_label=None, labels=labels)
    print('Training complete!')

    return precision, recall, accuracy, f1,model

def get_train_data(words_df):
    '''
        Split the training data into test and train data
    '''
    return  train_test_split(words_df.iloc[:, 1:].values, words_df.iloc[:, 0].values,train_size=0.7, stratify=words_df.iloc[:, 0].values,random_state=123)

def predict(model, tweet,tokens):
    '''
        Predicts the sentiment of the tweet using the trained model
    '''
    stemmed = stem_tweet(tokenize_tweet(clean_tweet(tweet)))
    mask    =  [stemmed.count(token) for token in tokens ]

    return model.predict([mask])[0]
