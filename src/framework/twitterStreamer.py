import json
from pandas.io.json import json_normalize

import pandas as pd
from  tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import datetime
from tweepy import TweepError
from src.collectData import twitter_credentials

consumer_key = twitter_credentials.consumer_key
consumer_secret = twitter_credentials.consumer_secret
access_token = twitter_credentials.access_key
access_token_secret = twitter_credentials.access_secret


class TwitterStreamer():

    def __init__(self,batch_size,logger):
        self.logger=logger
        self.batch_size=batch_size
    """
    Class for streaming and processing live tweets
    """
    def stream_tweets(self, output_file_path, hashtag_list):
        self.logger.info("Start Connecting to Live Stream Twitter")
        listener = STDOutListener(output_file_path,self.batch_size,self.logger)

        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        stream = Stream(auth, listener,wait_on_rate_limit=True)
        self.logger.info("Connection established")

        self.logger.info("Start collecting  Live Stream Tweets")
        stream.filter(languages=["en"],track=hashtag_list,is_async=True)




class STDOutListener(StreamListener):
    """
    The basic listener Cass
    """
    def __init__(self,output_file_path,batch_size,logger):

        self.output_file_path=output_file_path
        self.batch_size=batch_size
        self.logger=logger
        self.tweets=[]


    def on_data(self, raw_data):
        raw_data=json.loads(raw_data)
        if 'retweeted_status'  in raw_data:
            return
        if raw_data['in_reply_to_status_id'] != None:
            return
        if len(self.tweets) < self.batch_size:
            self.tweets.append(raw_data)
        else:
            cur = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            out_put = self.output_file_path + 'tweets-' + cur + '.json'
            try:
                result = self.tweets.copy()
                self.tweets.clear()
                self.tweets.append(raw_data)
                result= self.tune_features(result)
                file = open(out_put, "w")
                # with open(out_put,'a') as tf:
                #     for tweet in result.iterate:
                #         tf.write(tweet)
                result.to_json(file, orient='records', lines=True)
                file.close()
            except TweepError as e:
                self.logger.error("Error on fetching tweet", exc_info=True)





    def on_error(self, status_code):
        self.logger.error("Exception occurred", exc_info=True)

    def tune_features(self,tweets):
        df = pd.DataFrame(json_normalize(tweets),
                          columns=['created_at', 'id', 'user.id', 'text', 'extended_tweet.full_text',
                                   'extended_tweet.entities.hashtags', 'truncated', 'retweeted_status.id',
                                   'in_reply_to_status_id', 'entities.hashtags', 'entities.urls',
                                   'extended_tweet.entities.urls',
                                   'favorite_count', 'retweet_count', 'quote_count', 'user.verified',
                                   'user.friends_count', 'user.followers_count', 'user.created_at', 'user.description',
                                   'use.statuses_count'])
        df.sort_values("id", inplace=True)
        ###### Removing all duplicate tweets #############
        df.drop_duplicates(subset="id", keep=False, inplace=True)


        ############ Getting Full Text and Hashtags for truncated tweets ###############
        for index, row in df.iterrows():
            if row['truncated'] == False:
                df.loc[index, "fulltext"] = row["text"]
                if row['entities.hashtags'] != []:
                    df.loc[index, "hashtags"] = [[element['text']] for element in row['entities.hashtags']]
                if row['entities.urls'] != []:
                   df.loc[index, "urls"] = [[element['url']] for element in row['entities.urls']]
            else:
                df.loc[index, "fulltext"] = row["extended_tweet.full_text"]
                if row['extended_tweet.entities.hashtags'] != []:
                    df.loc[index, "hashtags"] = [[element['text']] for element in
                                                      row['extended_tweet.entities.hashtags']]

                if row['extended_tweet.entities.urls'] != []:
                    df.loc[index, "urls"] = [[element['url']] for element in row['extended_tweet.entities.urls']]

        df["number_of_hashtags"] = [0 if x!=x else len(x)  for x in df['hashtags']]
        df['created_at'] = pd.to_datetime(df.created_at)
        df = df.drop(
            columns=['text', 'extended_tweet.full_text', 'entities.hashtags', 'extended_tweet.entities.hashtags',
                     'truncated', 'retweeted_status.id', 'in_reply_to_status_id', 'entities.urls',
                     'extended_tweet.entities.urls'])

        return df

