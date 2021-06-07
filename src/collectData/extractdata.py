import json
import re
import sys
import os
import pandas as pd
import ast
from pandas import json_normalize
from collections import Counter
output_path="../../data/"
def merge_Data(input_path,category):
    """
    Merge all datas from different spammy hashtags category and assign category labels for them
    """
    tweets_data = []
    with open(input_path, "r") as tweets_file:
        for line in tweets_file:
            try:
                temp = json_normalize(json.loads(line))
                if temp['truncated'][0] == False:
                    temp["fulltext"] = temp["text"][0]
                else:
                    temp["fulltext"] = temp["extended_tweet.full_text"][0]
                tweet=temp[['id',"fulltext"]]
                tweet['category']= category
                tweets_data.append(tweet)
            except:
                continue

    return tweets_data

def get_Data(file_path):
    data = pd.DataFrame()


    for filename in os.listdir(file_path):
            index=filename.find('-')
            df=merge_Data(file_path + filename,filename[0:index])
            data = data.append(df)




    data.to_csv(output_path + "all_general.csv")
    data.to_json(output_path + "all_general.json", orient='records', lines=True)


def main():
    get_Data(output_path+'generak/')



if __name__ == "__main__":
    main()