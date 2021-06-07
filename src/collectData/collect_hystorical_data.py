import json
from searchtweets import ResultStream, gen_rule_payload, load_credentials





def __init__(self, api=None):
    """
    Collect Historical data from Oct 2017 till Nov 2019
    """
    self.file = open(OUTPUT_FILE, "w")

dev_env=[
         "search_Popo_tweets_fullarchive_dev","search_Ali_tweets_fullarchive_dev","search_PUTD_tweets_fullarchive_dev"]

From_Date=["2017-10-01","2017-11-01","2017-12-01","2018-01-01","2018-02-01","2018-03-01","2018-04-01"
           ,"2018-05-01","2018-06-01","2018-07-01","2018-08-01","2018-09-01","2018-10-01","2018-11-01",
           "2018-12-01","2019-01-01","2019-02-01","2019-03-01","2019-04-01"
           ,"2019-05-01","2019-06-01","2019-07-01","2019-08-01","2019-09-01","2019-10-01","2019-11-01"]

To_Date=["2017-10-31","2017-11-30","2017-12-31","2018-01-31","2018-02-28","2018-03-31","2018-04-30"
           ,"2018-05-31","2018-06-30","2018-07-31","2018-08-31","2018-09-30","2018-10-31","2018-11-30",
           "2018-12-31","2019-01-31","2019-02-28","2019-03-31","2019-04-30"
           ,"2019-05-31","2019-06-30","2019-07-31","2019-08-31","2019-09-30","2019-10-31","2019-11-30"]

Category=["Politics","Brands","Others "]
Query=[
       "#namo OR #congress OR #AAP OR #BJP OR #namobirthday lang:en"," #puma OR #adidas OR #Samsung OR #Lakme lang:en",
       "#happy OR #Birthday OR #Rain OR #Sunny OR #KillMe lang:en"]

for i in range(len(Query)):
     premium_search_args = load_credentials("./twitter_keys.yaml",
                                           yaml_key=dev_env[i],
                                           env_overwrite=False)
     for j in range(len(From_Date)):


        OUTPUT_FILE="../../data/"+Category[i]+"-"+From_Date[j]+"_"+To_Date[j]+".txt"

        rule = gen_rule_payload(Query[i],
                        from_date=From_Date[j], #UTC 2017-09-01 00:00
                        to_date=To_Date[j],#UTC 2018-11-30 00:00
                        results_per_call=100)
        print(rule)

# tweets = collect_results(rule, max_results=1000, result_stream_args=premium_search_args)
        rs = ResultStream(rule_payload=rule,
                  max_results=100,
                  **premium_search_args)


        tweets = list(rs.stream())
        output = open(OUTPUT_FILE, "w")

        for tweet in tweets:
            output.write(json.dumps(tweet) + '\n')