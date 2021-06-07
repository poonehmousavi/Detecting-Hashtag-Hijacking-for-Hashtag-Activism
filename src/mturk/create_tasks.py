
import boto3
import json
import pandas as pd


from src.mturk import mturk_credentials

"""
Class for creating Mechanical Mturk task for labeling tweets
"""
create_hits_in_production = False
environments = {
  "production": {
    "endpoint": "https://mturk-requester.us-east-1.amazonaws.com",
    "preview": "https://www.mturk.com/mturk/preview"
  },
  "sandbox": {
    "endpoint":
          "https://mturk-requester-sandbox.us-east-1.amazonaws.com",
    "preview": "https://workersandbox.mturk.com/mturk/preview"
  },
}
mturk_environment = environments["production"] if create_hits_in_production else environments["sandbox"]
client = boto3.client(
    service_name='mturk',
    aws_access_key_id=mturk_credentials.AWSAccessKeyId,
    aws_secret_access_key=mturk_credentials.AWSSecretKey,
    region_name='us-east-1',
    endpoint_url=mturk_environment['endpoint'],
)
# print(client.get_account_balance()['AvailableBalance'])
html_layout = open('metoodetection.html', 'r').read()
QUESTION_XML = """<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
        <HTMLContent><![CDATA[{}]]></HTMLContent>
        <FrameHeight>650</FrameHeight>
        </HTMLQuestion>"""
question_xml = QUESTION_XML.format(html_layout)

TaskAttributes = {
    'MaxAssignments': 2,
    # How long the task will be available on MTurk
    'LifetimeInSeconds': 60*60*24*14,
    # How long Workers have to complete each item
    'AssignmentDurationInSeconds': 60*3,
    'AutoApprovalDelayInSeconds': 60*60*24*2,
    # The reward you will offer Workers for each response
    'Reward': '0.01',
    'Title': 'Is this tweet related to MeToo Movement(women Movement)?_livedata_validation',
    'Keywords': 'metoo, tweet, hashtag,hijacking',
    'Description': 'Is this tweet relevant to MeToo Movement?'
}

results = []

#### remove emojes from tweet ##########
def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

### Activate URL link embedded in  Tweet ########
def make_url(text, urls):
    for url in urls:
        text= text.replace(url, " <a href=\""+url+"\"target=\"_blank\"> "+ url+" </a> ")
    return  text



tweets =  pd.read_json('../../data/finalrun/Final/live1/sample_live.json', orient='records', lines=True)


result_file = open('../../data/mturk_livedata_validation_hits.json', "w", encoding='utf-8')
start= True
for index,tweet in tweets.iterrows():
    urls= [] if tweet['urls'] == None else tweet['urls']
    question = question_xml.replace('${text}', make_url(deEmojify(tweet['fulltext']),urls))
    response = client.create_hit(**TaskAttributes, Question=question )
    hit_type_id = response['HIT']['HITTypeId']
    if start== True:
        start=False
        print("You can view the HITs here:")
        print(mturk_environment['preview'] + "?groupId={}".format(hit_type_id))

    # results.append({
    #     'tweet_id': tweet['id'],
    #     'hit_id': response['HIT']['HITId']
    #         })
    result_file.write(json.dumps({
        'tweet_id': tweet['id'],
        'hit_id': response['HIT']['HITId']
            }) + '\n')


# with open('./Metoo_Archive/report/mturk_data.json', 'w', encoding='utf-8') as f:
#     json.dump(results, f, ensure_ascii=False, indent=4)

print("You can view the HITs here:")
print(mturk_environment['preview'] + "?groupId={}".format(hit_type_id))
print("All hits are posted")




