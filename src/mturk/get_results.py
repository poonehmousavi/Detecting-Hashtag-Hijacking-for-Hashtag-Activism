import boto3
import sys
import json
import xmltodict
import  pandas as pd
from src.mturk import mturk_credentials

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

"""
Gathering result from Mechanical Mturk and get majority votes
"""



def getResponse(client, item):
    hit = client.get_hit(HITId=item['hit_id'])
    item['status'] = hit['HIT']['HITStatus']
    # Get a list of the Assignments that have been submitted
    assignmentsList = client.list_assignments_for_hit(
        HITId=item['hit_id'],
        AssignmentStatuses=['Submitted', 'Approved'],
        MaxResults=2
    )
    assignments = assignmentsList['Assignments']
    item['assignments_submitted_count'] = len(assignments)
    answers = []

    if len(assignments) == 0:
        print("No submitted assignmet for hit id %s" , item['hit_id'])
        return item
    for assignment in assignments:

        # Retreive the attributes for each Assignment
        worker_id = assignment['WorkerId']
        assignment_id = assignment['AssignmentId']

        # Retrieve the value submitted by the Worker from the XML
        answer_dict = xmltodict.parse(assignment['Answer'])
        answer = answer_dict['QuestionFormAnswers']['Answer']['FreeText']
        answers.append((convert_to_label(answer),worker_id))


        # Approve the Assignment (if it hasn't been already)
        if assignment['AssignmentStatus'] == 'Submitted':
            client.approve_assignment(
                AssignmentId=assignment_id,
                OverrideRejection=False
            )


    # Add the answers that have been retrieved for this item
    item['answers'] = answers
    if len(answers) > 0:
        item['avg_answer'] =most_frequent([answer[0] for answer in answers])

    return  item

def most_frequent(num_list):
        idx, ctr = 0, 1

        for i in range(1, len(num_list)):
            if num_list[idx] == num_list[i]:
                ctr += 1
            else:
                ctr -= 1
                if ctr == 0:
                    idx = i
                    ctr = 1

        return num_list[idx]

def convert_to_label(answer):
    if 'Yes' in answer:
        return 1
    elif 'No' in answer:
        return -1
    else:
        return 0

def validate_answer(answers, index):
    if answers[index] == -1:
        return True
    else:
        return  False

def main(args):
    results=[]
    result_file = open('../../data/mturk/mturk_live_validation_hits_result.json', "w", encoding='utf-8')
    mturk_environment = environments["production"] if create_hits_in_production else environments["sandbox"]
    client = boto3.client(
        service_name='mturk',
        aws_access_key_id=mturk_credentials.AWSAccessKeyId,
        aws_secret_access_key=mturk_credentials.AWSSecretKey,
        region_name='us-east-1',
        endpoint_url=mturk_environment['endpoint'],
    )


    datas=pd.read_json('../../data/mturk_livedata_validation_hits.json', orient='records', lines=True)
    for index, item in datas.iterrows():
        result_file.write(getResponse(client, item).to_json() + '\n')



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

