import re, numpy as np, pandas as pd
import sys
from pprint import pprint
from gensim.test.utils import datapath

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
import matplotlib
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel, LdaModel
import matplotlib.pyplot as plt# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'model', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come','music','https','http','co','amp'])

warnings.filterwarnings("ignore",category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


class TopicDetector():
    def __init__(self,data):
        self.df=data[['postID', 'text', 'user.id', 'eventid']]
        self.df.dropna(inplace=True)
        self.df['user.id'] = self.df['user.id'].astype(str)
        self.data = self.df.text.values.tolist()
        self.data_words = list(self.sent_to_words(data))
        self.bigram = gensim.models.Phrases(self.data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        self.trigram = gensim.models.Phrases(self.bigram[self.data_words], threshold=100)
        self.bigram_mod = gensim.models.phrases.Phraser(self.bigram)
        self.trigram_mod = gensim.models.phrases.Phraser(self.trigram)

    def sent_to_words(self,sentences):
        for sent in sentences:
            sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
            sent = re.sub('\s+', ' ', sent)  # remove newline chars
            sent = re.sub("\'", "", sent)  # remove single quotes
            sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
            yield (sent)
output_path="../report/out/"
def main(args):
    training = pd.read_json('../../data/proceesed_clean_training_all.json',orient='records',lines=True)

if __name__ == '__main__':
        sys.exit(main(sys.argv[1:]))