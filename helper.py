from bs4 import BeautifulSoup
import re
import emoji
import nltk
from nltk.corpus import stopwords
import distance
from fuzzywuzzy import fuzz
import pickle
import numpy as np
import pandas as pd
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer, AutoModel
import torch
import gensim
import gensim.downloader as api
w2v = api.load('word2vec-google-news-300')

nltk.download('stopwords')


# cv=pickle.load(open('cv.pkl','rb'))


stop_words=stopwords.words('english')

def longest_common_substring(s1, s2):
    m = len(s1)
    n = len(s2)
    # Initialize matrix to store lengths of longest common suffixes
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
                longest = max(longest, dp[i+1][j+1])

    return longest

ps=PorterStemmer()
def preprocess(q):
  q=str(q).lower().strip()

  pattern=re.compile('<.*?>')
  q=re.sub(pattern,'',q)

  q=emoji.demojize(q)
  q=q.replace(':',' ')
  q=q.replace(',','')


  q=q.replace('%','percent')
  q=q.replace('$','dollar')
  q=q.replace('₹','rupee')
  q=q.replace('€','euro')
  q=q.replace('@','at')

  q=q.replace('[math]','')

  q=q.replace(',000,000,000 ', 'b ')
  q=q.replace(',000,000 ', 'm ')
  q=q.replace(',000 ', 'k ')
  q=re.sub(r'([0-9]+)000000000',r'\1b',q)
  q=re.sub(r'([0-9]+)000000',r'\1m',q)
  q=re.sub(r'([0-9]+)000',r'\1k',q)

  contractions = {
      "ain't": "am not / are not / is not / has not / have not",
      "aren't": "are not / am not",
      "can't": "cannot",
      "can't've": "cannot have",
      "'cause": "because",
      "could've": "could have",
      "couldn't": "could not",
      "couldn't've": "could not have",
      "didn't": "did not",
      "doesn't": "does not",
      "don't": "do not",
      "hadn't": "had not",
      "hadn't've": "had not have",
      "hasn't": "has not",
      "haven't": "have not",
      "he'd": "he had / he would",
      "he'd've": "he would have",
      "he'll": "he shall / he will",
      "he'll've": "he shall have / he will have",
      "he's": "he has / he is",
      "how'd": "how did",
      "how'd'y": "how do you",
      "how'll": "how will",
      "how's": "how has / how is / how does",
      "I'd": "I had / I would",
      "I'd've": "I would have",
      "I'll": "I shall / I will",
      "I'll've": "I shall have / I will have",
      "I'm": "I am",
      "I've": "I have",
      "isn't": "is not",
      "it'd": "it had / it would",
      "it'd've": "it would have",
      "it'll": "it shall / it will",
      "it'll've": "it shall have / it will have",
      "it's": "it has / it is",
      "let's": "let us",
      "ma'am": "madam",
      "mayn't": "may not",
      "might've": "might have",
      "mightn't": "might not",
      "mightn't've": "might not have",
      "must've": "must have",
      "mustn't": "must not",
      "mustn't've": "must not have",
      "needn't": "need not",
      "needn't've": "need not have",
      "o'clock": "of the clock",
      "oughtn't": "ought not",
      "oughtn't've": "ought not have",
      "shan't": "shall not",
      "sha'n't": "shall not",
      "shan't've": "shall not have",
      "she'd": "she had / she would",
      "she'd've": "she would have",
      "she'll": "she shall / she will",
      "she'll've": "she shall have / she will have",
      "she's": "she has / she is",
      "should've": "should have",
      "shouldn't": "should not",
      "shouldn't've": "should not have",
      "so've": "so have",
      "so's": "so as / so is",
      "that'd": "that would / that had",
      "that'd've": "that would have",
      "that's": "that has / that is",
      "there'd": "there had / there would",
      "there'd've": "there would have",
      "there's": "there has / there is",
      "they'd": "they had / they would",
      "they'd've": "they would have",
      "they'll": "they shall / they will",
      "they'll've": "they shall have / they will have",
      "they're": "they are",
      "they've": "they have",
      "to've": "to have",
      "wasn't": "was not",
      "we'd": "we had / we would",
      "we'd've": "we would have",
      "we'll": "we will",
      "we'll've": "we will have",
      "we're": "we are",
      "we've": "we have",
      "weren't": "were not",
      "what'll": "what shall / what will",
      "what'll've": "what shall have / what will have",
      "what're": "what are",
      "what's": "what has / what is",
      "what've": "what have",
      "when's": "when has / when is",
      "when've": "when have",
      "where'd": "where did",
      "where's": "where has / where is",
      "where've": "where have",
      "who'll": "who shall / who will",
      "who'll've": "who shall have / who will have",
      "who's": "who has / who is",
      "who've": "who have",
      "why's": "why has / why is",
      "why've": "why have",
      "will've": "will have",
      "won't": "will not",
      "won't've": "will not have",
      "would've": "would have",
      "wouldn't": "would not",
      "wouldn't've": "would not have",
      "y'all": "you all",
      "y'all'd": "you all would",
      "y'all'd've": "you all would have",
      "y'all're": "you all are",
      "y'all've": "you all have",
      "you'd": "you had / you would",
      "you'd've": "you would have",
      "you'll": "you shall / you will",
      "you'll've": "you shall have / you will have",
      "you're": "you are",
      "you've": "you have"
      }

  q_decontracted = []
  for word in q.split():
    if word in contractions:
      word=contractions[word]
    q_decontracted.append(word)
  q=' '.join(q_decontracted)
  q=q.replace("'ve"," have")
  q=q.replace("n't"," not")
  q=q.replace("'re"," are")
  q=q.replace("'ll"," will")

  # Removing Html tags
  q=BeautifulSoup(q)
  q=q.get_text()

  # removing punctuations
  pattern=re.compile('\W')
  q=re.sub(pattern,' ',q).strip()

  q=" ".join([ps.stem(word) for word in q.split()])

  return q
def tcommon_words(q1,q2):
    w1=set(map(lambda word: word.lower().strip(),q1.split(" ")))
    w2=set(map(lambda word: word.lower().strip(),q2.split(" ")))
    return len(w1&w2)
def ttotal_tokens(q1,q2):
    w1=set(map(lambda word: word.lower().strip(),q1.split(" ")))
    w2=set(map(lambda word: word.lower().strip(),q2.split(" ")))
    return len(w1)+len(w2)
negation_words = {"no", "not", "never", "none", "nothing", "nowhere", "neither", "nor"}
def tis_negation(q1,q2):
    w1 = list(map(lambda word: word.lower().strip(), str(q1).split()))
    w2 = list(map(lambda word: word.lower().strip(), str(q2).split()))

    t1=any(word in negation_words for word in w1)
    t2=any(word in negation_words for word in w2)

    return 1 if t1==t2 else 0
def ttotal_words(q1,q2):
    w1 = list(map(lambda word: word.lower().strip(), str(q1).split()))
    w2 = list(map(lambda word: word.lower().strip(), str(q2).split()))

    q1_word_count = len(w1)
    q2_word_count = len(w2)

    common_words = [word for word in set(w1).intersection(set(w2)) if word not in stop_words]
    common_word_count = len(common_words)

    common_token_count=len(set(w1).intersection(set(w2)))

    # return only the two ratios instead of raw counts
    cwc_min = common_word_count / min(q1_word_count, q2_word_count) if min(q1_word_count, q2_word_count) > 0 else 0
    cwc_max = common_word_count / max(q1_word_count, q2_word_count) if max(q1_word_count, q2_word_count) > 0 else 0

    ctc_min=common_token_count/min(q1_word_count,q2_word_count) if min(q1_word_count, q2_word_count) > 0 else 0
    ctc_max=common_token_count/max(q1_word_count,q2_word_count) if max(q1_word_count, q2_word_count) > 0 else 0

    cw_tw1=common_word_count/q1_word_count if q1_word_count > 0 else 0
    cw_tw2=common_word_count/q2_word_count if q2_word_count > 0 else 0

    return pd.Series([cw_tw1,cw_tw2,cwc_min,cwc_max,ctc_min,ctc_max])

def tstopword_features(q1,q2):
    q1_words = str(q1).lower().split()
    q2_words = str(q2).lower().split()

    # Get stopword lists
    q1_stops = [word for word in q1_words if word in stop_words]
    q2_stops = [word for word in q2_words if word in stop_words]

    q1_stop_count = len(q1_stops)
    q2_stop_count = len(q2_stops)

    common_stop_count = len(set(q1_stops).intersection(set(q2_stops)))

    # Avoid ZeroDivisionError
    if q1_stop_count == 0 or q2_stop_count == 0:
        csc_min = 0
        csc_max = 0
    else:
        csc_min = round(common_stop_count / min(q1_stop_count, q2_stop_count) if min(q1_stop_count, q2_stop_count) > 0 else 0, 2)
        csc_max = round(common_stop_count / max(q1_stop_count, q2_stop_count) if max(q1_stop_count, q2_stop_count) > 0 else 0, 2)

    return pd.Series([csc_min, csc_max,common_stop_count])

def tlast_first_word(q1,q2):
    w1 = q1.strip().split() if isinstance(q1, str) else []
    w2 = q2.strip().split() if isinstance(q2, str) else []

    if not w1 or not w2:  # if either is empty list
        return 0

    last= 1 if w1[-1].lower() == w2[-1].lower() else 0
    first= 1 if w1[0].lower() == w2[0].lower() else 0
    return pd.Series([last,first])
def tcommon_ngrams_count(q1,q2):
    q1_words = str(q1).lower().split()
    q2_words = str(q2).lower().split()

    q1_ngrams = set(ngrams(q1_words, 2))
    q2_ngrams = set(ngrams(q2_words, 2))
    common_ngrams2 = q1_ngrams.intersection(q2_ngrams)

    q1_ngrams = set(ngrams(q1_words, 3))
    q2_ngrams = set(ngrams(q2_words, 3))
    common_ngrams3 = q1_ngrams.intersection(q2_ngrams)

    return pd.Series([len(common_ngrams2),len(common_ngrams3)])
def tchar_features(q1,q2):
    q1_chars = set(str(q1).lower().replace(" ", ""))
    q2_chars = set(str(q2).lower().replace(" ", ""))

    common_chars = q1_chars & q2_chars
    total_chars = q1_chars | q2_chars

    common_count = len(common_chars)
    total_count = len(total_chars)

    ratio = common_count / total_count if total_count > 0 else 0

    return ratio
def tlength(q1,q2):
    w1 = list(map(lambda word: word.lower().strip(), str(q1).split()))
    w2 = list(map(lambda word: word.lower().strip(), str(q2).split()))

    l1 = len(w1)
    l2 = len(w2)

    t1 = sum(len(word) for word in w1) / len(w1) if len(w1)!=0 else 0
    t2 = sum(len(word) for word in w2) / len(w2) if len(w2)!=0 else 0
    mean_word_diff = abs(t1 - t2)



    meanl = (l1 + l2) / 2
    abs_len = abs(l1 - l2)
    return pd.Series([meanl, abs_len,mean_word_diff])

def tlongest_substr_ratio(q1,q2):
    q1 = str(q1).lower()
    q2 = str(q2).lower()

    lcs_len = longest_common_substring(q1, q2)
    min_len = min(len(q1), len(q2))

    if min_len == 0:
        return 0  # avoid division by zero

    return lcs_len / min_len
def tfuzzy_features(q1,q2):
    w1=str(q1)
    w2=str(q2)
    fuzzy=[0.0]*4

    fuzzy[0]= fuzz.ratio(w1,w2)
    fuzzy[1]= fuzz.partial_ratio(w1,w2)
    fuzzy[2]= fuzz.token_sort_ratio(w1,w2)
    fuzzy[3]= fuzz.token_set_ratio(w1,w2)

    return fuzzy
def tget_avg_w2v_vector(text):
    words = str(text).lower().split()
    word_vectors = []
    for word in words:
        if word in w2v:
            word_vectors.append(w2v[word])

    if not word_vectors:
        return np.zeros(300)
    return np.mean(word_vectors, axis=0)

def query_point_creator(q1,q2):

    input_query=[]
    q1=preprocess(q1)
    q2=preprocess(q2)

    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    input_query.append(tcommon_words(q1,q2))
    input_query.append(ttotal_tokens(q1,q2))
    input_query.append(input_query[-2]/input_query[-1])

    # token features
    input_query.append(tis_negation(q1,q2))

    t_total_words=ttotal_words(q1,q2)
    input_query.extend(t_total_words)

    t_stopword_features=tstopword_features(q1,q2)
    input_query.extend(t_stopword_features)

    t_last_first_word=tlast_first_word(q1,q2)
    input_query.extend(t_last_first_word)

    t_common_ngrams_count=tcommon_ngrams_count(q1,q2)
    input_query.extend(t_common_ngrams_count)

    input_query.append(tchar_features(q1,q2))

    t_length=tlength(q1,q2)
    input_query.extend(t_length)

    input_query.append(tlongest_substr_ratio(q1,q2))

    t_fuzzy_features=tfuzzy_features(q1,q2)
    input_query.extend(t_fuzzy_features)

    #BOW
    # q1_bow=cv.transform([q1]).toarray()
    # q2_bow=cv.transform([q2]).toarray()
    
    #W2V
    q1_vec = tget_avg_w2v_vector(q1)
    q2_vec = tget_avg_w2v_vector(q2)

    # # Handle None or invalid vectors
    # if q1_vec is None or isinstance(q1_vec, float):
    #     q1_vec = np.zeros(300)
    # if q2_vec is None or isinstance(q2_vec, float):
    #     q2_vec = np.zeros(300)

    #BERT
    # q1_vec = tget_bert_cls_embedding(q1)
    # q2_vec = tget_bert_cls_embedding(q2)
    
    return np.hstack((np.array(input_query), q1_vec, q2_vec))


# Use BERT base uncased
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")

# # Set model to eval mode (important to disable dropout)
# model.eval()
# def tget_bert_cls_embedding(text):
#     with torch.no_grad():
#         inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=64)
#         outputs = model(**inputs)
#         cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
#         return cls_embedding.squeeze().numpy()  # shape: (768,)
