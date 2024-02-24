import spacy
import json
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import nltk
import sklearn
from sklearn_crfsuite import metrics
import sklearn_crfsuite as crfsuite
import utils.metrics_modified as m
import joblib
import os
from utils.ner_eval import collect_named_entities
from utils.ner_eval import compute_metrics
from utils.ner_eval import Evaluator


nlp = spacy.load("en_core_web_sm")

# Load needed datasets
with open('/content/drive/MyDrive/NLP_project/data/NER_TRAIN/NER_TRAIN_JUDGEMENT.json', 'r') as file:
    dataset = json.load(file)
with open('/content/drive/MyDrive/NLP_project/data/NER_DEV/NER_DEV_JUDGEMENT.json', 'r') as file:
    dataset_dev = json.load(file)

# To tokenize the datasets in format (word, POS tag, entity tag) following the BIO convention
def tokenize_dataset(dataset):
    
  tokenized_data = []

  for entry in dataset:
      text = entry["data"]["text"]
      annotations = entry["annotations"][0]["result"]

      doc = nlp(text)
      tokens = []

      for token in doc:
          pos_tag = token.pos_
          entity_tag = "O"

            # find entity tag based on token position
          for annotation in annotations:
            start = annotation["value"]["start"]
            end = annotation["value"]["end"]
            entity_type = annotation["value"]["labels"][0]

            if start <= token.idx and end >= token.idx + len(token.text):
                if start == token.idx:
                    entity_tag = f"B-{entity_type}" # first word of the token
                else:
                    entity_tag = f"I-{entity_type}" # following words of the token

          tokens.append((token.text, pos_tag, entity_tag))
      
      tokenized_data.append(tokens)

  return tokenized_data

# prepare train, dev and test ds
tokenized_dataset = tokenize_dataset(dataset)
test_ds = tokenize_dataset(dataset_dev)
train_ds, dev_ds = train_test_split(tokenized_dataset, test_size=0.2, random_state=42)

def read_clusters(cluster_file):
    word2cluster = {}
    df = pd.read_csv(cluster_file, sep='\t')

    for index, row in df.iterrows():
        word = row['Word']
        cluster = int(row['Cluster'])  
        word2cluster[word] = cluster
    return word2cluster


def word2features(sent, i, word2cluster):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'word.cluster=%s' % word2cluster[word.lower()] if word.lower() in word2cluster else "0",
        'postag=' + postag
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1
        ])
    else:
        features.append('BOS')

    if i > 1: 
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        features.extend([
            '-2:word.lower=' + word2.lower(),
            '-2:word.istitle=%s' % word2.istitle(),
            '-2:word.isupper=%s' % word2.isupper(),
            '-2:postag=' + postag2
        ])        

        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1
        ])
    else:
        features.append('EOS')

    if i < len(sent)-2:
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        features.extend([
            '+2:word.lower=' + word2.lower(),
            '+2:word.istitle=%s' % word2.istitle(),
            '+2:word.isupper=%s' % word2.isupper(),
            '+2:postag=' + postag2
        ])

        
    return features


def sent2features(sent, word2cluster):
    return [word2features(sent, i, word2cluster) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

# Load file with mapped clustered word embeddings
word2cluster = read_clusters("/content/drive/MyDrive/NLP_project/data/law_clusters.tsv")

# Prepare train,dev,test datasets with extracted features
X_train = [sent2features(s, word2cluster) for s in train_ds]
y_train = [sent2labels(s) for s in train_ds]

X_dev = [sent2features(s, word2cluster) for s in dev_ds]
y_dev = [sent2labels(s) for s in dev_ds]

X_test = [sent2features(s, word2cluster) for s in test_ds]
y_test = [sent2labels(s) for s in test_ds]

'''
#----------TRAINING THE MODEL------------
crf = crfsuite.CRF(
    verbose='true',
    algorithm='lbfgs',
    max_iterations=100
)

crf.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)

OUTPUT_PATH = "/content/drive/MyDrive/NLP_project"
OUTPUT_FILE = "crf_model_final"

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

# Load the trained model 
joblib.dump(crf, os.path.join(OUTPUT_PATH, OUTPUT_FILE))

crf = joblib.load(os.path.join(OUTPUT_PATH, OUTPUT_FILE))


#----------TESTING ON A SAMPLE------------
y_pred = crf.predict(X_test)

example_sent = test_ds[0]

print("Sentence:", ' '.join(sent2tokens(example_sent)))
print("Predicted:", ' '.join(crf.predict([sent2features(example_sent, word2cluster)])[0]))
print("Correct:  ", ' '.join(sent2labels(example_sent)))
'''

#--------TESTING ON THE WHOLE TEST SET---------

# load the crf model
OUTPUT_PATH = "/content/drive/MyDrive/NLP_project/models"
OUTPUT_FILE = "crf_model_final"
crf = joblib.load(os.path.join(OUTPUT_PATH, OUTPUT_FILE))

#predict the labels
y_pred = crf.predict(X_test)

# remove 'O' label from evaluation
labels = list(crf.classes_)
labels.remove('O') 

# group B and I results
sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0])) 
tags = ['CASE_NUMBER',  'COURT',  'DATE', 'GPE',  'JUDGE',  'ORG', 'OTHER_PERSON', 'PETITIONER',  'PRECEDENT', 'PROVISION', 'RESPONDENT', 'STATUTE', 'WITNESS']

# evaluate each entity
print(m.flat_classification_report(y_test, y_pred,sorted_labels))

# overall results f1 strict, exact, partial, type match
test_sents_labels = []
for sentence in test_ds:
    sentence = [token[2] for token in sentence]
    test_sents_labels.append(sentence)

evaluator = Evaluator(test_sents_labels, y_pred, tags)
results, results_agg = evaluator.evaluate()

def calculate_f1_scores(results):
    f1_scores = {}

    for matching_type in results.keys():
        precision = results[matching_type]['precision']
        recall = results[matching_type]['recall']

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        f1_scores[matching_type] = f1_score

    return f1_scores


# Print the F1 scores
f1_scores = calculate_f1_scores(results)

for matching_type, f1_score in f1_scores.items():
    print(f"F1 {matching_type}: {f1_score}")






