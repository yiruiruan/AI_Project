import sys
import glob
import re
import spacy
import pytextrank
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
from base import extract
from rake import rake
from textrank import textrank
from window import extract as window
from window_w_tf_idf import extract as window_w_tf_idf
from tf_idf import extract as tf_idf
from sentiment_pos import extract as sentiment_pos
from sentiment_pos_tfidf import extract as sentiment_pos_tfidf

ALGOS = {'rake', 'textrank', 'window', 'window_w_tf_idf', 'tf_idf', 'sentiment_pos', 'sentiment_pos_tfidf', 'base'} # 'base' is default

def results(algo=None):
  print("algorithm:", algo if algo in ALGOS else None)

  samples = 500 # up to 2000
  print("sample size:", samples)

  keys = glob.glob('Inspec/keys/*.key')
  res = [0]*samples

  if algo == 'textrank':
    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")
    # add PyTextRank to the spaCy pipeline
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
  elif algo == 'sentiment_pos' or algo == 'sentiment_pos_tfidf':
    sid = SentimentIntensityAnalyzer()

  for i, key in enumerate(keys[:samples]):
    # get actual keywords
    key_file = open(key)
    whitespace = re.compile(r"\s+")
    # remove whitespace and convert to lowercase
    actual = [whitespace.sub(" ", w).strip().lower() for w in key_file.readlines()]

    # get text document corresponding to current key
    num = re.findall(r'\d+', key)[0]
    doc = 'Inspec/docsutf8/{}.txt'.format(num)

    # get extracted keywords
    if algo == 'rake':
      extracted = rake(doc)
    elif algo == 'textrank':
      extracted = textrank(doc, nlp)
    elif algo == 'window':
      extracted = window(doc)
    elif algo == 'window_w_tf_idf':
      extracted = window_w_tf_idf(doc)
    elif algo == 'tf_idf':
      extracted = tf_idf(doc)
    elif algo == 'sentiment_pos':
      extracted = sentiment_pos(doc, sid)
    elif algo == 'sentiment_pos_tfidf':
      extracted = sentiment_pos_tfidf(doc, sid)
    else:
      extracted = extract(doc)

    # calculate results
    tp = len(set(extracted).intersection(set(actual))) # number of true positives
    precision = tp / len(extracted)
    recall = tp / len(actual)
    f_measure = (2 * precision * recall) / (precision + recall) if precision + recall else 0
    res[i] = (precision, recall, f_measure)
  
  # calculate average results
  avg_res = [sum(x)/len(x) for x in zip(*res)]
  print("precision: {}, recall: {}, F-measure: {}".format(*avg_res))

def main(argv):
  start = datetime.now()
  algo = argv[1] if len(argv) > 1 else 'base'
  results(algo=algo)
  end = datetime.now()
  print("runtime:", end-start)

if __name__ == "__main__":
  main(sys.argv)
