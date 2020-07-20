import sys
import glob
import re
from base import extract
from rake import rake
from textrank import textrank
from window import extract as window
from window_w_tf_idf import extract as window_w_tf_idf
from tf_idf import extract as tf_idf
from sentiment_pos import extract as sentiment_pos

ALGOS = {'rake', 'textrank', 'window', 'window_w_tf_idf', 'td_idf', 'sentiment_pos', 'base'} # 'base' is default

def results(algo=None):
  print("algorithm:", algo if algo in ALGOS else None)

  samples = 500 # up to 2000
  print("sample size:", samples)

  keys = glob.glob('Inspec/keys/*.key')
  res = [0]*samples
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
      extracted = textrank(doc)
    elif algo == 'window':
      extracted = window(doc)
    elif algo == 'window_w_tf_idf':
      extracted = window_w_tf_idf(doc)
    elif algo == 'tf_idf':
      extracted = tf_idf(doc)
    elif algo == 'sentiment_pos':
      extracted = sentiment_pos(doc)
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
  algo = argv[1] if len(argv) > 1 else 'base'
  results(algo=algo)

if __name__ == "__main__":
  main(sys.argv)
