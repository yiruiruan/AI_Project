import glob
import re
from base import extract

def results():
  samples = 500 # up to 2000
  print("sample size: {}".format(samples))
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

results()

