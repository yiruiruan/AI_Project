import re
from rake_nltk import Rake
from common import num_top_words, stopwords, punctuation

def rake(file_name):
  f = open(file_name, "r")
  text = f.read()
  whitespace = re.compile(r"\s+")
  text = whitespace.sub(" ", text).strip()
  r = Rake(
    stopwords=stopwords,
    punctuations=punctuation
  )
  r.extract_keywords_from_text(text)
  return r.get_ranked_phrases()[:num_top_words]

# print(rake('Inspec/docsutf8/2.txt'))
