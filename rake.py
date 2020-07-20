from rake_nltk import Rake
from common import num_words, stopwords, punctuation

def rake(file_name):
  f = open(file_name, "r")
  text = f.read()
  r = Rake(
    stopwords=stopwords,
    punctuations=punctuation
  )
  r.extract_keywords_from_text(text)
  return r.get_ranked_phrases()[:num_words]

# print(rake('Inspec/docsutf8/2.txt'))
