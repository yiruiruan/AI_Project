import re
from common import num_top_words

def textrank(file_name, nlp):
  f = open(file_name, "r")
  text = f.read()
  whitespace = re.compile(r"\s+")
  text = whitespace.sub(" ", text).strip()

  doc = nlp(text)
  top_ranked = [p.text for p in doc._.phrases]
  return top_ranked[:num_top_words]

# print(textrank('Inspec/docsutf8/2.txt'))
