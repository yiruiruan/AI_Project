import spacy
import pytextrank
import re
from common import num_top_words

def textrank(file_name):
  f = open(file_name, "r")
  text = f.read()
  whitespace = re.compile(r"\s+")
  text = whitespace.sub(" ", text).strip()

  # load a spaCy model, depending on language, scale, etc.
  nlp = spacy.load("en_core_web_sm")
  # add PyTextRank to the spaCy pipeline
  tr = pytextrank.TextRank()
  nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

  doc = nlp(text)
  top_ranked = [p.text for p in doc._.phrases]
  return top_ranked[:num_top_words]

# print(textrank('Inspec/docsutf8/2.txt'))
