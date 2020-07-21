import re
import sys
import string
import nltk
from collections import Counter
from common import num_top_words, stopwords

sentence_splits = [".", ",", "?", "!", "\n", "\t", ";"]
word_splits = [" ", "/", "[", "]", "(", ")", "\"", "%"]

def split(string, delimiters):
  regexPattern = '|'.join(map(re.escape, delimiters))
  return re.split(regexPattern, string)

def clean(string):
  # Lower-case
  string = string.lower()

  # Remove numbers
  string = ''.join([i for i in string if not i.isdigit()]) 

  return string

tf_dict = {}

def score_words(words_to_id, id_to_words, words_to_score, graph, words_analysis):
  # Returns the words_to_score dictionary filled in
  # Implementation of this is up to us
  # Baseline is RAKE's measure

  for w in words_to_score:
    idx = words_to_id[w]

    # Degree is the sum of graph[idx][n] and graph[n][idx] minus graph[idx][idx]
    degree = 0
    freq = graph[idx][idx]

    for row in graph:
      degree += row[idx]
    for col in graph[idx]:
      degree += col
    
    degree -= freq

    degree = float(degree * tf_dict[w])

    words_to_score[w] = degree/(1 if freq == 0 else freq)

    # Sentiment
    words_to_score[w] += 6 * words_analysis[w]['sentiment']

    # Part-of-speech
    pos = Counter(words_analysis[w]['pos']).most_common(1)[0]
    pos_prioritize = {'NN', 'NNS', 'NNP'} # singular noun, plural noun, proper noun
    words_to_score[w] += 6 if pos[0] in pos_prioritize else 0
  
  return words_to_score

def extract(file_name, sid):
  f = open(file_name, "r")
  body = f.read()
  whitespace = re.compile(r"\s+")
  body = whitespace.sub(" ", body).strip()

  # Key is the phrase, val is the score
  candidate_phrases = {}

  # Map between words and IDs for (somewhat) faster lookup
  words_to_id = {}
  id_to_words = {}
  index = 0

  # Map words and their score
  words_to_score = {}

  # Map words to semtiment and part-of-speech
  words_analysis = {}

  # Split by sentences
  sentences = split(body, sentence_splits)
  for sentence in sentences:
    # Split by whitespace
    tokens = split(sentence, word_splits)

    # Get sentiment for the sentence
    sentiment = sid.polarity_scores(sentence)

    # Part-of-speech tagging by token
    pos_tokens = list(filter(None, tokens))
    pos = nltk.pos_tag(pos_tokens)

    p = ""
    for t in tokens:
      cleaned = clean(t)
      if cleaned in stopwords:
        if p != "":
          p = p[1:]
          candidate_phrases[p] = 0
          p = ""
      else:
        if cleaned:
          p = p + " " + cleaned

          if cleaned not in words_to_score:
            words_to_id[cleaned] = index
            words_to_score[cleaned] = 0
            id_to_words[index] = cleaned
            index += 1

    # Store sentiment and part-of-speech results in words_analysis
    for t in pos:
      cleaned = clean(t[0])
      if cleaned in words_analysis:
        words_analysis[cleaned]['sentiment'] += abs(sentiment['compound'])
        words_analysis[cleaned]['pos'].append(t[1])
      else:
        words_analysis[cleaned] = {
          'sentiment': abs(sentiment['compound']),
          'pos': [t[1]]
        }

  # Separate body into words
  num_words = len(words_to_score)

  # Represented as a 2D-array
  # Each keyword is associated with an index, graph[i][i] is a strict count of occurences of word at index i
  # graph[i][j] is a count of word i followed by word j
  graph = [[0 for i in range(num_words)] for j in range(num_words)]

  # Go through again to compute co-occurrences
  # This for a window of 3 words, thus each with <= 1 word in between
  for sentence in sentences:
    tokens = split(sentence, word_splits)
    num_t = len(tokens)
    prev = clean(tokens[0])
    if (num_t > 0):
        curr = clean(tokens[0])
        if curr in stopwords or curr == "":
          curr = ""
        else:
          # update the dictionary for TF
          if curr in tf_dict:
            tf_dict[curr] = tf_dict[curr] + 1
          else:
            tf_dict[curr] = 1
            graph[words_to_id[curr]][words_to_id[curr]] += 1
    for i in range(1, num_t):
      prev = clean(tokens[i-1])
      curr = clean(tokens[i])

      if curr in stopwords or curr == "":
        continue

      # update the dictionary for TF
      if curr in tf_dict:
        tf_dict[curr] = tf_dict[curr] + 1
      else:
        tf_dict[curr] = 1
      
      # Increase freq of curr
      graph[words_to_id[curr]][words_to_id[curr]] += 1

      if prev in stopwords or prev == "":
        continue

      # Increase count of prev followed by curr
      graph[words_to_id[prev]][words_to_id[curr]] += 1

  for ele in tf_dict:
    tf_dict[ele] = tf_dict[ele] / num_words

  words_to_score = score_words(words_to_id, id_to_words, words_to_score, graph, words_analysis)

  for c in candidate_phrases:
    score = 0
    for w in c.split():
      # For each word in c
      score += words_to_score[w]
    
    candidate_phrases[c] = score

  sorted_candidates = sorted(candidate_phrases, key=candidate_phrases.get, reverse=True)

  result = []
  for w in sorted_candidates:
    result.append(w)
  
  # Result is an array of candidate phrases ordered by score
  # Get the top k by returning result[:k]
  return result[:num_top_words]
