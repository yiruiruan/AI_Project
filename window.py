import re
import sys
import string
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

def score_words(words_to_id, id_to_words, words_to_score, graph):
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

    words_to_score[w] = degree/(1 if freq == 0 else freq)
  
  return words_to_score

def extract(file_name):
  f = open(file_name, "r")
  body = f.read()

  # Key is the phrase, val is the score
  candidate_phrases = {}

  # Map between words and IDs for (somewhat) faster lookup
  words_to_id = {}
  id_to_words = {}
  index = 0

  # Map words and their score
  words_to_score = {}

  # Split by sentences
  sentences = split(body, sentence_splits)
  for sentence in sentences:
    # Split by whitespace
    tokens = split(sentence, word_splits)

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
        curr = clean(tokens[1])
        if curr in stopwords or curr == "":
          curr = ""
        else:
          graph[words_to_id[curr]][words_to_id[curr]] += 1
    if (num_t > 1):
      prev = prev = clean(tokens[0])
      curr = clean(tokens[1])
      if curr in stopwords or curr == "":
        curr = ""
      else:
        graph[words_to_id[curr]][words_to_id[curr]] += 1
        if prev in stopwords or prev == "":
          prev = ""
        else:
          graph[words_to_id[prev]][words_to_id[curr]] += 1

    for i in range(2, num_t):
      prev_2 = clean(tokens[i-2])
      prev = clean(tokens[i-1])
      curr = clean(tokens[i])

      if curr in stopwords or curr == "":
        continue
      
      # Increase freq of curr
      graph[words_to_id[curr]][words_to_id[curr]] += 1

      if prev in stopwords or prev == "":
        prev = ""
      else:
        # Increase count of prev followed by curr
        graph[words_to_id[prev]][words_to_id[curr]] += 1

      if prev_2 in stopwords or prev_2 == "":
        continue

      # Increase count of prev followed by curr
      graph[words_to_id[prev_2]][words_to_id[curr]] += 1

  
  words_to_score = score_words(words_to_id, id_to_words, words_to_score, graph)

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