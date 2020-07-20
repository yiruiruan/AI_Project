# CS486 Project

## Setup
*The following are only necessary for benchmarking with existing keyword extraction algorithms.*
- Install [rake-nltk](https://pypi.org/project/rake-nltk/)
  ```
  pip install rake-nltk
  ```
- Install [pytextrank](https://pypi.org/project/pytextrank/)
  ```
  pip install pytextrank
  python -m spacy download en_core_web_sm
  ```

## Run Algorithms on Text Samples
Use the following command to run our keyword extraction algorithm on text samples and get the precision, recall, and F-measure:
```
python test.py
```

To run an existing keyword extraction algorithm and get the results, use one of the following commands:
```
python test.py rake
python test.py textrank
```
