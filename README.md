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
Use the following command to run a keyword extraction algorithm on text samples and get the precision, recall, and F-measure:
```
python test.py <algo>
```
`algo` can be `base` (default), `window`, `sentiment_pos`, `rake`, `textrank`.

For example,
```
python test.py sentiment_pos
```
