# fuzzy-lightning

Fuzzy lightning is a fast and customizable package for finding the closest matches in a list of target strings (documents) using fuzzy string matching. It is particularly effective for short string matching against large document sets, and includes the fastest implementation of the Damerau-Levenshtein and longest common substring algorithms in its class.

## Introduction
Fuzzy lightning works by:
1. Converts strings to embedding vectors using an sklearn TF-IDF vectorizer on character n-grams.
2. Generates a shortlist of match candidates from the top N nearest neighbours (using cosine similarity).
3. This list of candidates is then pruned to select the best match using the longest common substring to length ratio.

### Quick Start

#### Installation

`pip install fuzzy-lightning`

Finding the closest matches in a list of documents for a list of input strings:

```
from fuzzy_lightning import FuzzyMatch

documents = ["SMARTEST ENERGY", "SMARTPIG"]
fuzzy_matcher = FuzzyMatch(documents=documents)
strings = ['SMARTPIGGIE', 'THE SMARTEST ENERGY']
matches = fuzzy_matcher.get_document_matches(strings=strings)
print(matches)
>>> [
    DocumentMatch(match='SMARTPIG', confidence=1.0),
    DocumentMatch(match='SMARTEST ENERGY', confidence=1.0)
]
```

The output is a list of DocumentMatch objects, each with a match attribute that contains the closest matching document and a confidence attribute that represents the confidence of the match (a value between 0 and 1):

If you want to find the closest match for a single string, you can use the get_lookup_match method:

```
match = fuzzy_matcher.get_lookup_match('SMART PIGGIE')
print(match)
>>> DocumentMatch(match='SMARTPIG', confidence=1.0)
```

### Configuration

The FuzzyMatch class has a number of configurable parameters that you can set using the `FuzzyMatchConfig` class. 

- **n_gram_range** (Tuple[int, int]): Range of lengths of n-grams to use with the TF-IDF vectorizer. For example,
    n_gram_range = (2, 3) will use bi-grams and tri-grams.
- **min_document_freq** (int, optional): Minimum number of documents a term must appear in to be considered.
    Defaults to 1.
- **tfidf_similarity_threshold** (float, optional): Minimum cosine similarity to a viable candidate for LCS.
    Defaults to 0.1.
- **n_top_candidates** (int, optional): Maximum number of candidates to return that exceed the
    similarity_threshold. Defaults to 40.
- **lcs_min_characters** (int): Minimum length of the string to qualify for matching to the target strings.
- **lcs_min_length_ratio** (float): Minimum ratio of string length to target string length for an string <> target
    string match to qualify.
- **lcs_similarity_threshold** (float, optional): Minimum LCS match ratio to accept classification.
use_threads** (bool): Whether to use threads to parallelise the work to find the n_top_candidates for each
    string.
- **n_threads** (int): Number of threads to use when finding the n_top_candidates. Increasing the number of threads
    reduces the run time, but there becomes a trade off in production where there may be 'thread congestion'.
- **string_preprocessor** (Optional[Callable[[str], str]]): A callable that takes in a string and returns a processed
    string. This can be used to perform any preprocessing steps on the input strings before they are compared.

For example, to change the range of n-grams used by the TF-IDF vectorizer, and to add some string preprocessing prior
to the fuzzy matching you can do the following:

```
from fuzzy_lightning import FuzzyMatch, FuzzyMatchConfig

def preprocessor(string: str) -> str:
    return string.lower().replace(" ", "")

config = FuzzyMatchConfig(n_gram_range=(1, 2), string_preprocessor=preprocessor)
fuzzy_matcher = FuzzyMatch(documents=documents, config=config)
```

## Longest Common Substring

Finds the longest substring that is common to two strings. It is used to calculate the confidence of the fuzzy match.

```
from fuzzy_lightning import lcs
lcs.longest_common_substring_length('beersteinbeer', 'stein')
>>> 5
```

## Edit Distance Algorithms

### Damerau-Levenshtein

The Damerau-Levenshtein algorithm is a string edit distance algorithm calculates the minimum number of operations (insertions, deletions, substitutions, and transpositions) required to transform one string into another. Basically Levenshtein but also
allow for transpositions.

```
from fuzzy_lightning import edit_distance as ed
dist = ed.damerau_levenshtein('my nam is spetl wrrong', 'my name is spelt wrong')
print(dist)
>>> 3
```

## Appendix

### Why is this super fast?

1. C++
2. Dynamic Programming
3. Cache locality benefits of using a 1D array to mimic the behaviour of a 2D array
