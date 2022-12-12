"""
Fuzzy matching strings to target strings (entities) in dictionaries.
Converts strings and the dictionary keys to vectors using sklearn TF-IDF vectorizers on character n-grams. Then
generates a shortlist of match candidates from the top N nearest neighbours (cosine similarity) of dictionary key
vectors for each string (ordered from best match to worst). This list of candidates is then pruned to select the best
match using the longest common substring to length ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial, singledispatchmethod
from itertools import groupby
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn as get_topn_nearest_neighbours

import lcs
from fast_fuzzy_matching.profile_utils import profile


def get_ngrams(string: str, n: int, pad: bool = True) -> List[str]:
    """Get all substrings of length n in the string.

    Args:
        string (str): String to get ngrams from.
        n (int): Length of ngram.
        pad (bool): Whether to pad the beginning and end of the string with '-'.

    Returns:
        List[str]: All ngrams extracted from the string.

    Example:
        get_ngrams('ABCD', 2) = ['AB', 'BC', 'CD']
    """
    if pad:
        string = '-' + string + '-'
    return [string[i : i + n] for i in range(len(string) - n + 1)]


def tokenize_string_chars(string: str, ngram_range: Tuple[int, int]) -> List[str]:
    """Partition strings up into a list of character ngrams. You can pass a function that transforms (e.g. removes
    spaces and uppercases) through the text_preprocessor argument.

    Args:
        string (str): String to tokenize.
        ngram_range (Tuple[int, int]): Minimum and maximum number of characters to split the string into.

    Returns:
        List[str]: Character ngrams.

    Example:
        tokenize_string_chars('ABCD', (2, 4)) = ['-A', 'AB', 'BC', 'CD', 'D-', '-AB', 'ABC', 'BCD', 'CD-', '-ABC',
                                                'ABCD', 'BCD-']
    """
    tokens = []
    if not string:
        return tokens

    min_char_len, max_char_len = ngram_range
    for ngram_len in range(min_char_len, max_char_len + 1):
        ngrams = get_ngrams(string=string, n=ngram_len, pad=True)
        tokens.extend(ngrams)
    return tokens


@dataclass(frozen=True)
class EntityMatch:
    """Output from individual fuzzy matches."""

    entity: str
    confidence: float = field(default=0.0, metadata={"validator": "float_between_0_and_1"})

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be a float between 0 and 1")


@dataclass
class FuzzyMatchConfig:
    """
    Class with configuration variables for the FuzzyMatch algorithm.

    Attributes:
        tfidf (TFIDFConfig): Configuration for the TF-IDF vectorizer.
        lcs (LCSConfig): Configuration for the LCS algorithm.
        threads (NearestNeighbourSearchConfig): Configuration for using threads.
        string_preprocessor (Optional[Callable[[str], str]]): A callable that takes in a string and returns a processed
            string. This can be used to perform any preprocessing steps on the input strings before they are compared.
    """

    @dataclass
    class TFIDF:
        """
        Class with configuration variables for the TF-IDF vectorizer.

        Attributes:
            n_gram_range (Tuple[int, int]): Range of lengths of n-grams to use with the TF-IDF vectorizer. For example,
                n_gram_range = (2, 3) will use bi-grams and tri-grams.
            min_document_freq (int, optional): Minimum number of documents a term must appear in to be considered.
                Defaults to 1.
            similarity_threshold (float, optional): Minimum cosine similarity to a viable candidate for LCS.
                Defaults to 0.1.
            n_top_candidates (int, optional): Maximum number of candidates to return that exceed the
                similarity_threshold. Defaults to 40.
        """

        n_gram_range: Tuple[int, int] = (2, 4)
        min_document_freq: int = 1
        similarity_threshold: float = 0.1
        n_top_candidates: int = 40

    @dataclass
    class LCS:
        """
        Class with configuration variables for the LCS algorithm.

        Attributes:
            min_characters (int): Minimum length of the string to qualify for matching to the target strings.
            min_length_ratio (float): Minimum ratio of string length to target string length for an string <> target
                string match to qualify.
            similarity_threshold (float, optional): Minimum LCS match ratio to accept classification.
        """

        min_characters: int = 4
        min_length_ratio: float = 0.7
        similarity_threshold: float = 0.8

    @dataclass
    class NNSearch:
        """
        Class with configuration variables for finding the nearest neighbour embedding vectors using cosine similarity.
        You can experiment with using threading to optimise latency for your use case.

        Attributes:
            use_threads (bool): Whether to use threads to parallelise the work to find the n_top_candidates for each
                string.
            n_threads (int): Number of threads to use when finding the n_top_candidates. Increasing the number of threads
                reduces the run time, but there becomes a trade off in production where there may be 'thread congestion'.
        """

        use_threads: bool = True
        n_threads: int = 4

    tfidf: TFIDF = field(default_factory=TFIDF)
    lcs: LCS = field(default_factory=LCS)
    nn_search: NNSearch = field(default_factory=NNSearch)
    string_preprocessor: Optional[Callable[[str], str]] = None


class FuzzyMatch:
    """
    Fuzzy string matching for a list of documents.

    Example:
        documents = ["SMARTEST ENERGY", "SMARTPIG"]
        fuzzy_matcher = FuzzyMatch(documents=documents)
        strings = ['Smart Piggie', 'the smartest energy']
        fuzzy_matcher.get_document_matches(strings=strings)
        >>> [
                EntityMatch(entity='SMARTPIG', confidence=1.0),
                EntityMatch(entity='SMARTEST ENERGY', confidence=1.0)
            ]
        fuzzy_matcher.get_lookup_match('SMART PIGGIE')
        >>> EntityMatch(entity='SMARTPIG', confidence=1.0)
    """

    def __init__(
        self,
        documents: List[str],
        config: FuzzyMatchConfig = FuzzyMatchConfig(),
    ):
        """
        Initialize the FuzzyMatch object with a list of documents and an optional configuration object.

        Args:
            documents (List[str]): A list of strings representing the documents to match against.
            config (FuzzyMatchConfig): An optional configuration object that specifies settings for the fuzzy match.
        """
        self._config = config
        self._fit_vectorizer(documents)

    @property
    def config(self):
        return self._config

    @singledispatchmethod
    def apply_string_preprocessor(self, arg):
        """Apply string preprocessor"""
        raise TypeError(f"Type {type(arg)} is not supported by apply_string_preprocessor")

    @apply_string_preprocessor.register
    def _(self, arg: str):
        if self.config.string_preprocessor is not None:
            return self.config.string_preprocessor(arg)
        return arg

    @apply_string_preprocessor.register
    def _(self, arg: list):
        if self.config.string_preprocessor is not None:
            return list(map(self.config.string_preprocessor, arg))
        return arg

    def _fit_vectorizer(self, documents: List[str]):
        """Fit a TF-IDF vectorizer on all documents, each lookup has an associated vectorizer fit on that lookup
        than can be used to turn strings into numerical vectors to search against those documents.

        Args:
            documents (List[str]): documents.
        """
        self.documents = documents
        preprocessed_documents = self.apply_string_preprocessor(documents)

        self.vectorizer = TfidfVectorizer(
            min_df=self.config.tfidf.min_document_freq,
            analyzer=partial(tokenize_string_chars, ngram_range=self.config.tfidf.n_gram_range),
            lowercase=False,
            dtype=np.float32,
            norm='l2',
            use_idf=True,
        )
        self.vectorizer.fit(preprocessed_documents)
        self.document_vectors = self.vectorizer.transform(preprocessed_documents)

    def get_document_matches(self, strings: List[str]) -> List[EntityMatch]:
        """
        For a series of strings, find the closest match in the lookup specified along with the match confidence.
        Args:
            strings (pd.Series): Strings to match, the index of the series is used as an identifier when returning the
                                match results.
            lookup_name (str): Name of the lookup to search for entity matches in.
        Returns:
            List[EntityMatch]: Id and fuzzy match (entity and confidence).
        """
        if not strings:
            return []

        embedding_vectors = self.vectorizer.transform(self.apply_string_preprocessor(strings))
        match_candidates = self._get_match_candidates(embedding_vectors=embedding_vectors)
        match_results = self._get_lcs_matches(match_candidates=match_candidates, strings=strings)
        return match_results

    def get_document_match(self, string: str) -> Optional[EntityMatch]:
        """
        Find the closest match in the lookup specified along with the match confidence.

        Args:
            string (str): String to match.
            lookup_name (str): Name of the lookup to search for entity matches in.
        Returns:
            EntityMatch: Fuzzy match (entity and confidence).
        """
        embedding_vectors = self.vectorizer.transform([self.apply_string_preprocessor(string)])
        match_candidates = self._get_match_candidates(embedding_vectors=embedding_vectors)
        return self._get_lcs_matches(match_candidates=match_candidates, strings=[string])[0]

    def _get_lcs_best_match(self, string: str, entity_matches: List[str]) -> Optional[EntityMatch]:
        """Selects the best match from a set of candidates using the highest match ratio of longest common substring to
            string length.

        Args:
            string (str): String to find best LCS match for.
            entity_matches (List[str]): The entity match candidates identified from the TF-IDF top nearest neighbours to
                                        the string.
        Returns:
            Optional[EntityMatch]: Fuzzy match (entity and confidence).
        """
        best_match_idx, confidence = lcs.get_lcs_best_match_idx(
            preprocessed_str=self.apply_string_preprocessor(string),
            preprocessed_entity_matches=self.apply_string_preprocessor(entity_matches),
            min_characters=self.config.lcs.min_characters,
            min_length_ratio=self.config.lcs.min_length_ratio,
        )

        # best_match_idx of -1 indicates no valid matches found.
        if confidence >= self.config.lcs.similarity_threshold and best_match_idx != -1:
            return EntityMatch(entity=entity_matches[best_match_idx], confidence=confidence)

    def _get_lcs_matches(self, match_candidates: csr_matrix, strings: List[str]) -> List[Optional[EntityMatch]]:
        """Prunes the match candidates using LCS to identify the best match for each string.
        Args:
            match_candidates (csr_matrix): Coordinate format sparse matrix, match candidates ~ (data, (row, col)).
                                    The row and col are the row and column position of the value in the matrix, data
                                    represents the numerical value at that point. The match candidates is ordered from
                                    best match to worst match for each successive string.
            strings (List[str]): Strings to match.

        Returns:
            List[Optional[EntityMatch]]: Fuzzy matches (entity and confidence).
        """
        lcs_best_matches = {}
        # Assumes match_candidates.row is ordered - it's a pre-requisite of itertools.groupby.
        for string_idx, document_idxs in groupby(zip(match_candidates.row, match_candidates.col), key=lambda x: x[0]):
            string = strings[string_idx]
            matches = [self.documents[document_idx] for _, document_idx in document_idxs]
            if match := self._get_lcs_best_match(string=string, entity_matches=matches):
                lcs_best_matches[string] = match

        # match_candidates might not contain all strings, so we need to return a list of EntityMatch in the same order
        return [lcs_best_matches.get(string) for string in strings]

    def _get_match_candidates(self, embedding_vectors: csr_matrix) -> csr_matrix:
        """
        Finds the top N nearest neighbours in the matrices of document-vectors using cosine similarity (ordered from
        best match to worst).

        If the a strings embeddings has no similarity scores larger than tfidf_similarity_threshold, then it's index
        will not be present in the any rows of the csr matrix returned.

        Args:
            embedding_vectors (csr_matrix): Sparse matrix of TF-IDF vectors for each string to match.
        Returns:
            csr_matrix: Of the form (row, column) value. Where the row is the index of the string, the column is
                        the index of the matched entity, and the value is the cosine similarity between the two
                        vectors.
        Example:
            (0, 0)	0.2
            (0, 1)	0.4
            (0, 2)	0.5
            (1, 20)	0.3
            (2, 25)	0.55
            (2, 26)	0.2
            (2, 30)	0.1
            (2, 40)	0.05
            ...
        """
        match_candidates = get_topn_nearest_neighbours(
            embedding_vectors,
            self.document_vectors.transpose(),
            ntop=self.config.tfidf.n_top_candidates,
            use_threads=self.config.nn_search.use_threads,
            n_jobs=self.config.nn_search.n_threads,
            lower_bound=self.config.tfidf.similarity_threshold,
            return_best_ntop=True,
        )[0].tocoo()
        return match_candidates


if __name__ == "__main__":
    import json

    with open('fast_fuzzy_matching/credit_fuzzy.json') as open_file:
        documents = json.load(open_file).keys()

    fm = FuzzyMatch(documents=pd.Series(documents), config=FuzzyMatchConfig(N_GRAM_RANGE=(2, 4)))

    # print(fm.get_document_match("GIVEAWAY GUYS"))
    # from timeit import timeit

    # import string_utils

    # print('cpp: ', timeit(lambda: string_utils.get_ngrams('abcdefhiglmnop' * 20, 3, True), number=1000000))
    # print('py: ', timeit(lambda: get_ngrams('abcdefhiglmnop' * 20, n=3, pad=True), number=1000000))

    @profile
    def case():
        cases = ["GIVEAWAY GUYS"] * 1000

        for case in cases:
            fm.get_document_match(case)

    case()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import umap
# from sklearn.preprocessing import StandardScaler

# LOOKUP_NAME = 'debit_fuzzy'

# text_preprocessor = lambda text: text.replace(' AND ', ' & ').replace(' ', '').upper()
# fuzzy_match_lookup_names = [
#     'debit_fuzzy',
#     'credit_fuzzy',
# ]

# documents = read_documents()

# fuzzy_match_documents = {name: lookup for name, lookup in documents.items() if name in fuzzy_match_lookup_names}
# fm = FuzzyMatch(
#     documents=fuzzy_match_documents, text_preprocessor=text_preprocessor, config=FuzzyMatchConfig(N_GRAM_RANGE=(2, 3))
# )


# # string_embeddings = fm.vectorizers[LOOKUP_NAME].transform(documents[LOOKUP_NAME].keys()).toarray()
# # print(string_embeddings.shape)

# # reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='wminkowski')
# # scaled_data = StandardScaler().fit_transform(string_embeddings)

# # print(scaled_data.shape)
# # embedding = reducer.fit_transform(scaled_data)
# # print(embedding.shape)

# # plt.scatter(
# #     embedding[:, 0],
# #     embedding[:, 1],
# #     # c=[sns.color_palette()[x] for x in penguins.species_short.map({"Adelie":0, "Chinstrap":1, "Gentoo":2})]
# # )
# # plt.gca().set_aspect('equal', 'datalim')
# # plt.title('UMAP projection of the TF-IDF Merchant Embeddings', fontsize=24)
# # plt.show()


# import pyinstrument

# profiler_kwargs = dict(
#     pyinst_interval=0.0001, pyinst_color=True, pyinst_show_all=True, pyinst_timeline=False, cprof_sort='cumtime'
# )
# profiler = pyinstrument.Profiler(interval=profiler_kwargs['pyinst_interval'])
# profiler.start()

# cases = ["SAMSUNG", "118 118 MON", "MACDONALDS SOUTHGATE"] * 334

# for case in cases:
#     fm.get_lookup_match(case, lookup_name=LOOKUP_NAME)
#     # tokenize_string_chars(case, (2,4))

# profiler.stop()
# prof_output = profiler.output_text(
#     color=profiler_kwargs['pyinst_color'],
#     show_all=profiler_kwargs['pyinst_show_all'],
#     timeline=profiler_kwargs['pyinst_timeline'],
# )
# print(prof_output)
