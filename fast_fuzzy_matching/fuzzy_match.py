"""
Fuzzy matching strings to target strings (entities) in dictionaries.
Converts strings and the dictionary keys to vectors using sklearn TF-IDF vectorizers on character n-grams. Then
generates a shortlist of match candidates from the top N nearest neighbours (cosine similarity) of dictionary key
vectors for each string (ordered from best match to worst). This list of candidates is then pruned to select the best
match using the longest common substring to length ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from itertools import groupby
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn as get_topn_nearest_neighbours

# import pylcs
import lcs

Id = Union[str, int]


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
    Class with configuration variables.
    N_GRAM_RANGE (Tuple[int, int]): Range of lengths of characters to chop strings up into before sending to the TF-IDF
                                    vectorizer, e.g. N_GRAM_RANGE = (2,3) will use all bi-grams and tri-grams with a
                                    string.
    TFIDF_SIMILARITY_THRESHOLD (float, optional): Minimum cosine similarity to a viable candidate for LCS.
                                                    Defaults to 0.1.
    N_TOP_TFIDF_CANDIDATES (int, optional): Maximum number of candidates to return that exceed the
                                            TFIDF_SIMILARITY_THRESHOLD. Defaults to 40.
    MIN_LCS_CHARACTERS (int): Minimum length of the string to qualify for matching to the target strings.
    MIN_LCS_LENGTH_RATIO (float): Minimum ratio of string length to target string length for an string <> target string
                                    match to qualify.
    LCS_SIMILARITY_THRESHOLD (float, optional): Minimum LCS match ratio to accept classification.
                                                Defaults to 0.9.
    USE_THREADS (int): Whether to use threads to parallelise the work to find the N_TOP_TFIDF_CANDIDATES for each
                        string.
    N_THREADS (int): Number of threads to use when finding the N_TOP_TFIDF_CANDIDATES. Increasing the number of threads
                    reduces the run time, but there becomes a trade off in production where there may be 'thread
                    congestion'.
    """

    N_GRAM_RANGE: Tuple[int, int] = (2, 4)
    MIN_DOCUMENT_FREQ: int = 1
    TFIDF_SIMILARITY_THRESHOLD: float = 0.1
    N_TOP_TFIDF_CANDIDATES: int = 40
    MIN_LCS_CHARACTERS: int = 4
    MIN_LCS_LENGTH_RATIO: float = 0.7
    LCS_SIMILARITY_THRESHOLD: float = 0.8
    USE_THREADS: bool = True
    N_THREADS: int = 4
    STRING_PREPROCESSOR: Optional[Callable[[str], str]] = None


class FuzzyMatch:
    """
    Fuzzy matches strings to target entities.

    documents (List[str]): The documents to match against.
    vectorizer (TfidfVectorizer): A vectorizer for each lookup that has already been fitted.
    document_vectors (csr_matrix): Document vector matrices calculated from applying the vectorizer transform.

    Example:
        documents = ["SMARTEST ENERGY", "SMARTPIG"]
        fuzzy_matcher = FuzzyMatch(documents=documents)
        strings = pd.Series(['Smart Piggie', 'the smartest energy'], index=[10, 20])
        fuzzy_matcher.get_lookup_matches(strings=strings)
        >>> {
                10: EntityMatch(entity='SMARTPIG', confidence=1.0),
                20: EntityMatch(entity='SMARTEST ENERGY', confidence=1.0)
            }
        fuzzy_matcher.get_lookup_match('Smart Piggie')
        >>> EntityMatch(entity='SMARTPIG', confidence=1.0)
    """

    def __init__(
        self,
        documents: List[str],
        config: FuzzyMatchConfig = FuzzyMatchConfig(),
    ):
        self._config = config
        self._fit_vectorizer(documents)

    @property
    def config(self):
        return self._config

    def apply_string_preprocessor(self, strings: List[str]) -> str:
        """Apply string preprocessor to a string.

        Args:
            string (str): String to preprocess.

        Returns:
            str: Preprocessed string.
        """
        if self._config.STRING_PREPROCESSOR is not None:
            return list(map(self._config.STRING_PREPROCESSOR, strings))
        return strings

    def _fit_vectorizer(self, documents: Sequence[str]):
        """Fit a TF-IDF vectorizer on all documents, each lookup has an associated vectorizer fit on that lookup
        than can be used to turn strings into numerical vectors to search against those documents.

        Args:
            documents (List[str]): documents.
        """
        self.documents = documents

        self.vectorizer = TfidfVectorizer(
            min_df=self._config.MIN_DOCUMENT_FREQ,
            analyzer=partial(tokenize_string_chars, ngram_range=self._config.N_GRAM_RANGE),
            lowercase=False,
            dtype=np.float32,
            norm='l2',
            use_idf=True,
        )
        self.vectorizer.fit(self.apply_string_preprocessor(documents))
        self.document_vectors = self.vectorizer.transform(documents)

    def get_document_matches(self, strings: pd.Series) -> Dict[Id, EntityMatch]:
        """
        For a series of strings, find the closest match in the lookup specified along with the match confidence.
        Args:
            strings (pd.Series): Strings to match, the index of the series is used as an identifier when returning the
                                match results.
            lookup_name (str): Name of the lookup to search for entity matches in.
        Returns:
            Dict[Id, EntityMatch]: Txn Id and fuzzy match (entity and confidence).
        """
        if not isinstance(strings, pd.Series):
            raise TypeError('documents must be a pd.Series')

        if strings.empty:
            return {}

        ids = list(strings.index)
        embedding_vectors = self.vectorizer.transform(strings)

        match_candidates = self._get_match_candidates(embedding_vectors=embedding_vectors)
        if match_candidates.nnz == 0:
            # No match candidates
            return {}

        match_results = self._get_lcs_matches(match_candidates=match_candidates, ids=ids, strings=strings)
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
        embedding_vectors = self.vectorizer.transform(self.apply_string_preprocessor([string]))

        match_candidates = self._get_match_candidates(embedding_vectors=embedding_vectors)
        if match_candidates.nnz == 0:
            # No match candidates
            return

        match_result = self._get_lcs_matches(match_candidates=match_candidates, ids=[0], strings=[string])
        if match_result:
            return match_result[0]

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

        no_space_entity_matches = [string.replace(' ', '') for string in entity_matches]

        best_match_idx, confidence = lcs.get_lcs_best_match_idx(
            no_space_str=string.replace(' ', ''),
            no_space_entity_matches=no_space_entity_matches,
            min_characters=self._config.MIN_LCS_CHARACTERS,
            min_length_ratio=self._config.MIN_LCS_LENGTH_RATIO,
        )

        # best_match_idx of -1 indicates no valid matches found.
        if confidence >= self._config.LCS_SIMILARITY_THRESHOLD and best_match_idx != -1:
            return EntityMatch(entity=entity_matches[best_match_idx], confidence=confidence)

    def _get_lcs_matches(
        self, match_candidates: csr_matrix, ids: List[Id], strings: List[str]
    ) -> Dict[Id, EntityMatch]:
        """Prunes the match candidates using LCS to identify the best match for each string.
        Args:
            match_candidates (csr_matrix): Coordinate format sparse matrix, match candidates ~ (data, (row, col)).
                                    The row and col are the row and column position of the value in the matrix, data
                                    represents the numerical value at that point. The match candidates is ordered from
                                    best match to worst match for each successive string.
            ids (List[Id]): String identifiers.
            strings (List[str]): Strings to match.
            idx_entity_map (dict): Dictionary mapping the column index from the match candidates sparse matrix to the
                                    corresponding entity in the lookup.
        Returns:
            Dict[Id, EntityMatch]: Id mapping to the fuzzy match (entity and confidence).
        """
        lcs_best_matches = {}
        # Assumes match_candidates.row is ordered - it's a pre-requisite of itertools.groupby.
        for string_idx, document_idxs in groupby(zip(match_candidates.row, match_candidates.col), key=lambda x: x[0]):
            string = strings[string_idx]
            matches = [self.documents[document_idx] for _, document_idx in document_idxs]
            lcs_best_match = self._get_lcs_best_match(string=string, entity_matches=matches)
            if lcs_best_match:
                lcs_best_matches[ids[string_idx]] = lcs_best_match
        return lcs_best_matches

    def _get_match_candidates(self, embedding_vectors: csr_matrix) -> csr_matrix:
        """
        Finds the top N nearest neighbours in the matrices of document-vectors using cosine similarity (ordered from
        best match to worst).
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
            ntop=self._config.N_TOP_TFIDF_CANDIDATES,
            use_threads=self._config.USE_THREADS,
            n_jobs=self._config.N_THREADS,
            lower_bound=self._config.TFIDF_SIMILARITY_THRESHOLD,
            return_best_ntop=True,
        )[0].tocoo()
        return match_candidates


if __name__ == "__main__":
    import json

    with open('fast_fuzzy_matching/credit_fuzzy.json') as open_file:
        documents = json.load(open_file).keys()

    fm = FuzzyMatch(documents=pd.Series(documents), config=FuzzyMatchConfig(N_GRAM_RANGE=(2, 4)))

    print(fm.get_document_match("GIVEAWAY GUYS"))
    # from timeit import timeit

    # import string_utils

    # print('cpp: ', timeit(lambda: string_utils.get_ngrams('abcdefhiglmnop' * 20, 3, True), number=1000000))
    # print('py: ', timeit(lambda: get_ngrams('abcdefhiglmnop' * 20, n=3, pad=True), number=1000000))

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
