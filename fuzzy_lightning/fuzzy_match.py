"""
Fuzzy matching strings to target strings (document matches).
Converts strings to vectors using sklearn TF-IDF vectorizers on character n-grams. The generates a shortlist of match
candidates from the top N nearest neighbours (cosine similarity) of dictionary key vectors for each string (ordered
from best match to worst). This list of candidates is then pruned to select the best match using the longest common
substring to length ratio.
"""

from __future__ import annotations

import itertools as it
from dataclasses import dataclass
from functools import partial, singledispatchmethod
from itertools import groupby
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn as get_topn_nearest_neighbours

import lcs
from fuzzy_lightning.tokenize import tokenize_string_chars


@dataclass
class DocumentMatch:
    """Output from individual fuzzy matches."""

    match: str
    confidence: float

    def __post_init__(self):
        self.confidence = round(self.confidence, 5)
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence was {self.confidence}, must be a float between 0 and 1")


@dataclass
class FuzzyMatchConfig:
    """
    Class with configuration variables for the FuzzyMatch algorithm.

    Attributes:
        n_gram_range (Tuple[int, int]): Range of lengths of n-grams to use with the TF-IDF vectorizer. For example,
            n_gram_range = (2, 3) will use bi-grams and tri-grams.
        min_document_freq (int, optional): Minimum number of documents a term must appear in to be considered.
            Defaults to 1.
        tfidf_similarity_threshold (float, optional): Minimum cosine similarity to a viable candidate for LCS.
            Defaults to 0.1.
        n_top_candidates (int, optional): Maximum number of candidates to return that exceed the
            similarity_threshold. Defaults to 40.
        lcs_min_characters (int): Minimum length of the string to qualify for matching to the target strings.
        lcs_min_length_ratio (float): Minimum ratio of string length to target string length for an string <> target
            string match to qualify.
        lcs_similarity_threshold (float, optional): Minimum LCS match ratio to accept classification.
        use_threads (bool): Whether to use threads to parallelise the work to find the n_top_candidates for each
            string.
        n_threads (int): Number of threads to use when finding the n_top_candidates. Increasing the number of threads
            reduces the run time, but there becomes a trade off in production where there may be 'thread congestion'.
        pad_string (bool): Whether to pad the start and end of strings with '-' before tokenization. This gives a fairer
            amount of importance to ngrams at the start and end of strings.
        string_preprocessor (Optional[Callable[[str], str]]): A callable that takes in a string and returns a processed
            string. This can be used to perform any preprocessing steps on the input strings before they are compared.
    """

    # tifidf config
    n_gram_range: Tuple[int, int] = (2, 4)
    min_document_freq: int = 1
    tfidf_similarity_threshold: float = 0.1
    n_top_candidates: int = 40
    # lcs config
    lcs_min_characters: int = 4
    lcs_min_length_ratio: float = 0.7
    lcs_similarity_threshold: float = 0.8
    # nearest neighour search config
    use_threads: bool = True
    n_threads: int = 4
    # preprocessing config
    pad_string: bool = True
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
                DocumentMatch(match='SMARTPIG', confidence=1.0),
                DocumentMatch(match='SMARTEST ENERGY', confidence=1.0)
            ]
        fuzzy_matcher.get_lookup_match('SMART PIGGIE')
        >>> DocumentMatch(match='SMARTPIG', confidence=1.0)
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
            min_df=self.config.min_document_freq,
            analyzer=partial(tokenize_string_chars, ngram_range=self.config.n_gram_range),
            lowercase=False,
            dtype=np.float32,
            norm='l2',
            use_idf=True,
        )
        self.vectorizer.fit(preprocessed_documents)
        self.document_vectors = self.vectorizer.transform(preprocessed_documents)

    def get_document_matches(self, strings: List[str], prune_with_lcs: bool = True) -> List[Optional[DocumentMatch]]:
        """
        For a series of strings, find the closest match in the lookup specified along with the match confidence.
        Args:
            strings (List[str]): Strings to match.
            prune_with_lcs (bool): Whether to use LCS (match ratio) confidence instead of default TFIDF
            (cosine similarity).
        Returns:
            List[DocumentMatch]: Fuzzy matches (match and confidence).
        """
        if not strings:
            return []

        embedding_vectors = self.vectorizer.transform(self.apply_string_preprocessor(strings))
        str_similarity_matrix = self.get_string_similarity_matrix(embedding_vectors=embedding_vectors)
        if str_similarity_matrix.nnz == 0:
            # No matches
            return list(it.repeat(None, len(strings)))

        if prune_with_lcs:
            return self._get_lcs_matches(str_similarity_matrix=str_similarity_matrix, strings=strings)
        else:
            matches = self._get_tfidf_matches(str_similarity_matrix)
            return matches

    def get_document_match(self, string: str, prune_with_lcs: bool = True) -> Optional[DocumentMatch]:
        """
        Find the closest match in the lookup specified along with the match confidence.

        Args:
            string (str): String to match.
            prune_with_lcs (bool): Whether to use LCS (match ratio) confidence instead of default TFIDF
            (cosine similarity).
        Returns:
            Optional[DocumentMatch]: Fuzzy match (match and confidence).
        """
        embedding_vectors = self.vectorizer.transform([self.apply_string_preprocessor(string)])
        str_similarity_matrix = self.get_string_similarity_matrix(embedding_vectors=embedding_vectors)

        if str_similarity_matrix.nnz == 0:
            # No matches
            return None

        row_matches = str_similarity_matrix.getrow(0)
        document_matches = [self.documents[doc_idx] for doc_idx in row_matches.indices]

        if prune_with_lcs:
            return self._get_lcs_best_match(string=string, document_matches=document_matches)
        else:
            max_conf_doc_idx = row_matches.data.argmax()
            match = document_matches[max_conf_doc_idx]
            confidence = row_matches.data[max_conf_doc_idx]
            if confidence >= self.config.tfidf_similarity_threshold:
                return DocumentMatch(match=match, confidence=confidence)

    def get_string_similarity_matrix(self, embedding_vectors: csr_matrix) -> csr_matrix:
        """
        Finds the top N nearest neighbours in the matrices of document-vectors using cosine similarity (ordered from
        best match to worst).

        If the a strings embeddings has no similarity scores larger than tfidf_similarity_threshold, then it's index
        will not be present in the any rows of the csr matrix returned.

        Args:
            embedding_vectors (csr_matrix): Sparse matrix of TF-IDF vectors for each string to match.
        Returns:
            csr_matrix: Of the form (row, column) value. Where the row is the index of the string, the column is
                        the index of the matched match, and the value is the cosine similarity between the two
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
        str_similarity_matrix = get_topn_nearest_neighbours(
            embedding_vectors,
            self.document_vectors.transpose(),
            ntop=self.config.n_top_candidates,
            use_threads=self.config.use_threads,
            n_jobs=self.config.n_threads,
            lower_bound=self.config.tfidf_similarity_threshold,
            return_best_ntop=True,
        )[0].tocoo()
        return str_similarity_matrix

    def _get_tfidf_matches(self, str_similarity_matrix: csr_matrix) -> List[Optional[DocumentMatch]]:
        """Get DocumentMatch objects for each string from the str_similarity_matrix - of cosine similarity scores.

        Args:
            str_similarity_matrix (csr_matrix): Of the form (row, column) value. Where the row is the index of the
                string, the column is the index of the matched match, and the value is the cosine similarity between the
                two vectors.

        Returns:
            List[DocumentMatch]: Fuzzy matches (match and confidence).
        """
        idxs_present_in_csr = set(str_similarity_matrix.row.tolist())
        matches = []
        # convert to csr so we can index the matrix
        str_similarity_matrix = str_similarity_matrix.tocsr()
        for str_idx, doc_idx in enumerate(str_similarity_matrix.argmax(axis=1).tolist()):
            doc_idx = doc_idx[0]
            if str_idx in idxs_present_in_csr:
                # Result of argmax gives the idx=0 document, even if the string idx is not present in the
                # str_similarity_matrix matrix, so we need to filter out those cases.
                match = DocumentMatch(match=self.documents[doc_idx], confidence=str_similarity_matrix[str_idx, doc_idx])
            else:
                match = None
            matches.append(match)
        return matches

    def _get_lcs_best_match(self, string: str, document_matches: List[str]) -> Optional[DocumentMatch]:
        """Selects the best match from a set of candidates using the highest match ratio of longest common substring to
            string length.

        Args:
            string (str): String to find best LCS match for.
            document_matches (List[str]): The document match candidates identified from the TF-IDF top nearest neighbours to
                                        the string.
        Returns:
            Optional[DocumentMatch]: Fuzzy match (match and confidence).
        """
        best_match_idx, confidence = lcs.get_lcs_best_match_idx(
            preprocessed_str=self.apply_string_preprocessor(string),
            preprocessed_document_matches=self.apply_string_preprocessor(document_matches),
            min_characters=self.config.lcs_min_characters,
            min_length_ratio=self.config.lcs_min_length_ratio,
        )

        # best_match_idx of -1 indicates no valid matches found.
        if confidence >= self.config.lcs_similarity_threshold and best_match_idx != -1:
            return DocumentMatch(match=document_matches[best_match_idx], confidence=confidence)

    def _get_lcs_matches(self, str_similarity_matrix: csr_matrix, strings: List[str]) -> List[Optional[DocumentMatch]]:
        """Prunes the match candidates using LCS to identify the best match for each string.
        Args:
            str_similarity_matrix (csr_matrix): Coordinate format sparse matrix, match candidates ~ (data, (row, col)).
                                    The row and col are the row and column position of the value in the matrix, data
                                    represents the numerical value at that point. The match candidates is ordered from
                                    best match to worst match for each successive string.
            strings (List[str]): Strings to match.

        Returns:
            List[Optional[DocumentMatch]]: Fuzzy matches (match and confidence).
        """
        lcs_best_matches = {}
        # Assumes str_similarity_matrix.row is ordered - it's a pre-requisite of itertools.groupby.
        for string_idx, document_idxs in groupby(
            zip(str_similarity_matrix.row, str_similarity_matrix.col), key=lambda x: x[0]
        ):
            string = strings[string_idx]
            matches = [self.documents[document_idx] for _, document_idx in document_idxs]
            if match := self._get_lcs_best_match(string=string, document_matches=matches):
                lcs_best_matches[string] = match

        # str_similarity_matrix might not contain all strings, so we need to return a list of DocumentMatch in the
        # same order
        return [lcs_best_matches.get(string) for string in strings]
