import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted

import lcs
from fast_fuzzy_matching.fuzzy_match import EntityMatch, FuzzyMatch, FuzzyMatchConfig


@pytest.fixture(scope='module')
def example_documents():
    example_documents_ = [
        "apple",
        "banana",
        "orange",
        "strawberry",
        "pear",
        "grape",
        "pineapple",
        "mango",
        "watermelon",
        "coconut",
        "lemon",
        "lime",
        "blueberry",
        "blackberry",
        "raspberry",
        "cherry",
        "peach",
        "plum",
        "apricot",
        "cantaloupe",
        "fig",
        "honeydew",
        "kiwi",
        "papaya",
        "passion fruit",
        "pomegranate",
        "quince",
        "star fruit",
        "tangerine",
        "grapefruit",
        "mandarin",
        "persimmon",
        "fig",
        "guava",
        "nectarine",
        "currant",
        "cranberry",
        "date",
        "elderberry",
        "gooseberry",
        "jackfruit",
        "jujube",
        "loganberry",
        "longan",
        "mulberry",
        "olive",
        "pomelo",
        "soursop",
        "tamarind",
        "yuzu",
    ]
    return example_documents_


@pytest.fixture(scope='module')
def fuzzy_matcher(example_documents):
    return FuzzyMatch(example_documents)


def test_vectorizer_initialised(example_documents):
    fm = FuzzyMatch(documents=example_documents)
    check_is_fitted(fm.vectorizer)
    assert isinstance(fm.document_vectors, csr_matrix)


def test_vectorizer_text_processor_applied():
    """Text processor should modify the input text as expected to result in the final expected vocab after
    tokenizing.
    """

    def text_preprocessor(text: str) -> str:
        return text.replace('CAT', 'DOG')

    fm = FuzzyMatch(
        documents=['CAT'], config=FuzzyMatchConfig(N_GRAM_RANGE=(2, 3), STRING_PREPROCESSOR=text_preprocessor)
    )

    vocab = set(fm.vectorizer.vocabulary_.keys())
    expected_vocab = {'DOG', '-D', 'OG', '-DO', 'DO', 'G-', 'OG-'}
    assert vocab == expected_vocab


def test_get_document_matches(fuzzy_matcher):
    """Should return fuzzy matching responses for each of these strings, with the correct id (the index) and
    response.
    """

    strings = pd.Series(['strawberrry', 'kiwi', 'gooseberru'], index=['a', 'b', 'c'])
    output_matches = fuzzy_matcher.get_document_matches(strings)
    print(output_matches)
    expected_output = {
        'a': EntityMatch('strawberry', 0.900),
        'b': EntityMatch('kiwi', 1.0),
        'c': EntityMatch('gooseberry', 0.900),
    }

    assert expected_output['a'].entity == output_matches['a'].entity
    assert expected_output['a'].confidence == pytest.approx(expected=output_matches['a'].confidence, abs=1e-3)

    assert expected_output['b'] == output_matches['b']

    assert expected_output['c'].entity == output_matches['c'].entity
    assert expected_output['c'].confidence == pytest.approx(expected=output_matches['c'].confidence, abs=1e-3)


def test_get_document_matches_empty_strings(fuzzy_matcher):
    """Should return an empty dict as there are no matches."""

    strings = pd.Series(['', '', ''], index=['a', 'b', 'c'])
    output_matches = fuzzy_matcher.get_document_matches(strings=strings)

    expected_output = {}
    assert expected_output == output_matches


def test_get_document_matches_empty_series(fuzzy_matcher):
    """Get document matches should return an empty dict if no strings are passed."""
    strings = pd.Series()
    output_matches = fuzzy_matcher.get_document_matches(strings=strings)
    assert output_matches == {}


def test_document_vectors_maps_to_entity_in_range(fuzzy_matcher):
    """The column index from the document vectors generated should always be in the valid index range of the
    entities.
    """
    matrix = fuzzy_matcher.document_vectors
    entities = fuzzy_matcher.documents
    assert max(matrix.tocoo().row) <= len(entities)


def test_get_document_match(fuzzy_matcher):
    """Should return the fuzzy match response for the string."""
    match = fuzzy_matcher.get_document_match(string='pineapplee')
    expected_match = EntityMatch('pineapple', 1.0)

    assert expected_match.entity == match.entity
    assert expected_match.confidence == pytest.approx(expected=match.confidence, abs=1e-3)

    assert match == expected_match


def test_get_document_match_empty_string(fuzzy_matcher):
    """Should return None as no match was found."""

    match = fuzzy_matcher.get_document_match(string='')
    assert match is None


def test_get_lcs_matches(fuzzy_matcher):
    """Should return the best LCS match for each string."""
    txn_ids = ['a', 'b', 'c']
    strings = ['AA MEMBER', 'O2 ACAD', 'QD STORES SAMSUNG']
    idx_entity_map = {
        0: 'AA MEMBERSHIP',
        1: 'M BERGER',
        2: 'UBER',
        20: 'O2 ACADEMY',
        25: 'SAMSUNG',
        26: 'QD STORES',
        30: 'FOODSTORE',
        40: 'GREENEND STORES',
    }
    # Mock the usage of a scipy sparse matrix in coo format.
    row = np.array([0, 0, 0, 1, 2, 2, 2, 2])
    col = np.array([0, 1, 2, 20, 25, 26, 30, 40])
    data = np.array([0.2, 0.4, 0.5, 0.3, 0.55, 0.2, 0.1, 0.05])
    match_candidates = csr_matrix((data, (row, col))).tocoo()

    output = fuzzy_matcher._get_lcs_matches(
        match_candidates=match_candidates, ids=txn_ids, strings=strings, idx_entity_map=idx_entity_map
    )
    expected_output = {
        'a': EntityMatch('AA MEMBERSHIP', 1.0),
        'b': EntityMatch('O2 ACADEMY', 1.0),
        'c': EntityMatch('SAMSUNG', 1.0),
    }

    assert output == expected_output


def test_is_valid_entity_match_entity_too_long():
    """Invalid match candidates should be removed here because of the entity name being longer than the
    string.
    """
    string = 'NANDOS SOUTHGATE'
    matches = ['NANDOS', 'NANDY PP', 'ANDOSERRA', 'NANDOS BUT TOO LONG ENTITY']
    no_space_matches = [match.replace(' ', '') for match in matches]

    config = FuzzyMatchConfig()

    output = [
        lcs.is_valid_entity_match(
            no_space_str=string.replace(' ', ''),
            no_space_entity_match=no_space_match,
            min_characters=config.MIN_LCS_CHARACTERS,
            min_length_ratio=config.MIN_LCS_LENGTH_RATIO,
        )
        for no_space_match in no_space_matches
    ]

    expected_output = [True, True, True, False]

    assert output == expected_output


# def test_is_valid_entity_match_string_too_short():
#     """All match candidates should be removed - none are valid because the string is too short to be qualify
#     for LCS.
#     """
#     string = 'NAN'
#     matches = ['NANDOS', 'NANDY PP', 'ANDOSERRA']
#     no_space_matches = [match.replace(' ', '') for match in matches]

#     config = FuzzyMatchConfig()

#     output = [
#         lcs.is_valid_entity_match(
#             no_space_str=string.replace(' ', ''),
#             no_space_entity_match=no_space_match,
#             min_characters=config.MIN_LCS_CHARACTERS,
#             min_length_ratio=config.MIN_LCS_LENGTH_RATIO,
#         )
#         for no_space_match in no_space_matches
#     ]

#     expected_output = [False, False, False]

#     assert output == expected_output


# def test_is_valid_entity_match_string_long_enough():
#     """All match candidates should be kept - all are valid because the string is long enough to be qualify
#     for LCS.
#     """
#     string = 'NAND'
#     matches = ['NANDOS', 'NANDY PP']
#     no_space_matches = [match.replace(' ', '') for match in matches]

#     config = FuzzyMatchConfig(MIN_LCS_CHARACTERS=4)
#     output = [
#         lcs.is_valid_entity_match(
#             no_space_str=string.replace(' ', ''),
#             no_space_entity_match=no_space_match,
#             min_characters=config.MIN_LCS_CHARACTERS,
#             min_length_ratio=config.MIN_LCS_LENGTH_RATIO,
#         )
#         for no_space_match in no_space_matches
#     ]
#     assert all(output)


# def test_is_valid_entity_match_no_substring_match():
#     """In all these cases the string is shorter than the match candidates so we have to verify that in
#     each case the string at least the beginning of the strings match up to the minimum length.
#     """
#     matches = ['NANSOD', 'NANDY PP', 'NANDOS']
#     no_space_matches = [match.replace(' ', '') for match in matches]
#     config = FuzzyMatchConfig(MIN_LCS_CHARACTERS=4)

#     string1 = 'NANDO'

#     output1 = [
#         lcs.is_valid_entity_match(
#             no_space_str=string1.replace(' ', ''),
#             no_space_entity_match=no_space_match,
#             min_characters=config.MIN_LCS_CHARACTERS,
#             min_length_ratio=config.MIN_LCS_LENGTH_RATIO,
#         )
#         for no_space_match in no_space_matches
#     ]
#     expected_output1 = [False, True, True]
#     assert output1 == expected_output1

#     string2 = 'NAND'
#     output2 = [
#         lcs.is_valid_entity_match(
#             no_space_str=string2.replace(' ', ''),
#             no_space_entity_match=no_space_match,
#             min_characters=config.MIN_LCS_CHARACTERS,
#             min_length_ratio=config.MIN_LCS_LENGTH_RATIO,
#         )
#         for no_space_match in no_space_matches
#     ]
#     assert output2 == expected_output1
