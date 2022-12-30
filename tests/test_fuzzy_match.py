"""Test fuzzy match."""

import lcs
import numpy as np
import pytest
from fuzzy_lightning.fuzzy_match import DocumentMatch, FuzzyMatch, FuzzyMatchConfig
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted


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
        documents=['CAT'],
        config=FuzzyMatchConfig(n_gram_range=(2, 3), string_preprocessor=text_preprocessor),
    )

    vocab = set(fm.vectorizer.vocabulary_.keys())
    expected_vocab = {'DOG', '-D', 'OG', '-DO', 'DO', 'G-', 'OG-'}
    assert vocab == expected_vocab


def test_get_document_matches(fuzzy_matcher):
    """Should return fuzzy matching responses for each of these strings, with the correct id (the index) and
    response.
    """

    strings = ['strawberrry', 'kiwi', 'gooseberru']
    output_matches = fuzzy_matcher.get_document_matches(strings)
    expected_output = [
        DocumentMatch('strawberry', 0.900),
        DocumentMatch('kiwi', 1.0),
        DocumentMatch('gooseberry', 0.900),
    ]

    assert expected_output[0].match == output_matches[0].match
    assert expected_output[0].confidence == pytest.approx(expected=output_matches[0].confidence, abs=1e-3)

    assert expected_output[1] == output_matches[1]

    assert expected_output[2].match == output_matches[2].match
    assert expected_output[2].confidence == pytest.approx(expected=output_matches[2].confidence, abs=1e-3)


def test_get_document_matches_empty_strings(fuzzy_matcher):
    """Should return an empty dict as there are no matches."""

    strings = ['', '', '']
    assert fuzzy_matcher.get_document_matches(strings=strings) == [None, None, None]


def test_get_document_matches_empty_list(fuzzy_matcher):
    """Get document matches should return an empty dict if no strings are passed."""
    output_matches = fuzzy_matcher.get_document_matches(strings=[])
    assert output_matches == []


def test_document_vectors_maps_to_document_in_range(fuzzy_matcher):
    """The column index from the document vectors generated should always be in the valid index range of the
    entities.
    """
    matrix = fuzzy_matcher.document_vectors
    entities = fuzzy_matcher.documents
    assert max(matrix.tocoo().row) <= len(entities)


@pytest.mark.parametrize('prune_with_lcs', [True, False])
def test_get_document_match(fuzzy_matcher, prune_with_lcs):
    """Should return the fuzzy match response for the string."""
    match = fuzzy_matcher.get_document_match(string='pineapplee', prune_with_lcs=prune_with_lcs)
    expected_match = DocumentMatch('pineapple', 1.0)

    assert expected_match.match == match.match
    assert expected_match.confidence == pytest.approx(expected=match.confidence, abs=0.1)


def test_get_document_match_empty_string(fuzzy_matcher):
    """Should return None as no match was found."""

    match = fuzzy_matcher.get_document_match(string='')
    assert match is None


def test_get_lcs_matches():
    """Should return the best LCS match for each string."""
    fm = FuzzyMatch(
        documents=[
            'AA MEMBERSHIP',
            'M BERGER',
            'UBER',
            'O2 ACADEMY',
            'SAMSUNG',
            'QD STORES',
            'FOODSTORE',
            'GREENEND STORES',
        ]
    )

    strings = ['AA MEMBER', 'O2 ACAD', 'QD STORES SAMSUNG']

    # Mock the usage of a scipy sparse matrix in coo format.
    row = np.array([0, 0, 0, 1, 2, 2, 2, 2])
    col = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    data = np.array([0.2, 0.4, 0.5, 0.3, 0.55, 0.2, 0.1, 0.05])
    str_similarity_matrix = csr_matrix((data, (row, col))).tocoo()

    output = fm._get_lcs_matches(str_similarity_matrix=str_similarity_matrix, strings=strings)
    expected_output = [
        DocumentMatch('AA MEMBERSHIP', 1.0),
        DocumentMatch('O2 ACADEMY', 1.0),
        DocumentMatch('SAMSUNG', 1.0),
    ]

    assert output == expected_output


def test_is_valid_document_match_too_long():
    """Invalid match candidates should be removed here because of the match name being longer than the
    string.
    """
    string = 'NANDOS SOUTHGATE'
    matches = ['NANDOS', 'NANDY PP', 'ANDOSERRA', 'NANDOS BUT TOO LONG MATCH']
    no_space_matches = [match.replace(' ', '') for match in matches]

    config = FuzzyMatchConfig()

    output = [
        lcs.is_valid_document_match(
            preprocessed_str=string.replace(' ', ''),
            preprocessed_document_match=no_space_match,
            min_characters=config.lcs_min_characters,
            min_length_ratio=config.lcs_min_length_ratio,
        )
        for no_space_match in no_space_matches
    ]

    expected_output = [True, True, True, False]

    assert output == expected_output


def test_is_valid_document_match_string_too_short():
    """All match candidates should be removed - none are valid because the string is too short to be qualify
    for LCS.
    """
    string = 'NAN'
    matches = ['NANDOS', 'NANDY PP', 'ANDOSERRA']
    no_space_matches = [match.replace(' ', '') for match in matches]

    config = FuzzyMatchConfig()

    output = [
        lcs.is_valid_document_match(
            preprocessed_str=string.replace(' ', ''),
            preprocessed_document_match=no_space_match,
            min_characters=config.lcs_min_characters,
            min_length_ratio=config.lcs_min_length_ratio,
        )
        for no_space_match in no_space_matches
    ]

    expected_output = [False, False, False]

    assert output == expected_output


def test_is_valid_document_match_string_long_enough():
    """All match candidates should be kept - all are valid because the string is long enough to be qualify
    for LCS.
    """
    string = 'NAND'
    matches = ['NANDOS', 'NANDY PP']
    no_space_matches = [match.replace(' ', '') for match in matches]

    config = FuzzyMatchConfig(lcs_min_characters=4)
    output = [
        lcs.is_valid_document_match(
            preprocessed_str=string.replace(' ', ''),
            preprocessed_document_match=no_space_match,
            min_characters=config.lcs_min_characters,
            min_length_ratio=config.lcs_min_length_ratio,
        )
        for no_space_match in no_space_matches
    ]
    assert all(output)


def test_is_valid_document_match_no_substring_match():
    """In all these cases the string is shorter than the match candidates so we have to verify that in
    each case the string at least the beginning of the strings match up to the minimum length.
    """
    matches = ['NANSOD', 'NANDY PP', 'NANDOS']
    no_space_matches = [match.replace(' ', '') for match in matches]
    config = FuzzyMatchConfig(lcs_min_characters=4)

    string1 = 'NANDO'

    output1 = [
        lcs.is_valid_document_match(
            preprocessed_str=string1.replace(' ', ''),
            preprocessed_document_match=no_space_match,
            min_characters=config.lcs_min_characters,
            min_length_ratio=config.lcs_min_length_ratio,
        )
        for no_space_match in no_space_matches
    ]
    expected_output1 = [False, True, True]
    assert output1 == expected_output1

    string2 = 'NAND'
    output2 = [
        lcs.is_valid_document_match(
            preprocessed_str=string2.replace(' ', ''),
            preprocessed_document_match=no_space_match,
            min_characters=config.lcs_min_characters,
            min_length_ratio=config.lcs_min_length_ratio,
        )
        for no_space_match in no_space_matches
    ]
    assert output2 == expected_output1
