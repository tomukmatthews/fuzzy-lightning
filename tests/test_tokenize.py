"""Test tokenize."""
import pytest

from fuzzy_lightning.tokenize import get_ngrams, tokenize_string_chars


@pytest.mark.parametrize(
    "string, n, pad, expected_output",
    [
        ('ABCDEFG', 1, False, ['A', 'B', 'C', 'D', 'E', 'F', 'G']),
        ('ABCDEFG', 2, False, ['AB', 'BC', 'CD', 'DE', 'EF', 'FG']),
        ('ABCDEFG', 3, False, ['ABC', 'BCD', 'CDE', 'DEF', 'EFG']),
        ('ABCDEFG', 7, False, ['ABCDEFG']),
    ],
)
def test_get_ngrams(string, n, pad, expected_output):
    """String should be split into a list of ngrams of length n."""
    output = get_ngrams(string=string, n=n, pad=pad)
    assert output == expected_output


@pytest.mark.parametrize(
    "string, n, pad, expected_output",
    [
        ('ABCDEFG', 2, True, ['-A', 'AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'G-']),
        ('ABCDEFG', 3, True, ['-AB', 'ABC', 'BCD', 'CDE', 'DEF', 'EFG', 'FG-']),
        ('ABCDEFG', 7, True, ['-ABCDEF', 'ABCDEFG', 'BCDEFG-']),
    ],
)
def test_get_ngrams_with_padding(string, n, pad, expected_output):
    """String should be split into a list of ngrams of length n. The beginning and end ngrams should have a '-' for
    padding.
    """
    output = get_ngrams(string=string, n=n, pad=pad)
    assert output == expected_output


def test_tokenizer():
    """String should be split into a list of ngrams, an ngram_range of (2,5) means the string should be chopped up
    into all (including duplicates) possible ngrams of that each length, i.e. sets of two characters, sets of three
    characters and so on up till 5. The beginning and end ngrams should have a '-' for padding.
    """
    input1 = 'TESTINGTESTDOG'
    output1 = tokenize_string_chars(string=input1, ngram_range=(2, 5))
    expected_output1 = [
        '-T',
        'TE',
        'ES',
        'ST',
        'TI',
        'IN',
        'NG',
        'GT',
        'TE',
        'ES',
        'ST',
        'TD',
        'DO',
        'OG',
        'G-',
        '-TE',
        'TES',
        'EST',
        'STI',
        'TIN',
        'ING',
        'NGT',
        'GTE',
        'TES',
        'EST',
        'STD',
        'TDO',
        'DOG',
        'OG-',
        '-TES',
        'TEST',
        'ESTI',
        'STIN',
        'TING',
        'INGT',
        'NGTE',
        'GTES',
        'TEST',
        'ESTD',
        'STDO',
        'TDOG',
        'DOG-',
        '-TEST',
        'TESTI',
        'ESTIN',
        'STING',
        'TINGT',
        'INGTE',
        'NGTES',
        'GTEST',
        'TESTD',
        'ESTDO',
        'STDOG',
        'TDOG-',
    ]

    input2 = 'TEST1234'

    output2 = tokenize_string_chars(string=input2, ngram_range=(4, 5))
    expected_output2 = [
        '-TES',
        'TEST',
        'EST1',
        'ST12',
        'T123',
        '1234',
        '234-',
        '-TEST',
        'TEST1',
        'EST12',
        'ST123',
        'T1234',
        '1234-',
    ]

    input3 = 'test 1234'
    output3 = tokenize_string_chars(string=input3, ngram_range=(4, 5))
    expected_output3 = [
        '-tes',
        'test',
        'est ',
        'st 1',
        't 12',
        ' 123',
        '1234',
        '234-',
        '-test',
        'test ',
        'est 1',
        'st 12',
        't 123',
        ' 1234',
        '1234-',
    ]

    input4 = ''
    output4 = tokenize_string_chars(string=input4, ngram_range=(2, 5))
    expected_output4 = []

    assert output1 == expected_output1
    assert output2 == expected_output2
    assert output3 == expected_output3
    assert output4 == expected_output4
