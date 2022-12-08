import pytest

import lcs


@pytest.mark.parametrize(
    argnames='str1, str2, expected_lcs_len',
    argvalues=[
        ('', '', 0),
        ('foo', '', 0),
        ('', 'foo', 0),
        ('dog', 'cat', 0),
        ('stein', 'steinbeer', 5),
        ('steinbeer', 'stein', 5),
        ('beersteinbeer', 'stein', 5),
        ('1234', '3456', 2),
        ('thejoker', 'thejoker', 8),
        ('aabc', 'aabc', 4),
        ('bbacc', 'bbaabbcc', 3),
    ],
)
def test_longest_common_substring_length_general(str1, str2, expected_lcs_len):
    lcs_len = lcs.longest_common_substring_length(str1, str2)
    assert lcs_len == expected_lcs_len


def test_longest_common_substring_length_case_sensitivity():
    lcs_len = lcs.longest_common_substring_length('zzz', 'ZZZ')
    expected_lcs_len = 0
    assert lcs_len == expected_lcs_len


@pytest.mark.parametrize(
    argnames='str1, str2, expected_lcs_len',
    argvalues=[
        ('£kłüv£v', 'kluv£v', 3),
        ('££$$$$££%', '%£$£$$$$£', 6),
        ('£11 transfer', '£11', 3),
        ('laurén', 'lauren', 4),
        ('?$100 & a mēal', '?$100 & a mēga meal', 12),
    ],
)
def test_longest_common_substring_length_non_ascii(str1, str2, expected_lcs_len):
    lcs_len = lcs.longest_common_substring_length(str1, str2)
    assert lcs_len == expected_lcs_len


@pytest.mark.skip('Unicode normalisation has not been implemented yet.')
def test_unicode_normalisation():
    """Both of these print as ñ so should be considered equal."""
    str1 = u'\u006e\u0303'
    str2 = u'\u00f1'
    lcs_len = lcs.longest_common_substring_length(str1, str2)

    expected_lcs_len = 1
    assert lcs_len == expected_lcs_len


@pytest.mark.parametrize(
    argnames='string, list_of_strings, expected_lcs_lens',
    argvalues=[
        ('aabc', ['aabc', 'ddd', 'ababac', 'aabcaabc', ''], [4, 0, 2, 4, 0]),
        ('bbacc', [], []),
        ('', ['abc'], [0]),
        ('tiger', ['rhinot', 'rhinotigerrhino', 'ti ger'], [1, 5, 3]),
        ('123', ['234', '1234', '232', '9', 'abc'], [2, 3, 2, 0, 0]),
        ('BCD', ['abcdef', 'ABCDEF'], [0, 3]),
    ],
)
def test_longest_common_substring_lengths_general(string, list_of_strings, expected_lcs_lens):
    lcs_lens = lcs.longest_common_substring_lengths(string, list_of_strings)
    assert lcs_lens == expected_lcs_lens
