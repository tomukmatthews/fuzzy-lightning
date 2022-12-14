"""
Edit distance tests.
"""

import pytest

import edit_distance as ed


@pytest.mark.parametrize(
    argnames='str1, str2, expected_edit_dist',
    argvalues=[
        ('', 'a', 1),
        ('a', '', 1),
        ('ab', 'abc', 1),
        ('ab', 'cab', 1),
        ('ab', 'acb', 1),
        ('dog', 'dwoofog', 4),
        ('a11a', 'aa', 2),
    ],
)
def test_damerau_levenshtein_insertion_and_deletion(str1, str2, expected_edit_dist):
    edit_dist = ed.damerau_levenshtein(str1, str2)
    assert edit_dist == expected_edit_dist


@pytest.mark.parametrize(
    argnames='str1, str2, expected_edit_dist',
    argvalues=[
        ('a', 'b', 1),
        ('b', 'a', 1),
        ('ax', 'az', 1),
        ('abc', 'def', 3),
        ('a12a', 'a34a', 2),
    ],
)
def test_damerau_levenshtein_substitution(str1, str2, expected_edit_dist):
    edit_dist = ed.damerau_levenshtein(str1, str2)
    assert edit_dist == expected_edit_dist


@pytest.mark.parametrize(
    argnames='str1, str2, expected_edit_dist',
    argvalues=[('ab', 'ba', 1), ('ba', 'ab', 1), ('abcde', 'acbed', 2), ('123456789', '214365879', 4)],
)
def test_damerau_levenshtein_transposition(str1, str2, expected_edit_dist):
    edit_dist = ed.damerau_levenshtein(str1, str2)
    assert edit_dist == expected_edit_dist


@pytest.mark.parametrize(
    argnames='str1, str2, expected_edit_dist',
    argvalues=[
        ('', '', 0),
        ('dog', 'cat', 3),
        ('steinbeer', 'stein', 4),
        ('beersteinbeer', 'stein', 8),
        ('rufus', 'rufis', 1),
        ('rolph steer', 'rolph steup', 2),
        ('xyzxyz', 'xyzxyz', 0),
    ],
)
def test_damerau_levenshtein_general(str1, str2, expected_edit_dist):
    edit_dist = ed.damerau_levenshtein(str1, str2)
    assert edit_dist == expected_edit_dist


@pytest.mark.parametrize(argnames='str1, str2', argvalues=[(None, 'abc'), ('dog', 1.1)])
def test_damerau_levenshtein_bad_dtypes_raises_exception(str1, str2):
    with pytest.raises(TypeError):
        ed.damerau_levenshtein(str1, str2)
