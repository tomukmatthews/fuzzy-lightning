"""
Tokenize strings into character ngrams.
"""

from typing import List, Tuple


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


def tokenize_string_chars(string: str, ngram_range: Tuple[int, int], pad: bool = True) -> List[str]:
    """Partition strings up into a list of character ngrams. You can pass a function that transforms (e.g. removes
    spaces and uppercases) through the text_preprocessor argument.

    Args:
        string (str): String to tokenize.
        ngram_range (Tuple[int, int]): Minimum and maximum number of characters to split the string into.
        pad (bool): Whether to pad the beginning and end of the string with '-' before tokenisation. (Places
            greater importance on the beginning and end of the string).

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
        ngrams = get_ngrams(string=string, n=ngram_len, pad=pad)
        tokens.extend(ngrams)
    return tokens
