#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "array.h"

#include <vector>
#include <tuple>

using namespace std;
namespace py = pybind11;

/**
 * Finds the length of the longest common substring in the given strings.
 * This implementation uses the dynamic programming algorithm to find the longest common substring,
 * and a 1D array to store the common substring lengths.
 *
 * @param str1 The first string to search for common substrings.
 * @param str2 The second string to search for common substrings.
 *
 * @return The length of the longest common substring.
 */
int longest_common_substring_length(const u32string &str1, const u32string &str2)
{
    // Either string is empty
    if (str1.empty() || str2.empty())
    {
        return 0;
    }

    const int str1_length = str1.length();
    const int str2_length = str2.length();

    // Initialise 1D array to store the length of the longest common substring ending at each index.
    IntArray2D common_substr_lens(str1_length + 1, str2_length + 1);

    int longest_common_substr_len = 0;

    // Start from 1 so we have space look back to compare if the previous two characters matched.
    for (size_t i = 1; i <= str1_length; i++)
    {
        for (size_t j = 1; j <= str2_length; j++)
        {
            if (str1[i - 1] == str2[j - 1])
            {
                // If characters match, add 1 to the length of the longest common substring ending at the previous index.
                common_substr_lens.set(i, j, common_substr_lens.get(i - 1, j - 1) + 1);
                longest_common_substr_len = std::max(common_substr_lens.get(i, j), longest_common_substr_len);
            }
        }
    }
    return longest_common_substr_len;
}

/**
 * Returns an array of longest common substring lengths for each string in strs compared to the first string, str.
 *
 * @param str The first string to compare against all the strings in strs.
 * @param strs A vector of strings to compare against the first string, str.
 *
 * @return An array of longest common substring lengths for each string in strs compared to the first string, str.
 */
std::vector<int> longest_common_substring_lengths(const u32string &str, const vector<u32string> &strs)
{
    std::vector<int> common_substr_lens(strs.size());
    for (size_t i = 0; i < strs.size(); i++)
    {
        common_substr_lens[i] = longest_common_substring_length(str, strs[i]);
    }
    return common_substr_lens;
}

/**
 * Identifies match candidates which don't qualify for consideration in longest common substring.
 *
 * @param preprocessed_str: a Unicode string in UTF-32 encoding after being preprocessed.
 * @param preprocessed_entity_match: a Unicode string in UTF-32 encoding after being preprocessed, to be compared with
 * preprocessed_str.
 * @param min_characters: the minimum number of characters that the strings must have in common to be considered a valid
 * match.
 * @param min_length_ratio: the minimum ratio of the length of the preprocessed_entity_match to the length of
 * preprocessed_str that must be satisfied for the match to be considered valid.
 *
 * @return: true if the preprocessed_entity_match is a valid match for preprocessed_str, or false otherwise.
 */
bool is_valid_entity_match(const u32string &preprocessed_str,
                           const u32string &preprocessed_entity_match,
                           const int min_characters,
                           const float min_length_ratio)
{
    // Identifies match candidates which don't qualify for consideration in longest common substring.

    if (preprocessed_entity_match.length() > preprocessed_str.length())
    {
        size_t min_length = max((int)(preprocessed_entity_match.length() * min_length_ratio), min_characters);
        bool no_substring_match = preprocessed_str.substr(0, min_length) != preprocessed_entity_match.substr(0, min_length);

        if (preprocessed_str.length() < min_length || no_substring_match)
        {
            return false;
        }
    }
    return true;
}

/**
 * Selects the best match from a set of candidates using the highest match ratio of longest common substring to
 * string length.
 *
 * @param preprocessed_str: A Unicode string that has been preprocessed to remove unwanted characters.
 * @param preprocessed_entity_matches: A vector of Unicode strings that have been preprocessed to remove unwanted
 * characters.
 * @param min_characters: The minimum number of characters that a match must have to be considered valid.
 * @param min_length_ratio: The minimum ratio of length of the longest common substring to the length of the shortest
 * string that a match must have to be considered valid.
 *
 * @return: A tuple containing the index of the best match in preprocessed_entity_matches, and the ratio of the longest
 * common substring to the length of the shortest string.
 */
tuple<int, float> get_lcs_best_match_idx(const u32string &preprocessed_str,
                                         const vector<u32string> &preprocessed_entity_matches,
                                         const int min_characters,
                                         const float min_length_ratio)
{
    // Selects the best match from a set of candidates using the highest match ratio of longest common substring to
    // string length.

    // Use -1 index to denote no valid matches.
    int best_match_idx = -1;
    float best_match_ratio = 0.0;
    for (size_t i = 0; i < preprocessed_entity_matches.size(); i++)
    {
        if (is_valid_entity_match(preprocessed_str,
                                  preprocessed_entity_matches[i],
                                  min_characters,
                                  min_length_ratio))
        {
            int shortest_str_len = min(preprocessed_str.length(), preprocessed_entity_matches[i].length());
            float match_ratio = longest_common_substring_length(preprocessed_str, preprocessed_entity_matches[i]) / (float)shortest_str_len;

            if (match_ratio > best_match_ratio)
            {
                best_match_idx = i;
                best_match_ratio = match_ratio;
            }
        }
    }
    return make_tuple(best_match_idx, best_match_ratio);
}

PYBIND11_MODULE(lcs, m)
{
    m.def("longest_common_substring_length", &longest_common_substring_length, R"pbdoc(
                        A function to find the length of the longest substring between two strings.
            )pbdoc");
    m.def("longest_common_substring_lengths", &longest_common_substring_lengths, R"pbdoc(
                    A function to find the length of the longest substring between a string and a list of strings.
        )pbdoc");
    m.def("get_lcs_best_match_idx", &get_lcs_best_match_idx, R"pbdoc(
                        A function to find the best match between a str and a set of entity matches.
            )pbdoc",
          py::arg("preprocessed_str"), py::arg("preprocessed_entity_matches"), py::arg("min_characters"),
          py::arg("min_length_ratio"));
    m.def("is_valid_entity_match", &is_valid_entity_match, R"pbdoc(
                        A function to determine if a candidate entity match is valid (exposing this for testing only!).
            )pbdoc",
          py::arg("preprocessed_str"), py::arg("preprocessed_entity_match"), py::arg("min_characters"),
          py::arg("min_length_ratio"));
}
