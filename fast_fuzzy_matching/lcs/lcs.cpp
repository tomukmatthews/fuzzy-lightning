#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

using namespace std;
namespace py = pybind11;

/**
 * Finds the length of the longest common substring in the given strings.
 * This implementation uses the dynamic programming algorithm to find the longest common substring,
 * and a 1D array to store the common substring lengths. This allows us to take advantage of data locality.
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
    vector<int> common_substr_lens((str1_length + 1) * (str2_length + 1), 0);

    int longest_common_substr_len = 0;

    // Start from 1 so we have space look back to compare if the previous two characters matched.
    for (size_t i = 1; i <= str1_length; i++)
    {
        for (size_t j = 1; j <= str2_length; j++)
        {
            if (str1[i - 1] == str2[j - 1])
            {
                // If characters match, add 1 to the length of the longest common substring ending at the previous index.
                int common_substr_lens_idx = i * (str2_length + 1) + j;
                int prev_common_substr_lens_idx = common_substr_lens_idx - str2_length - 2;
                common_substr_lens[common_substr_lens_idx] = common_substr_lens[prev_common_substr_lens_idx] + 1;
                longest_common_substr_len = std::max(common_substr_lens[common_substr_lens_idx], longest_common_substr_len);
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

PYBIND11_MODULE(lcs, m)
{
    m.def("longest_common_substring_length", &longest_common_substring_length, R"pbdoc(
                        A function to find the length of the longest substring between two strings.
            )pbdoc");
    m.def("longest_common_substring_lengths", &longest_common_substring_lengths, R"pbdoc(
                    A function to find the length of the longest substring between a string and a list of strings.
        )pbdoc");
}
