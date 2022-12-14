#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "array.h"

#include <unordered_map>
#include <vector>

using namespace std;

// Takes two strings, returns the minimum number of edits to transform str_1
// onto str_2 or vice versa. An edit includes any of: insertion, deletion,
// substitution or (adjacent character) transposition.

/**
 * Calculate the Damerau-Levenshtein distance between two Unicode strings.
 * This is based on the algorithm described here:
 * https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
 *
 * Takes two strings, returns the minimum number of edits to transform str_1
 * onto str_2 or vice versa. An edit includes any of: insertion, deletion,
 * substitution or (adjacent character) transposition.
 *
 * @param str_1 The first Unicode string to compare.
 * @param str_2 The second Unicode string to compare.
 * @return The Damerau-Levenshtein distance between str_1 and str_2.
 */
int damerau_levenshtein(const u32string &str_1, const u32string &str_2)
{
    const int str_1_length = str_1.length();
    const int str_2_length = str_2.length();

    IntArray2D edit_matrix(str_1_length + 2, str_2_length + 2);

    // Create map for tracking the latest idx of str_1 that each char appeared in as we iterate.
    unordered_map<char32_t, int> char_last_idx_tracker;

    const int maxdist = str_1_length + str_2_length;
    edit_matrix.set(0, 0, maxdist);
    for (size_t i = 0; i <= str_1_length; i++)
    {
        edit_matrix.set(i + 1, 1, i);
        edit_matrix.set(i + 1, 0, maxdist);
    }

    for (size_t j = 0; j <= str_2_length; j++)
    {
        edit_matrix.set(1, j + 1, j);
        edit_matrix.set(0, j + 1, maxdist);
    }

    for (size_t i = 1; i <= str_1_length; i++)
    {
        int db = 0;
        for (size_t j = 1; j <= str_2_length; j++)
        {
            // char_last_idx_tracker value defaults to 0 if key doesn't exist
            int k = char_last_idx_tracker[str_2[j - 1]];
            int l = db;

            int cost = 0;
            if (str_1[i - 1] == str_2[j - 1])
            {
                db = j;
            }
            else
            {
                cost = 1;
            }

            int substitution = edit_matrix.get(i, j) + cost;
            int insertion = edit_matrix.get(i + 1, j) + 1;
            int deletion = edit_matrix.get(i, j + 1) + 1;
            int transposition = edit_matrix.get(k, l) + (i - k - 1) + 1 + (j - l - 1);

            edit_matrix.set(i + 1, j + 1, min({substitution, insertion, deletion, transposition}));
        }
        char_last_idx_tracker[str_1[i - 1]] = i;
    }
    return edit_matrix.get(str_1_length + 1, str_2_length + 1);
}

PYBIND11_MODULE(edit_distance, m)
{
    m.def("damerau_levenshtein", &damerau_levenshtein, R"pbdoc(
                Calculates the edit distance between two strings using insertions, deletions, substitutions or
                transpositions.
            )pbdoc");
}