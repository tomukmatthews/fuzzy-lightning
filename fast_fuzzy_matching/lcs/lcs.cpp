#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tuple>
#include <vector>

using namespace std;
namespace py = pybind11;

int longest_common_substring_length(const u32string &str_1, const u32string &str_2)
{
    // Takes two strings, returns the length of the longest common substring. Uses u32string (wide string) data type to
    // support non-ascii characters.

    // Either string is empty
    if (str_1.empty() || str_2.empty())
    {
        return 0;
    }

    // Initialise 2D array of zeros. Each element will accumulates ones for the number of consecutive identical
    // characters that come before it across both strings.
    // num_rows = len_str_1 + 1
    // num_cols = len_str_2 + 1
    // array_init_values = 0
    vector<vector<int>> common_substr_lens(str_1.length() + 1, vector<int>(str_2.length() + 1, 0));

    int longest_common_substr_len = 0;
    // Start from 1 so we have space look back to compare if the previous two characters matched.
    for (size_t i = 1; i <= str_1.length(); i++)
    {
        for (size_t j = 1; j <= str_2.length(); j++)
        {
            if (str_1[i - 1] == str_2[j - 1])
            {
                common_substr_lens[i][j] = common_substr_lens[i - 1][j - 1] + 1;
                longest_common_substr_len = max(common_substr_lens[i][j], longest_common_substr_len);
            }
        }
    }
    return longest_common_substr_len;
}

// int longest_common_substring_length(const u32string &str_1, const u32string &str_2)
// // In this implementation, we use a hash map to store the common substring lengths. This allows us to avoid the overhead of allocating and initializing a 2D array, which could potentially make the function faster and more efficient. Additionally, we use the u32string data type to support non-ASCII characters.
// {
//     // Either string is empty
//     if (str_1.empty() || str_2.empty())
//     {
//         return 0;
//     }

//     // Initialise hash map to store the length of the longest common substring ending at each index.
//     // map_init_values = 0
//     unordered_map<int, unordered_map<int, int>> common_substr_lens;

//     int longest_common_substr_len = 0;
//     // Start from 1 so we have space look back to compare if the previous two characters matched.
//     for (size_t i = 1; i <= str_1.length(); i++)
//     {
//         for (size_t j = 1; j <= str_2.length(); j++)
//         {
//             if (str_1[i - 1] == str_2[j - 1])
//             {
//                 // If characters match, add 1 to the length of the longest common substring ending at the previous index.
//                 common_substr_lens[i][j] = common_substr_lens[i - 1][j - 1] + 1;
//                 longest_common_substr_len = max(common_substr_lens[i][j], longest_common_substr_len);
//             }
//         }
//     }
//     return longest_common_substr_len;
// }

// int longest_common_substring_length(const u32string &str_1, const u32string &str_2)
// // In this implementation, we use a 1D array to store the common substring lengths. This allows us to reduce the memory overhead of the function, and makes it easier to access the common substring lengths in a way that takes advantage of data locality. Additionally, we use the u32string data type to support non-ASCII characters.
// {
//     // Either string is empty
//     if (str_1.empty() || str_2.empty())
//     {
//         return 0;
//     }

//     // Initialise 1D array to store the length of the longest common substring ending at each index.
//     // array_size = len_str_1 * len_str_2
//     // array_init_values = 0
//     vector<int> common_substr_lens(str_1.length() * str_2.length(), 0);

//     int longest_common_substr_len = 0;
//     // Start from 1 so we have space look back to compare if the previous two characters matched.
//     for (size_t i = 1; i <= str_1.length(); i++)
//     {
//         for (size_t j = 1; j <= str_2.length(); j++)
//         {
//             if (str_1[i - 1] == str_2[j - 1])
//             {
//                 // If characters match, add 1 to the length of the longest common substring ending at the previous index.
//                 // Use 1D array index to store the common substring length for the current index.
//                 common_substr_lens[(i - 1) * str_2.length() + (j - 1)] = common_substr_lens[(i - 1) * str_2.length() + (j - 2)] + 1;
//                 longest_common_substr_len = max(common_substr_lens[(i - 1) * str_2.length() + (j - 1)], longest_common_substr_len);
//             }
//         }
//     }
//     return longest_common_substr_len;
// }

// int longest_common_substring_length(const u32string &str_1, const u32string &str_2)
// // In this implementation, we store the value of str_2.length() in a variable and use that variable in the for loops instead of computing the length of str_2 each time. This allows us to avoid redundant computations, which could potentially make the function faster. Additionally, we use the u32string data type to support non-ASCII characters.
// {
//     // Either string is empty
//     if (str_1.empty() || str_2.empty())
//     {
//         return 0;
//     }

//     // Initialise 1D array to store the length of the longest common substring ending at each index.
//     // array_size = len_str_1 * len_str_2
//     // array_init_values = 0
//     vector<int> common_substr_lens(str_1.length() * str_2.length(), 0);

//     int longest_common_substr_len = 0;
//     // Store str_2.length() in a variable to avoid computing it multiple times.
//     int str_2_length = str_2.length();
//     // Start from 1 so we have space look back to compare if the previous two characters matched.
//     for (size_t i = 1; i <= str_1.length(); i++)
//     {
//         for (size_t j = 1; j <= str_2_length; j++)
//         {
//             if (str_1[i - 1] == str_2[j - 1])
//             {
//                 // If characters match, add 1 to the length of the longest common substring ending at the previous index.
//                 // Use 1D array index to store the common substring length for the current index.
//                 common_substr_lens[(i - 1) * str_2_length + (j - 1)] = common_substr_lens[(i - 1) * str_2_length + (j - 2)] + 1;
//                 longest_common_substr_len = max(common_substr_lens[(i - 1) * str_2_length + (j - 1)], longest_common_substr_len);
//             }
//         }
//     }
//     return longest_common_substr_len;
// }

// int longest_common_substring_length(const u32string &str_1, const u32string &str_2)
// // In this implementation, we use the dynamic programming algorithm to find the longest common substring. We also use a 1D array to store the common substring lengths, which allows us to take advantage of data locality and reduce the memory overhead of the function. Additionally, we use the u32string data type to support non-ASCII characters. This implementation may be faster and more efficient than the previous ones because it uses the dynamic programming algorithm and a 1D array to store the common substring lengths.
// {
//     // Either string is empty
//     if (str_1.empty() || str_2.empty())
//     {
//         return 0;
//     }

//     // Initialise 1D array to store the length of the longest common substring ending at each index.
//     // array_size = len_str_1 * len_str_2
//     // array_init_values = 0
//     vector<int> common_substr_lens(str_1.length() * str_2.length(), 0);

//     int longest_common_substr_len = 0;
//     // Store str_2.length() in a variable to avoid computing it multiple times.
//     int str_2_length = str_2.length();
//     // Start from 1 so we have space look back to compare if the previous two characters matched.
//     for (size_t i = 1; i <= str_1.length(); i++)
//     {
//         for (size_t j = 1; j <= str_2_length; j++)
//         {
//             if (str_1[i - 1] == str_2[j - 1])
//             {
//                 // If characters match, add 1 to the length of the longest common substring ending at the previous index.
//                 // Use 1D array index to store the common substring length for the current index.
//                 common_substr_lens[(i - 1) * str_2_length + j] = common_substr_lens[(i - 1) * str_2_length + (j - 1)] + 1;
//                 longest_common_substr_len = max(common_substr_lens[(i - 1) * str_2_length + j], longest_common_substr_len);
//             }
//         }
//     }
//     return longest_common_substr_len;
// }

// int longest_common_substring_length(const u32string &str_1, const u32string &str_2)
// // In this implementation, we use the dynamic programming algorithm to compute the longest common substring. We also use a rolling hash map to store the common substring lengths, which allows us to avoid the overhead of allocating and initializing a 2D or 1D array, and
// {
//     // Either string is empty
//     if (str_1.empty() || str_2.empty())
//     {
//         return 0;
//     }

//     // Initialise rolling hash map to store the length of the longest common substring ending at each index.
//     // map_init_values = 0
//     unordered_map<int, int> common_substr_lens;

//     // Initialise rolling hash value for each string.
//     int str_1_hash = 0;
//     int str_2_hash = 0;

//     int longest_common_substr_len = 0;
//     // Start from 1 so we have space look back to compare if the previous two characters matched.
//     for (size_t i = 1; i <= str_1.length(); i++)
//     {
//         for (size_t j = 1; j <= str_2.length(); j++)
//         {
//             // Update rolling hash values for each string.
//             str_1_hash = (str_1_hash + str_1[i - 1] * i) % MOD;
//             str_2_hash = (str_2_hash + str_2[j - 1] * j) % MOD;

//             // If characters match, add 1 to the length of the longest common substring ending at the previous index.
//             if (str_1_hash == str_2_hash)
//             {
//                 common_substr_lens[i * j] = common_substr_lens[(i - 1) * (j - 1)] + 1;
//                 longest_common_substr_len = max(common_substr_lens[i * j], longest_common_substr_len);
//             }
//         }
//     }
//     return longest_common_substr_len;
// }

PYBIND11_MODULE(lcs, m)
{
    m.def("longest_common_substring_length", &longest_common_substring_length, R"pbdoc(
                        A function to find the length of the longest substring between two strings.
            )pbdoc");
}
