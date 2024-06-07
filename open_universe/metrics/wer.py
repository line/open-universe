# Copyright 2024 LY Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# Levenshtein distance with wild cards

simple implementation of WER computaton with wildcards

Author: Robin Scheibler (@fakufaku)
"""
import re
import string
from typing import List, Dict, Optional

from whisper.normalizers import EnglishTextNormalizer

PUNCTUATION = "[" + string.punctuation + "]"

SYMBOLS = {
    "$": "dollar",
}


def normalize_sentence(
    sentence: str, wildcard: Optional[str] = None, style="whisper"
) -> str:
    """
    Normalizes a sentence
    - splits string into words on spaces
    - detects possible wildcard and replaces it with None in the output list
    - removes all punctuation (it is replaced by empty string)

    Parameters
    ----------
    sentence : str
        The sentence to normalize
    wildcard : Optional[str], optional
        An optional wildcard. If None, no wildcard is used, by default None

    Returns
    -------
    List[str]
        A list of strings or None (where the wildcard was)
    """

    if style == "whisper":
        textcleaner = EnglishTextNormalizer()

        sentence = sentence.lower()

        if wildcard is not None:
            wildcard = wildcard.lower()
            parts = sentence.split(wildcard)
            parts = [textcleaner(p) for p in parts]
        else:
            parts = [textcleaner(sentence)]

        words = []
        for part in parts:
            words += part.split()
            words.append(None)
        words.pop()  # remove the last None

    else:
        if not isinstance(sentence, list):
            sentence = sentence.split()

        words = []
        wildcard = wildcard.lower() if wildcard else None
        for word in sentence:
            word = word.lower()

            if wildcard and word == wildcard:
                words.append(None)
            else:
                words.append(re.sub(PUNCTUATION, "", word))

    return words


def tokenize(sentence: List[str], vocabulary: Dict[str, int]) -> List[int]:
    """
    Tokenizes a sentence using a vocabulary
    """
    tokens = []
    for word in sentence:
        if word in vocabulary:
            tokens.append(vocabulary[word])
        else:
            raise ValueError("Word {word} not in vocabulary")
    return tokens


def word_edit_distance(
    sentence1: List[str], sentence2: List[str], wildcard: Optional[str] = None
) -> int:
    """
    Compute the word edit distance between two sentences.
    The sentences are first normalized by making everything lower case and
    removing punctuation.

    A wildcard can be included. In this case, the wildcard covers for any
    missing or extra word that could be present at the wildcard location.
    The wildcard can cover multiple words.

    Example:
    "hello <ignore/> robin" and "hello robin" would have a distance of 0
    "hello <ignore/>" and "hello robin" would have a distance of 0
    "hello <ignore/>" and "hello robin shark" would have a distance of 0
    "hello <ignore/> robin" and "hello robin shark" would have a distance of 1

    Parameters
    ----------
    sentence1 : str
        The first sentence
    sentence2 : str
        The second sentence
    wildcard : Optional[str], optional
        An optional wildcard. If None, no wildcard is used, by default None

    Returns
    -------
    edit_distance: int
        The number of edits (add, sub, del) necessary to transform sentence1 into
        sentence2
    len_sentence1: int
        The number of words in sentence1 (after removing the wildcard)
    len_sentence2: int
        The number of words in sentence2 (after removing the wildcard)
    """
    # remove punctuation and make everything lowercase
    s1 = normalize_sentence(sentence1, wildcard=wildcard)
    s2 = normalize_sentence(sentence2, wildcard=wildcard)

    num_words_s1 = len([w for w in s1 if w is not None])
    num_words_s2 = len([w for w in s2 if w is not None])

    if len(s1) == 0:
        # distance is the length without the wild card
        return num_words_s2, num_words_s1, num_words_s2
    elif len(s2) == 0:
        # distance is the length without the wild card
        return num_words_s1, num_words_s1, num_words_s2

    # create vocabulary
    vocabulary = set(s1 + s2)
    has_wildcard = None in vocabulary
    if has_wildcard:  # put wildcard at the end
        vocabulary.remove(None)

    vocabulary = {word: i for i, word in enumerate(vocabulary)}
    vocabulary[None] = len(vocabulary)
    wildcard = vocabulary[None]

    # tokenize
    t1 = tokenize(s1, vocabulary)
    t2 = tokenize(s2, vocabulary)

    # create the distance matrix
    dist = [[None for _ in range(len(t2) + 1)] for _ in range(len(t1) + 1)]

    dist[0][0] = 0

    for m in range(1, len(t1) + 1):
        if t1[m - 1] == wildcard:
            dist[m][0] = dist[m - 1][0]
        else:
            dist[m][0] = dist[m - 1][0] + 1

    for n in range(1, len(t2) + 1):
        if t2[n - 1] == wildcard:
            dist[0][n] = dist[0][n - 1]
        else:
            dist[0][n] = dist[0][n - 1] + 1

    for m in range(1, len(t1) + 1):
        for n in range(1, len(t2) + 1):
            if t1[m - 1] == wildcard or t2[n - 1] == wildcard:
                dist[m][n] = min(dist[m - 1][n], dist[m][n - 1])
            elif t1[m - 1] == t2[n - 1]:
                dist[m][n] = dist[m - 1][n - 1]
            else:
                dist[m][n] = 1 + min(dist[m - 1][n], dist[m][n - 1], dist[m - 1][n - 1])

    return dist[-1][-1], num_words_s1, num_words_s2


def wer(ref: List[str], hyp: List[str], wildcard=None) -> float:
    """
    Word error rate (WER) computation

    Parameters
    ----------
    ref : List[str]
        The reference sentences
    hyp : List[str]
        The hypothesis sentences
    wildcard : Optional[str], optional
        An optional wildcard. If None, no wildcard is used, by default None.
        The wildcard absorbs any extra or missing word at the wildcard location.
    """
    total_dist = 0
    total_num_words = 0
    for r, h in zip(ref, hyp):
        dist, n_ref, n_hyp = word_edit_distance(r, h, wildcard=wildcard)
        total_dist += dist
        total_num_words += n_ref
    return total_dist / total_num_words


if __name__ == "__main__":

    def test(s1, s2, wildcard="<ignore/>"):
        dist = word_edit_distance(s1, s2, wildcard=wildcard)
        print(f"Comparing:")
        print(f" - '{s1}'")
        print(f" - '{s2}'")
        print(f"Word edit distance: {dist[0]}")
        print("")

    s1 = "hello robin!"
    s2 = "hello robin"
    s3 = "hello <ignore/> robin"
    s4 = "hello <ignore/>"
    s5 = "hello <ignore/> shark"
    s5 = "hello <ignore/> robin"
    s6 = "hello robin shark"
    test(s1, s1)
    test(s1, s2)
    test(s1, s3)
    test(s1, s4)
    test(s1, s5)
    test(s5, s6)

    s1 = (
        "for a moment he felt like a deep discovered five shows <UNKNOWN/> "
        "beneath the shirt the system may break down soon "
        "so save your files frequently why can't we get this shirt"
    )
    s2 = (
        "For a moment, he felt like a thief discovered. Fats shows in loose "
        "rows beneath the shirt. The system may break down soon, so save "
        "your files frequently. Where can we get this system?"
    )
    test(s1, s2, wildcard="<UNKNOWN/>")
