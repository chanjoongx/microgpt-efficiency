"""
data.py — dataset loading and tokenisation shared across all backends.

We use the same names.txt that Karpathy's original trains on.
Vocabulary convention: chars 0..25 = a..z, BOS = 26 (also used as EOS,
matching the original gist's single-special-token design).
"""

import os
import random
import urllib.request

NAMES_URL = (
    "https://raw.githubusercontent.com/karpathy/makemore/"
    "refs/heads/master/names.txt"
)


def load_dataset(seed: int = 42):
    """
    Download (once) names.txt and return everything needed for training.

    Returns
    -------
    docs       : list[str]   — shuffled names
    encode     : callable    — str  → list[int]  (BOS + ids + BOS)
    decode     : callable    — list[int] → str   (filters out BOS)
    bos_token  : int
    vocab_size : int
    """
    if not os.path.exists("input.txt"):
        print("Downloading names dataset …")
        urllib.request.urlretrieve(NAMES_URL, "input.txt")

    with open("input.txt") as fh:
        docs = [line.strip() for line in fh if line.strip()]

    random.seed(seed)
    random.shuffle(docs)

    chars      = sorted(set("".join(docs)))          # unique characters
    stoi       = {c: i for i, c in enumerate(chars)} # 'a'→0, …, 'z'→25
    itos       = {i: c for c, i in stoi.items()}
    BOS        = len(chars)                           # 26 — both start and end
    vocab_size = len(chars) + 1                       # 27

    def encode(doc: str):
        return [BOS] + [stoi[c] for c in doc] + [BOS]

    def decode(ids):
        return "".join(itos[i] for i in ids if i != BOS)

    return docs, encode, decode, BOS, vocab_size
