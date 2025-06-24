from collections import Counter

def get_char_ngrams(text, ngram_range=(2, 4)):
    """
    Extract character n-grams from a string.

    Args:
        text (str): Input text
        ngram_range (tuple): (min_n, max_n) n-gram sizes

    Returns:
        List[str]: List of n-grams
    """
    text = text.lower().replace("\n", " ")
    min_n, max_n = ngram_range
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i + n])
    return ngrams


def build_balanced_vocab(filepaths, ngram_range=(2, 4), vocab_size=50000, max_lines_per_lang=10000):
    """
    Builds a balanced n-gram vocabulary from multiple languages.

    Args:
        filepaths (dict): {'en': 'data/en.txt', 'fr': 'data/fr.txt', ...}
        ngram_range (tuple): Range of n-gram sizes
        vocab_size (int): Max vocabulary size
        max_lines_per_lang (int): Max number of lines to read per language

    Returns:
        dict: ngram -> integer ID (1-based)
    """
    counter = Counter()
    file_objs = {lang: open(path, encoding='utf-8') for lang, path in filepaths.items()}

    for _ in range(max_lines_per_lang):
        for lang, f in file_objs.items():
            line = f.readline()
            if not line:
                continue
            ngrams = get_char_ngrams(line.strip(), ngram_range)
            counter.update(ngrams)

    for f in file_objs.values():
        f.close()

    most_common = counter.most_common(vocab_size)
    vocab = {ngram: idx + 1 for idx, (ngram, _) in enumerate(most_common)}  # 0 reserved for unknown
    return vocab


def encode_ngrams(text, vocab, ngram_range=(2, 4)):
    """
    Encodes a text string as a list of n-gram token IDs.

    Args:
        text (str): Input text
        vocab (dict): ngram -> ID
        ngram_range (tuple): Range of n-gram sizes

    Returns:
        List[int]: List of token IDs
    """
    ngrams = get_char_ngrams(text, ngram_range)
    return [vocab.get(ngram, 0) for ngram in ngrams]  # 0 for unknown
