import copy
import os

import regex as re

GPT_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = r"\w+"
result = re.findall(PAT, "some text that i'll pre-tokenize")


def pre_tokenize(text, special_tokens, count_tuple_bytes):
    # 根据 special_tokens 分割文本，保证文本边界不会在一起
    pattern_1 = '|'.join(map(re.escape, special_tokens))
    docs = re.split(pattern_1, text)

    tokens = []
    for doc in docs:
        tokens.extend(re.findall(GPT_PAT, doc))

    # print(tokens)
    for token in tokens:
        key = tuple(bytes([b]) for b in token.encode("utf-8"))
        count_tuple_bytes[key] = count_tuple_bytes.get(key, 0) + 1


def merge(vocab_size, vocabulary, count_tuple_bytes, map_id2bytes, merges):
    count_of_two_bytes_to_be_merged: dict[tuple[bytes, bytes], int] = {}
    while len(vocabulary) < vocab_size and len(count_tuple_bytes) > 1:
        # print(count_tuple_bytes)
        for k, v in count_tuple_bytes.items():
            for i in range(0, len(k) - 1):
                key = (k[i], k[i + 1])
                count_of_two_bytes_to_be_merged[key] = count_of_two_bytes_to_be_merged.get(key, 0) + v

        if not len(count_of_two_bytes_to_be_merged):
            break

        max_key = max(
            count_of_two_bytes_to_be_merged,
            key=lambda k : (count_of_two_bytes_to_be_merged[k], k)
        )
        merges.append(max_key)
        # print(f"bytes to be merged: {max_key}")
        new_vocab = max_key[0] + max_key[1]
        vocabulary.append(new_vocab)
        map_id2bytes[len(vocabulary) - 1] = new_vocab

        # print(f"new_vocab: {new_vocab}")

        new_dict: dict[tuple[bytes, ...], int] = {}

        for key, value in count_tuple_bytes.items():
            new_key = []
            i = 0
            len_key = len(key)

            while i < len_key:
                if i + 1 < len_key and key[i] == max_key[0] and key[i + 1] == max_key[1]:
                    new_key.append(key[i] + key[i + 1])
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1

            new_key = tuple(new_key)

            new_dict[new_key] = new_dict.get(new_key, 0) + value

        count_tuple_bytes = new_dict.copy()
        count_of_two_bytes_to_be_merged.clear()


def init_vocabulary(special_tokens, vocabulary, map_id2bytes):
    for i in range(0, 256):
        vocabulary.append(bytes(i.to_bytes()))
        map_id2bytes[i] = vocabulary[-1]

    for tokens in special_tokens:
        vocabulary.append(tokens.encode("utf-8"))
        map_id2bytes[len(vocabulary) - 1] = vocabulary[-1]


def train_bpe(input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocabulary: list[bytes] = []
    map_id2bytes: dict[int, bytes] = {}
    count_tuple_bytes: dict[tuple[bytes, ...], int] = {}
    merges: list[tuple[bytes, bytes]] = []

    # 1. 初始化vocab
    init_vocabulary(special_tokens, vocabulary, map_id2bytes)

    # 2. split the whole text with special_tokens
    origin_text: str
    with open(input_path, 'r', encoding='utf-8') as file:
        origin_text = file.read()

    pre_tokenize(origin_text, special_tokens, count_tuple_bytes)

    # 2. merge
    merge(vocab_size, vocabulary, count_tuple_bytes, map_id2bytes, merges)

    return map_id2bytes, merges


if __name__ == "__main__":
    vocabulary = init_vocabulary(["<|endoftext|>"])

    pre_tokenize("low low low low low <|endoftext|>\
        lower lower widest widest widest <|endoftext|>\
        newest newest newest newest newest newest <|endoftext|>", ["<|endoftext|>"])
    # merge(1000)