import collections
import concurrent.futures
import copy
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from typing import BinaryIO
from collections import Counter

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

GPT_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = r"\w+"
result = re.findall(PAT, "some text that i'll pre-tokenize")

def multi_pre_tokenize(file: BinaryIO, num_processes, special_tokens):
    start_time = time.time()
    boundaries = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
    chunks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        file.seek(start)
        chunks.append(file.read(end - start).decode("utf-8", errors="ignore"))

    # print(f"split cost time: {time.time() - start_time}")

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_tokenize = [executor.submit(single_pre_tokenize, chunk, special_tokens) for chunk in chunks]

        merged_counter = collections.defaultdict(int)
        for future in concurrent.futures.as_completed(future_to_tokenize):
            try:
                res = future.result()
                for key, freq in res.items():
                    merged_counter[key] += freq
            except Exception as e:
                print(f"pre_tokenize failure: {e}")

        return merged_counter



def single_pre_tokenize(chunk, special_tokens):
    start_time = time.time()
    single_count_tuple_bytes: dict[tuple[bytes, ...], int] = {}
    # 1. 预编译正则表达式[3](@ref)
    pattern_1 = '|'.join(map(re.escape, special_tokens))
    compiled_pattern = re.compile(pattern_1)  # 预编译分割模式
    compiled_gpt_pattern = re.compile(GPT_PAT)  # 预编译GPT模式

    # 2. 分割文档[7](@ref)
    docs = compiled_pattern.split(chunk)
    # 3. 使用生成器表达式减少内存占用[3,4](@ref)
    tokens = (token for doc in docs for token in compiled_gpt_pattern.findall(doc))

    # 4. 统计token的频次

    counts = collections.Counter(tokens)
    token_counter = collections.defaultdict(int)
    for word, freq in counts.items():
        key = tuple(bytes([b]) for b in word.encode('utf-8'))
        token_counter[key] += freq


    # 6. 更新到外部字典
    single_count_tuple_bytes.update(token_counter)

    return single_count_tuple_bytes


def pre_tokenize(text, special_tokens):
    count_tuple_bytes: dict[tuple[bytes, ...], int] = {}
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

    return count_tuple_bytes


def merge(vocab_size, vocabulary, count_tuple_bytes, map_id2bytes, merges):
    # 1. 由bytes pair映射到tuple bytes
    indices = collections.defaultdict(lambda: collections.defaultdict(int))

    count_of_two_bytes_to_be_merged: dict[tuple[bytes, bytes], int] = {}
    # print(count_tuple_bytes)
    while len(vocabulary) < vocab_size and len(count_tuple_bytes) > 1:

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
        new_vocab = max_key[0] + max_key[1]
        vocabulary.append(new_vocab)
        map_id2bytes[len(vocabulary) - 1] = new_vocab


        new_dict = {}

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

        count_tuple_bytes = new_dict
        count_of_two_bytes_to_be_merged.clear()

def optimize_merge(vocab_size, vocabulary, count_tuple_bytes, map_id2bytes, merges):
    """
    param:
        vocab_size: the target vocab size
        vocab: tuple bytes -> frequency
        merges: bytes pair to be merged
    """
    # status: (a, b) -> frequency
    # indices: (a, b) -> tuple bytes
    status = collections.defaultdict(int)
    indices = collections.defaultdict(lambda : collections.defaultdict(int))

    # 1. init the vocab, status and indices
    vocab = []
    word_freqs = []
    for word, freq in count_tuple_bytes.items():
        vocab.append(word)
        word_freqs.append(freq)

    for i, word in enumerate(vocab):
        for j in range(len(word) - 1):
            pair = (word[j], word[j + 1])
            status[pair] += word_freqs[i]
            indices[pair][i] += 1

    # 2. make merge
    while len(vocabulary) < vocab_size:
        if not status: break

        best_pair = max(status, key=lambda k : (status[k], k))
        if status[best_pair] < 1: break

        merges.append(best_pair)

        new_token = best_pair[0] + best_pair[1]

        vocabulary.append(new_token)
        map_id2bytes[len(vocabulary) - 1] = new_token

        affected_words = indices[best_pair]

        for word_id, count_in_word in affected_words.items():
            if count_in_word < 1: continue

            word = vocab[word_id]
            freq = word_freqs[word_id]

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    if i > 0:
                        pre_pair = (word[i - 1], word[i])
                        status[pre_pair] -= freq
                        status[(word[i - 1], new_token)] += freq
                        indices[pre_pair][word_id] -= 1
                        indices[(word[i - 1], new_token)][word_id] += 1
                    if i + 2 < len(word):
                        post_pair = (word[i + 1], word[i + 2])
                        status[post_pair] -= freq
                        status[(new_token, word[i + 2])] += freq
                        indices[post_pair][word_id] -= 1
                        indices[(new_token, word[i + 2])][word_id] += 1

                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            vocab[word_id] = new_word

        del status[best_pair]
        del indices[best_pair]



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
    merges: list[tuple[bytes, bytes]] = []
    num_processes = 16

    # 1. 初始化vocab
    init_vocabulary(special_tokens, vocabulary, map_id2bytes)

    # 2. split the whole text with special_tokens
    start_time = time.time()
    with open(input_path, 'rb') as f:
        count_tuple_bytes = multi_pre_tokenize(f, num_processes, special_tokens)
    print(f"pre tokenize cost time: {time.time() - start_time}")

    # print(count_tuple_bytes)
    # 2. merge
    # merge(vocab_size, vocabulary, count_tuple_bytes, map_id2bytes, merges)
    optimize_merge(vocab_size, vocabulary, count_tuple_bytes, map_id2bytes, merges)


    # TODO: high performance merge method


    return map_id2bytes, merges


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = f"{PROJECT_ROOT}/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    start_time = time.time()

    # vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    # print(merges)

    # with open("tinystoriesV2-gpt4-train-vocab.pkl", 'wb') as f:
    #     pickle.dump(vocab, f)
    #
    # with open("tinystoriesV2-gpt4-train-merges.pkl", 'wb') as f:
    #     pickle.dump(merges, f)

    # print(f"cost time: {time.time() - start_time}")


    with open("tinystoriesV2-gpt4-train-vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
        print(len(vocab))
        # print(type(vocab))
        # print(vocab)