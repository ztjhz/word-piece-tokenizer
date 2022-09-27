# A Lightweight Word Piece Tokenizer

[![PyPI version shields.io](https://img.shields.io/pypi/v/word-piece-tokenizer.svg)](https://pypi.org/project/word-piece-tokenizer/)

This library is an implementation of a modified version of [Huggingface's Bert Tokenizer](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer) in pure python.

## Table of Contents

1. [Usage](#usage)
   - [Installing](#installing)
   - [Example](#example)
   - [Running Tests](#running-tests)
1. [Making it Lightweight](#making-it-lightweight)
   - [Optional Features](#optional-features)
   - [Unused Features](#unused-features)
1. [Matching Algorithm](#matching-algorithm)
   - [The Trie](#the-trie)

## Usage

### Installing

Install and update using [pip](https://pip.pypa.io/en/stable/getting-started/)

```shell
pip install word-piece-tokenizer
```

### Example

```python
from word_piece_tokenizer import WordPieceTokenizer
tokenizer = WordPieceTokenizer()

ids = tokenizer.tokenize('reading a storybook!')
# [101, 3752, 1037, 2466, 8654, 999, 102]

tokens = tokenizer.convert_ids_to_tokens(ids)
# ['[CLS]', 'reading', 'a', 'story', '##book', '!', '[SEP]']

tokenizer.convert_tokens_to_string(tokens)
# '[CLS] reading a storybook ! [SEP]'
```

### Running Tests

Test the tokenizer against hugging's face implementation:

```bash
pip install transformers
python tests/tokenizer_test.py
```

<br/>

## Making It Lightweight

To make the tokenizer more lightweight and versatile for usage such as embedded systems and browsers, the tokenizer has been stripped of optional and unused features.

### Optional Features

The following features has been enabled by default instead of being configurable:

| Category      | Feature                                                                                                                                                                                 |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tokenizer     | - The tokenizer utilises the pre-trained [bert-based-uncased](https://huggingface.co/bert-base-uncased) vocab list.<br>- Basic tokenization is performed before word piece tokenization |
| Text Cleaning | - Chinese characters are padded with whitespace<br>- Characters are converted to lowercase<br>- Input string is stripped of accent                                                      |

### Unused Features

The following features has been removed from the tokenizer:

- `pad_token`, `mask_token`, and special tokens
- Ability to add new tokens to the tokenizer
- Ability to never split certain strings (`never_split`)
- Unused functions such as `build_inputs_with_special_tokens`, `get_special_tokens_mask`, `get_vocab`, `save_vocabulary`, and more...

<br/>

## Matching Algorithm

The tokenizer's _longest substring token matching_ algorithm is implemented using a `trie` instead of _greedy longest-match-first_

### The Trie

The original `Trie` class has been modified to adapt to the modified _longest substring token matching_ algorithm.

Instead of a `split` function that seperates the input string into substrings, the new trie implements a `getLongestMatchToken` function that returns the _token value `(int)`_ of the longest substring match, and the _remaining unmatched substring `(str)`_
