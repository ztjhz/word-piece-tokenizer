from transformers import BertTokenizer
import unittest

import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.word_piece_tokenizer.WordPieceTokenizer import WordPieceTokenizer


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self._bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')
        self._my_tokenizer = WordPieceTokenizer()

    def tokenize_with_both_tokenizer(self, s: str):
        batch = self._bert_tokenizer([s])
        lib_res = batch.input_ids[0]
        my_res = self._my_tokenizer.tokenize(s)
        print()
        print(s)
        print(lib_res)
        print(my_res)
        return lib_res, my_res

    def test_normal_sentence(self):
        s = "This is the Hugging Face!"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)

    def test_long_word(self):
        s = "Pneumonoultramicroscopicsilicovolcanoconiosis"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)

    def test_long_word_in_sentence(self):
        s = "wow! Pneumonoultramicroscopicsilicovolcanoconiosis is such a long word!"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)

    def test_long_word_in_sentence_2(self):
        s = "internalization is the best thing in the world!"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)

    def test_random_characters(self):
        s = "sdaw aef asdf w"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)

    def test_chinese_with_english_words(self):
        s = "abc-土土"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)

    def test_chinese_words(self):
        s = "土地 土土地"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)

    def test_chinese_words_with_punctuation(self):
        s = "土地 '土'土地"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)

    def test_englishwords_with_punctuation(self):
        s = "I'm saying 'running' this morning! Huggingface"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)

    def test_unknown_words(self):
        s = "動 動動"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)

    def test_unknown_words_with_known_words(self):
        s = "you are動 動動bye bye"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)

    def test_random_sentences(self):
        path = os.path.join(str(Path(__file__).resolve().parent), "tests.txt")
        with open(path, 'r') as f:
            sentences = f.read().split('\n')
            for s in sentences:
                lib_res, my_res = self.tokenize_with_both_tokenizer(s)
                self.assertEqual(lib_res, my_res)
    
    def test_hashtags(self):
        s = "you are #good! ## bye bye"
        lib_res, my_res = self.tokenize_with_both_tokenizer(s)
        self.assertEqual(lib_res, my_res)


if __name__ == '__main__':
    unittest.main()