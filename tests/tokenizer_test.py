from transformers import BertTokenizer
import unittest
import timeit

import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.word_piece_tokenizer.WordPieceTokenizer import WordPieceTokenizer


class TestTokenizer(unittest.TestCase):
    performance = []

    def setUp(self, compare_performance=True):
        self._bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')
        self._my_tokenizer = WordPieceTokenizer()
        self._compare_performance = compare_performance
    
    def tearDown(self):
        if self._compare_performance:
            print(f"[Average] This tokenizer is {sum(TestTokenizer.performance) / len(TestTokenizer.performance) * 100:.2f}% faster")

    def tokenize_with_both_tokenizer(self, s: str):
        print()
        print(s)

        time_taken = []

        if (self._compare_performance): start_time = timeit.default_timer()
        batch = self._bert_tokenizer([s])
        lib_res = batch.input_ids[0]
        if (self._compare_performance): time_taken.append(timeit.default_timer() - start_time)
        if (self._compare_performance): start_time = timeit.default_timer()
        my_res = self._my_tokenizer.tokenize(s)
        if (self._compare_performance): time_taken.append(timeit.default_timer() - start_time)

        print(lib_res)
        print(my_res)

        if (self._compare_performance):
            print("\n Performance Results:")
            print(f"BERT tokenizer: {time_taken[0]}")
            print(f"This tokenizer: {time_taken[1]}")
            performance = 1 - time_taken[1] / time_taken[0]
            print(f"This tokenizer is {100 * performance:.2f}% faster")
            TestTokenizer.performance.append(performance)

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