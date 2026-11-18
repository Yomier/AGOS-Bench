"""Assistant-only supervision and prefix integrity on synthetic token batches."""
import unittest

from training import train

class TrainLabelTests(unittest.TestCase):
    def test_prefix_is_ignored_and_assistant_tokens_are_labelled(self):
        labels, supervised = train.build_training_labels([1, 2, 3], [1, 2, 3, 4, 5])
        self.assertEqual(labels, [-100, -100, -100, 4, 5])
        self.assertEqual(supervised, 2)

    def test_prefix_mismatch_is_hard_rejected(self):
        with self.assertRaises(train.TrainDataError) as caught:
            train.build_training_labels([1, 2, 3], [1, 9, 3, 4])
        self.assertIn('prefix mismatch', str(caught.exception))

    def test_longer_prompt_than_full_is_rejected(self):
        with self.assertRaises(train.TrainDataError):
            train.build_training_labels([1, 2, 3], [1, 2])

    def test_no_supervised_tokens_is_hard_rejected(self):
        with self.assertRaises(train.TrainDataError) as caught:
            train.build_training_labels([1, 2, 3], [1, 2, 3])
        self.assertIn('no supervised', str(caught.exception))

    def test_padding_positions_are_ignored(self):
        labels, supervised = train.build_training_labels(
            [1, 2, 3], [1, 2, 3, 4, 5], attention_mask=[1, 1, 1, 1, 0])
        self.assertEqual(labels, [-100, -100, -100, 4, -100])
        self.assertEqual(supervised, 1)

    def test_attention_mask_length_mismatch_is_rejected(self):
        with self.assertRaises(train.TrainDataError):
            train.build_training_labels([1, 2, 3], [1, 2, 3, 4], attention_mask=[1, 1])
