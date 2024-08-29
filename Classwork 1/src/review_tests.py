import unittest
import pytest
from review_exercises import *

class ReviewTests(unittest.TestCase):
    """
    Unit tests for validating answers on the review exercise. Notes:
    - Your correctness score on assignments will typically be assessed by a more complete,
      grading set of unit tests compared to a subset provided in the skeleton, but for this
      introduction, the full set of grading tests are provided.
    - A portion of your style grade will also come from proper type hints; remember to
      validate your submission using `mypy .` and ensure that no issues are found.
    """
    
    # get_conflict tests
    # ---------------------------------------------------------------------------
    
    def test_get_conflict_basic(self) -> None:
        self.assertEqual("A", get_conflict({"A": True}, {"A": False}))
        self.assertEqual("A", get_conflict({"A": True, "B": False, "C": True}, {"A": False, "B": False, "C": True}))
        self.assertIn(get_conflict({"A": True, "B": False, "C": True}, {"A": False, "B": True, "C": True}), {"A", "B"})
        
    def test_get_conflict_nones(self) -> None:
        self.assertEqual(None, get_conflict({"A": True}, {}))
        self.assertEqual(None, get_conflict({"A": True}, {"B": False}))
        self.assertEqual(None, get_conflict({"A": False, "B": False, "C": True}, {"A": False, "B": False, "C": True}))
        
    def test_get_conflict_efficiency(self) -> None:
        BIG_TEST_SIZE = 30000
        big_test_1 = {str(i): (i%2 == 0) for i in range(BIG_TEST_SIZE)}
        big_test_2 = big_test_1.copy()
        big_test_1[str(BIG_TEST_SIZE+1)] = False
        big_test_2[str(BIG_TEST_SIZE+1)] = True
        EFF_ERR = "[X] If your test timed out, it means you have an inefficient use of your data structures -- remember, dictionaries are hash tables!"
        self.assertEqual(str(BIG_TEST_SIZE+1), get_conflict(big_test_1, big_test_2), EFF_ERR)
    
    # count_set_member_pairs tests
    # ---------------------------------------------------------------------------
    
    def test_count_set_member_pairs(self) -> None:
        self.assertEqual(0, count_set_member_pairs([]))
        self.assertEqual(0, count_set_member_pairs([{"A"}]))
        self.assertEqual(0, count_set_member_pairs([
            {"A"},
            {"B"}
        ]))
        self.assertEqual(1, count_set_member_pairs([
            {"A"},
            {"B", "A"}
        ]))
        self.assertEqual(2, count_set_member_pairs([
            {"A"},
            {"B", "A"},
            {"C", "B"}
        ]))
        self.assertEqual(3, count_set_member_pairs([
            {"A", "C"},
            {"B", "A"},
            {"C", "B"}
        ]))
        self.assertEqual(6, count_set_member_pairs([
            {"A", "C"},
            {"B", "A"},
            {"C", "B"},
            {"A", "B", "C"}
        ]))
        self.assertEqual(0, count_set_member_pairs([
            {"A", "D"},
            {"B", "E"},
            {"C", "G"},
            {"X", "Y", "Z"}
        ]))
        
    def test_count_set_member_pairs_efficiency(self) -> None:
        BIG_TEST_SIZE = 1000
        BIG_TEST_SET_SIZE = 20
        big_list = [{str(i+j) for j in range(BIG_TEST_SET_SIZE)} for i in range(BIG_TEST_SIZE)]
        EFF_ERR = "[X] If your test timed out, it means you have an inefficient use of your data structures -- remember, sets are hash tables!"
        self.assertEqual(18810, count_set_member_pairs(big_list), EFF_ERR)
        
if __name__ == '__main__':
    unittest.main()