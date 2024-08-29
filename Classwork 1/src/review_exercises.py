'''
CMSI 3300 - Classwork 1
Author: SOLUTION

Complete each exercise as described in the Classwork spec, ensuring that its subsequent
unit test is satisfied by running the associated review_tests.py as indicated in the spec.
'''

from typing import *
import itertools


def get_conflict(dict1: dict[str, bool], dict2: dict[str, bool]) -> Optional[str]:
    '''
    Given 2 dictionaries of string keys mapped to boolean values, returns the first
    key that is mapped to True in one dictionary but False in the other. In the event
    of no such "conflicts," returns None, and if multiple conflicts exists, can return
    any one of them.
    
    Parameters:
        dict1, dict2: dict[str, bool]:
            Dictionaries of string keys mapped to boolean values.
    
    Returns:
          Optional[str]:
              The first key with a conflicting boolean value between dictionaries, or
              None if no such key exists. 
        
    Examples:
        get_conflict({"A": True, "B": False, "C": True}, {"A": False, "B": False, "C": True}) => "A"
        get_conflict({"A": True}, {"B": False}) => None
    '''
    # To reduce unnececary checks, loop through the smaller dictionary
    if len(dict1) > len(dict2):
        small, big = dict2, dict1
    else:
        small, big = dict1, dict2
    
    # For each key that is in both dictionaries, check if there is a conflict
    for key in small:
        if key in big and small[key] != big[key]:
            # If a conflict is found, return the conflict
            return key
    # If no conflicts are found, return None
    return None


def count_set_member_pairs(sets: list[set[str]]) -> int:
    '''
    Given a list of sets of strings, counts the number of pairs that can be made between
    these sets where a pair can be formed whenever at least one string is common between
    both sets. Should count each possible pair exactly once.
    
    [!] Hint: use a particular method in the itertools package to simplify your life!
    
    Parameters:
        sets: list[set[str]]:
            A list of sets of strings.
    
    Returns:
          int:
              The number of pairs that could be formed between sets in the given list.
        
    Examples:
        count_set_member_pairs([
            {"A"},
            {"B", "A"},
            {"C", "B"}
        ]) => 2     # (the first two and last two sets can be paired)
        
        count_set_member_pairs([
            {"A", "C"},
            {"B", "A"},
            {"C", "B"},
            {"A", "B", "C"}
        ]) => 6    # (can you count all of the pairs?)
        
        self.assertEqual(0, count_set_member_pairs([
            {"A", "D"},
            {"B", "E"},
            {"C", "G"},
            {"X", "Y", "Z"}
        ])) => 0    # (no common strings in these sets)
    '''
    num_pairs = 0
    # Loop through all possible set pairs
    for set1, set2 in itertools.combinations(sets, 2):
        # If there is an common string, count the pair
        if set1 & set2: # the & sign checks for shared values
            num_pairs += 1
    # Return the final tally of pairs
    return num_pairs

