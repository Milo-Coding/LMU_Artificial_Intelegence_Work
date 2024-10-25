'''
Constants to be used across various Ad Agent and Decision Network
tests, including network structure configurations and test file locations.

[!] Feel free to edit this file at will
[!] Note: any data frame in here should not be modified!
'''

import pandas as pd

# Lecture 5-2 Example
# -----------------------------------------------------------------------------------------
LECTURE_5_2_DATA = pd.read_csv("../dat/lecture5-2-data.csv")
LECTURE_5_2_STRUC = [("M", "C"), ("D", "C")]
LECTURE_5_2_DEC = ["D"]
LECTURE_5_2_UTIL = {"C": {0: 3, 1: 1}}

# Do-test Example
# -----------------------------------------------------------------------------------------
DO_TEST_DATA = pd.read_csv("../dat/do-test-data.csv")
DO_TEST_STRUC = [("Z", "D"), ("Z", "S"), ("D", "S"), ("S", "Y")]
DO_TEST_DEC = ["D"]
DO_TEST_UTIL = {"Y": {0: 100, 1: 0}}

# AdBot Example
# -----------------------------------------------------------------------------------------
ADBOT_DATA = pd.read_csv("../dat/adbot-data.csv")
ADBOT_STRUC = [("A", "F"), ("A", "H"), ("A", "T"), ("Ad1", "G"), ("Ad1", "S"), ("Ad2", "F"),
               ("F", "S"), ("G", "S"), ("H", "I"), ("P", "T"), ("P", "G")]
ADBOT_DEC = ["Ad1", "Ad2"]
ADBOT_UTIL = {"S": {0: 0, 1: 1776, 2: 500}}
