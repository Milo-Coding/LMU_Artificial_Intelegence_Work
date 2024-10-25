'''
warmup.py

Skeleton for answering warmup questions related to the
AdAgent assignment. By the end of this section, you should
be familiar with:
- Importing, selecting, and manipulating data using Pandas
- Creating and Querying a Bayesian Network
- Using Samples from a Bayesian Network for Approximate Inference

@author: <Milo Fritzen>
'''

from pgmpy.inference.CausalInference import CausalInference
from pgmpy.models import BayesianNetwork
import numpy as np
import pandas as pd

if __name__ == '__main__':
    """
    PROBLEM 2
    Using the pgmpy query example, determine the answers to the
    queries specified in the instructions.

    (just print out the CPT values with their labels and save to report)
    """
    # Load the data into a pandas data frame
    csv_data = pd.read_csv("../dat/warmup-data.csv")

    # Set the edges of the network: tuples of the format (parent, child)
    edges = [("W", "X"), ("W", "Y"), ("X", "Z"), ("Y", "Z")]

    # Build the network structure from the edges
    model = BayesianNetwork(edges)

    # "Fit" the model = learn the CPTs from the data and structure
    model.fit(csv_data)

    # Create the inference engine using the Variable Elimination algorithm
    # (a more efficient enumeration inference)
    inference = CausalInference(model)

    # Queries from HW:
    # P(W)
    query = inference.query(["W"], evidence={}, show_progress=False)
    print("P(W)", query)

    # P(X | W=1)
    query = inference.query(["X"], evidence={"W": 1}, show_progress=False)
    print("P(X | W=1)", query)

    # P(Z | W=0, do(X=0))
    query = inference.query(["Z"], evidence={"W": 1}, do={
                            "X": 0}, show_progress=False)
    print("P(Z | W=0, do(X=0))", query)

    # To programmatically access those table values, you can access the
    # values attribute of the CPT and then grab any by index, e.g.,
    print("P(Z | W=0, do(X=0)) = " + str(query.values[0]))
