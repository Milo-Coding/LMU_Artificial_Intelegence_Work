'''
ad_engine.py
Advertisement Selection Engine that employs a Decision Network
to Maximize Expected Utility associated with different Decision
variables in a stochastic reasoning environment.
'''
import math
import itertools
import unittest
import numpy as np
import pandas as pd
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork


class AdEngine:

    def __init__(self, data: "pd.DataFrame", structure: list[tuple[str, str]], dec_vars: list[str], util_map: dict[str, dict[int, int]]):
        """
        Responsible for initializing the Decision Network of the
        AdEngine by taking in the dataset, structure of network,
        any decision variables, and a map of utilities

        Parameters:
            data (pd.DataFrame):
                Pandas data frame containing all data on which the decision
                network's chance-node parameters are to be learned
            structure (list[tuple[str, str]]):
                The Bayesian Network's structure, a list of tuples denoting
                the edge directions where each tuple is (parent, child)
            dec_vars (list[str]):
                list of string names of variables to be
                considered decision variables for the agent. Example:
                ["Ad1", "Ad2"]
            util_map (dict[str, dict[int, int]]):
                Discrete, tabular, utility map whose keys
                are variables in network that are parents of a utility node, and
                values are dictionaries mapping that variable's values to a utility
                score, for example:
                  {
                    "X": {0: 20, 1: -10}
                  }
                represents a utility node with single parent X whose value of 0
                has a utility score of 20, and value 1 has a utility score of -10
        """

        # Build the network structure from the edges
        self.model = BayesianNetwork(structure)

        # "Fit" the model = learn the CPTs from the data and structure
        self.model.fit(data)

        self.inference = CausalInference(self.model)

        # Store the decision variables
        self.decision_vars = dec_vars

        # Infer chance variables (non-decision/utility nodes)
        self.chance_vars = [var for var in data.columns if var not in dec_vars]

        # Store the utility map
        self.utility_map = util_map

        # Store the parents of the utility node
        self.utility_chance_parents = [var for var in util_map.keys()]

        return

    def meu(self, evidence: dict[str, int]) -> tuple[dict[str, int], float]:
        """
        Computes the Maximum Expected Utility (MEU) defined as the choice of
        decision variable values that maximize expected utility of any evaluated
        chance nodes given in the agent's utility map.

        Parameters:
            evidence (dict[str, int]):
                dict mapping network variables to their observed values, 
                of the format: {"Obs1": val1, "Obs2": val2, ...}

        Returns: 
            tuple[dict[str, int], float]:
                A 2-tuple of the format (a*, MEU) where:
                [0] is a dictionary mapping decision variables to their MEU states
                [1] is the MEU value (a float) of that decision combo
        """
        # List to store the expected utility for each decision combo
        exp_utils = []

        # Generate all combinations of binary decisions for decision variables
        decision_combos = itertools.product([0, 1], repeat=len(self.decision_vars))

        for combo in decision_combos:
            # Create a decision dictionary for the current combination Ex: {'Ad1': 0, 'Ad2': 0}
            decision_dict = dict(zip(self.decision_vars, combo))

            # get probabilities given the evidence and "do" operator for decisions
            query = self.inference.query(self.utility_chance_parents, evidence=evidence, do=decision_dict, show_progress=False)

            for utility_var in self.utility_chance_parents:
                # Expected utility is the sum of P(s|do(a),e) * U(s,a) over all a
                expected_utility = sum(query.values[i] * self.utility_map[utility_var][i] for i in range(len(query.values)))

            # Store the decision combination and its expected utility
            exp_utils.append((decision_dict, expected_utility))

       # Find the decision combination with the highest expected utility
        best_decision = max(exp_utils, key=lambda eu: eu[1])

        # Return the best decision and its MEU value
        return (best_decision[0], best_decision[1])

    def vpi(self, potential_evidence: str, observed_evidence: dict[str, int]) -> float:
        """
        Given some observed demographic "evidence" about a potential
        consumer, returns the Value of Perfect Information (VPI)
        that would be received on the given "potential" evidence about
        that consumer.

        Parameters:
            potential_evidence (str):
                string representing the variable name of the variable 
                under consideration for potentially being obtained
            observed_evidence (tuple[dict[str, int], float]):
                dict mapping network variables 
                to their observed values, of the format: 
                {"Obs1": val1, "Obs2": val2, ...}

        Returns:
            float:
                float value indicating the VPI(potential | observed)
        """
        meu_observed = self.meu(observed_evidence)[1]
        meu_potential = 0

        # calculate the potiential value of knowing the evidence
        query = self.inference.query(
            [potential_evidence], evidence=observed_evidence, show_progress=False)
        
        for i in range(len(query.values)):
            evidence = observed_evidence.copy()
            # we know there is only one var given by potential evidence so we can add it directly
            evidence[potential_evidence] = i
            meu_potential += query.values[i] * self.meu(evidence)[1]

        # vpi = meu(E',E) - meu(E)
        vpi = meu_potential - meu_observed

        # Information can never be harmful
        if vpi < 0:
            return 0
        return vpi

    def most_likely_consumer(self, evidence: dict[str, int]) -> dict[str, int]:
        """
        Given some known traits about a particular consumer, makes the best guess
        of the values of any remaining hidden variables and returns the completed
        data point as a dictionary of variables mapped to their most likely values.
        (Observed evidence will always have the same values in the output).

        Parameters:
            evidence (dict[str, int]):
                dict mapping network variables 
                to their observed values, of the format: 
                {"Obs1": val1, "Obs2": val2, ...}

        Returns:
            dict[str, int]:
                The most likely values of all variables given what's already
                known about the consumer.
        """
        # make a new inference using Variable Elimination
        inference = VariableElimination(self.model)

        # separate the known and unkown vars
        unknown_vars = self.chance_vars.copy()
        for known_var in evidence.keys():
            unknown_vars.remove(known_var)

        # calculate the most likely values for the unknown variables
        most_likely = inference.map_query(variables=unknown_vars, evidence=evidence, virtual_evidence=None, elimination_order='MinFill', show_progress=False)

        # return the most likely state of the consumer given the evidence we have
        return {**most_likely, **evidence}
