'''
Pacman Agent employing a PacNet trained in another module to
navigate perilous ghostly pellet mazes.
'''

import time
import random
import numpy as np
import torch
from torch import nn
from pathfinder import *
from queue import Queue
from constants import *
from pac_trainer import *

class PacmanAgent:
    '''
    Deep learning Pacman agent that employs PacNets trained in the pac_trainer.py
    module.
    '''

    def __init__(self, maze):
        """
        Initializes the PacmanAgent with any attributes needed to make decisions;
        for the deep-learning implementation, really just needs the model and
        its plan Queue.
        :maze: The maze on which this agent is to operate. Must be the same maze
        structure as the one on which this agent's model was trained. (Will be
        same format as Constants.MAZE)
        """
        self.model = PacNet(maze)

        self.model.load_state_dict(torch.load(Constants.PARAM_PATH, weights_only=True))
        
        self.model.eval()
        return

    def choose_action(self, perception, legal_actions):
        """
        Returns an action from the options in Constants.MOVES based on the agent's
        perception (the current maze) and legal actions available
        :perception: The current maze state in which to act
        :legal_actions: Map of legal actions to their next agent states
        :return: String action choice from the set of legal_actions
        """
        vectorized_maze = PacmanMazeDataset.vectorize_maze(perception)

        outputs = self.model(vectorized_maze)
        
        # sort the outputs
        sorted_indices = torch.argsort(outputs, descending=True)

        # choose the best move if it is legal
        legal_moves = [action[0] for action in legal_actions]
        for index in sorted_indices:
            if Constants.MOVES[index] in legal_moves:
                return Constants.MOVES[index]

        return random.choice(legal_actions)[0]
