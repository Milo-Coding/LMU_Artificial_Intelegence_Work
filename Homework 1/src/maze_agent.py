import time
import random
import math
import heapq
import multiprocessing
from typing import Optional
from queue import Queue
from constants import *
from maze_clause import *
from maze_knowledge_base import *

class MazeAgent:
    '''
    BlindBot MazeAgent meant to employ Propositional Logic,
    Planning, and Active Learning to navigate the Pitsweeper
    Problem. Have fun!
    '''
    
    def __init__ (self, env: "Environment", perception: dict) -> None:
        """
        Initializes the MazeAgent with any attributes it will need to
        navigate the maze.
        [!] Add as many attributes as you see fit!
        
        Parameters:
            env (Environment):
                The Environment in which the agent is operating; make sure
                to see the spec / Environment class for public methods that
                your agent will use to solve the maze!
            perception (dict):
                The starting perception of the agent, which is a
                small dictionary with keys:
                  - loc:  the location of the agent as a (c,r) tuple
                  - tile: the type of tile the agent is currently standing upon
        """
        self.env: "Environment" = env
        self.goal: tuple[int, int] = env.get_goal_loc()
        # our starting location
        self.starting_loc = self.env.get_player_loc()
        
        # The agent's maze can be manipulated as a tracking mechanic
        # for what it has learned; changes to this maze will be drawn
        # by the environment and is simply for visuals / debugging
        # [!] Feel free to change self.maze at will
        self.maze: list = env.get_agent_maze()
        
        # Standard set of attributes you'll want to maintain
        self.kb: "MazeKnowledgeBase" = MazeKnowledgeBase()
        self.possible_pits: set[tuple[int, int]] = set()
        self.safe_tiles: set[tuple[int, int]] = set()
        self.pit_tiles: set[tuple[int, int]] = set()

        # add the initial rules 
        self.add_starting_rules(perception)
        
    ##################################################################
    # Methods
    ##################################################################

    def add_starting_rules(self, perception: dict) -> None:
        """
        Adds our intial knowledge and inferences to the knowledge base

        Parameters:
            perception (dict):
                A dictionary providing the agent's current location
                and current tile type being stood upon, of the format:
                {"loc": (x, y), "tile": tile_type}
        """
        # the start and goal are always safe tiles so we can start out knowing that
        self.safe_tiles.update({self.starting_loc, self.goal})

        # get all the paths to the goal
        paths_to_goal = self.env.get_cardinal_locs(self.goal, 1)

        # for each of the adjacnet tiles, create a prop that there is no pit and store it in a list
        props: list[tuple[tuple[str, tuple[int, int]], bool]] = []
        for loc in paths_to_goal:
            props.append(((Constants.PIT_BLOCK, loc), False))

        # we know there is always at least one safe path to the goal
        self.kb.tell(MazeClause(props))

        # if there was only one prop, we know that tile is safe
        if (len(props) == 1):
            self.safe_tiles.add(props[0][0][1])

        # if we start on a safe tile, we know our adjacent tiles are safe too
        if (perception["tile"] == Constants.SAFE_BLOCK or perception["tile"] == Constants.WRN_ZERO_BLOCK):
            self.safe_tiles.update(self.env.get_cardinal_locs(self.starting_loc, 1))

        # if we start on a warning tile, use the helper method for warning tiles
        if perception["tile"] in Constants.WRN_BLOCKS:
            self.on_warning_tile(perception["tile"], self.env.get_cardinal_locs(self.starting_loc, 1))
    
    def think(self, perception: dict) -> tuple[int, int]:
        """
        The main workhorse method of how your agent will process new information
        and use that to make deductions and decisions. In gist, it should follow
        this outline of steps:
        1. Process the given perception, i.e., the new location it is in and the
           type of tile on which it's currently standing (e.g., a safe tile, or
           warning tile like "1" or "2")
        2. Update the knowledge base and record-keeping of where known pits and
           safe tiles are located, as well as locations of possible pits.
        3. Query the knowledge base to see if any locations that possibly contain
           pits can be deduced as safe or not.
        4. Use all of the above to prioritize the next location along the frontier
           to move to next.
        
        Parameters:
            perception (dict):
                A dictionary providing the agent's current location
                and current tile type being stood upon, of the format:
                {"loc": (x, y), "tile": tile_type}
        
        Returns:
            tuple[int, int]:
                The maze location along the frontier that your agent will try to
                move into next.
        """
        # start by checking what sort of tile we're on
        current_tile = perception["tile"]
        current_loc = perception["loc"]

        # if we didn't already have it, add  the current tile to the kb
        if current_tile != Constants.PIT_BLOCK:
            self.kb.tell(MazeClause([((Constants.PIT_BLOCK, current_loc), False)]))
        else:
            self.kb.tell(MazeClause([((Constants.PIT_BLOCK, current_loc), True)]))

        # we may also be able to learn something new about the surrounding tiles
        adjacent_tiles = self.env.get_cardinal_locs(current_loc, 1)

        # if it's a clear tile, adjacent tiles are safe to traverse. WRN_ZERO_BLOCK shouldn't appear but it's basically the same thing
        if current_tile == Constants.SAFE_BLOCK or current_tile == Constants.WRN_ZERO_BLOCK:
            # we know that the tiles surrounding tiles are not pits
            for neighbor in adjacent_tiles:
                self.safe_tiles.add(neighbor)
                self.kb.tell(MazeClause([((Constants.PIT_BLOCK, neighbor), False)]))
        
        # if it's a warning tile, call the helper method that handles warning tiles
        if current_tile in Constants.WRN_BLOCKS:
            self.on_warning_tile(current_tile, adjacent_tiles)
        
        # use the pathfinding method to pick the best move to make
        return self.find_best_next_move(self.env.get_frontier_locs(), current_loc)  
    
    def on_warning_tile(self, current_tile: str, adjacent_tiles: set[tuple[int, int]]) -> None:
        """
        Helper method that considers where pits are located when we land on a warning tile.
        It consideres how many adjacent pits there are are tries to infer the location of
        those pits.

        Parameters:
            current_tile str:
                the type of tile we are on (should always be a type of warning tile)
            adjacent_tiles set[tuple[int, int]]:
                the list of tiles we checking for pits
        """
        # all pits all around: this should never happen but we wanna be prepared for it
        if current_tile == Constants.WRN_FOUR_BLOCK:
            # every adjacent tile is a pit
            for neighbor in adjacent_tiles:
                self.pit_tiles.add(neighbor)
                self.kb.tell(MazeClause([((Constants.PIT_BLOCK, neighbor), True)]))

        # three pits: at least we know how we got here so time to backtrack
        elif current_tile == Constants.WRN_THREE_BLOCK:
            for neighbor in adjacent_tiles:
                # the tiles we don't know to be safe from walking here are pits
                if neighbor not in self.safe_tiles:
                    self.pit_tiles.add(neighbor)
                    self.kb.tell(MazeClause([((Constants.PIT_BLOCK, neighbor), True)]))

        # two pits: time for inference
        elif current_tile == Constants.WRN_TWO_BLOCK:
            self.infer_adjacent_pits(adjacent_tiles, 2)

        # one pit: also time for inference
        elif current_tile == Constants.WRN_ONE_BLOCK:
            self.infer_adjacent_pits(adjacent_tiles, 1)

    def infer_adjacent_pits(self, adjacent_tiles: set[tuple[int, int]], num_pits: int) -> None:
        """
        Helper method for on_warning_tile that checks to see if we can infer which of the adjacent
        tiles are pits.

        Parameters:
            adjacent_tiles set[tuple[int, int]]:
                the list of tiles we checking for pits
            num_pits int:
                the number of pits in adjacent_tiles
        """
        # case 1: we know enough to identify the tiles without inference
        # make a copy of adjacent_tiles to modify
        adjacent_copy = adjacent_tiles.copy()
        # remove any safe tiles we know of from the copy
        for tile in adjacent_copy.copy():  # so many copies
            # if we previously discovered the tile is safe
            if self.is_safe_tile(tile):
                # remove it from the copy
                adjacent_copy.remove(tile)

        # if we found all the non-pits
        if len(adjacent_copy) == num_pits:
            # we know everything else is a pit
            for tile in adjacent_copy:
                self.pit_tiles.add(tile)
                self.kb.tell(MazeClause([((Constants.PIT_BLOCK, tile), True)]))

            # all pits were found so we're done here
            return None
        
        # case 2: we try to infer the tile types
        # we need to trim down our list of adjacent tiles to 3 so it fits with the cnf logif from the notes
        if len(adjacent_tiles) > 3:
            # remove exactly one known safe tile from the adjecent tiles
            for tile in adjacent_tiles.copy():
                # if the tile is safe
                if self.is_safe_tile(tile):
                    # it's the path we took, so we don't need it
                    adjacent_tiles.remove(tile)
                    break

        # for the three unknown tiles
        if len(adjacent_tiles) == 3:
            # add the CNF rules to our knowledge base to see if we can resolve them
            self.CNF_to_KB(list(adjacent_tiles), num_pits)

        # in any other case we don't know what's going on
        return None
            

    def find_best_next_move(self, possible_moves: set[tuple[int, int]], current_loc: tuple[int, int]) -> tuple[int, int]:
        """
        Helper method for think() that picks the tile closest to the goal from a set of tiles.
        The intended use is to come up with a set of safe moves to make on the frontier then pass
        that set to this function to reduce unnececary exploration

        Parameters:
            possible_moves set[tuple[int, int]]:
                the list of tiles we are considering moving to

        Returns:
            the "hopefully" optimal next tile to explore to reach the goal
        """
        # for efficiency, sort frontier into priority queue based on what tiles look most promising
        priority_frontier: list[Tuple[float, Tuple[int, int]]] = []

        # loop through each possible move
        for possible_move in possible_moves:
            # we absolutely want to move to the goal if possible, no sorting needed
            if (possible_move == self.goal):
                return possible_move
            
            # get distance of that tile from goal and distance from the agent
            move_distance_from_goal: int = (abs(self.goal[0] - possible_move[0])) + abs((self.goal[1] - possible_move[1]))
            move_cost: int = (abs(current_loc[0] - possible_move[0])) + abs((current_loc[1] - possible_move[1]))

            # we want to prioritize tiles close to the goal with a low move cost
            priority: float = move_distance_from_goal + move_cost * 0.5  # distance from goal is more impactful than move cost

            # possible pits have a lower priority so to see if we'll learn about it from exploring nearby safe tiles first
            if possible_move in self.possible_pits:
                priority += 4

            # we typically don't want to look at tiles we already know are pits
            if possible_move in self.pit_tiles:
                priority += 20
            
            # the frontier will be sorted (smallest priority -> largest priority)
            priority_frontier.append((priority, possible_move))

        # convert our list of moves into a sorted priority queue
        heapq.heapify(priority_frontier)

        # for item in heapq
        for sorted_move in priority_frontier:
            # return the first safe move we find
            if self.is_safe_tile(sorted_move[1]):
                # the second item in sorted_move is the tile (first item is the priority for sorting)
                return sorted_move[1]

        # if nothing was returned, we don't know where any safe moves are, therefore just jump to the best unsafe tile
        return priority_frontier[0][1]
        
    def is_safe_tile(self, loc: tuple[int, int]) -> Optional[bool]:
        """
        Determines whether or not the given maze location can be concluded as
        safe (i.e., not containing a pit), following the steps:
        1. Check to see if the location is already a known pit or safe tile,
           responding accordingly
        2. If not, performs the necessary queries on the knowledge base in an
           attempt to deduce its safety
        
        Parameters:
            loc (tuple[int, int]):
                The maze location in question
        
        Returns:
            One of three return values:
            1. True if the location is certainly safe (i.e., not pit)
            2. False if the location is certainly dangerous (i.e., pit)
            3. None if the safety of the location cannot be currently determined
        """
        # case 1: quick lookup
        # returns true if location is known to be safe
        if loc in self.safe_tiles:
            return True

        # returns false if the tile is known to be unsafe
        elif loc in self.pit_tiles:
            return False
        
        # case 2: use the ask method with a timout for long queries:
        ASK_TIMEOUT: float = 0.25  # maximum time in seconds ask() can run before timing out

        # start by reducing our kb as we may have redundant info
        self.kb.simplify_self(self.pit_tiles, self.safe_tiles)

        # store this query in an optional bool in case it times out
        no_pit: Optional[bool] = self.ask_with_timeout(MazeClause([((Constants.PIT_BLOCK, loc), False)]), ASK_TIMEOUT)
        
        # if a safe tile is infered at the target location
        if no_pit:
            # the tile is safe
            self.safe_tiles.add(loc)
            self.kb.tell(MazeClause([((Constants.PIT_BLOCK, loc), False)]))
            return True

        # if the query timed out, the second one probably will too so just skip it now
        elif no_pit is None:
            self.possible_pits.add(loc)
            return None
        
        # if a pit is infered at the target location
        if self.ask_with_timeout(MazeClause([((Constants.PIT_BLOCK, loc), True)]), ASK_TIMEOUT):
            # the tile is a pit
            self.pit_tiles.add(loc)
            self.kb.tell(MazeClause([((Constants.PIT_BLOCK, loc), True)]))
            return False
        
        # if we can't infer it quickly, we are unsure if it is a pit
        self.possible_pits.add(loc)
        return None
    
    def ask_with_timeout(self, clause: "MazeClause", timeout: float):
        """
        Helper for is_safe_tile that calls the ask method but terminates if it
        takes too long to return a response to prevent timeing out of tests

        Parameters:
            clause "MazeClause":
                The MazeClause we want to ask about
            timeout float:
                The number of seconds we let the function run for
        
        Returns:
            One of three return values:
            1. True if the location is certainly safe (i.e., not pit)
            2. False if the location is certainly dangerous (i.e., pit)
            3. None if the safety of the location cannot be determined quickly
        """
        # create a multiprocessing queue to run ask and timeout at the same time
        result_queue: "multiprocessing.queues.Queue[bool]" = multiprocessing.Queue()
        # create a process that can use the ask method
        process = multiprocessing.Process(target=self.ask_in_process, args=(clause, result_queue,))
        
        # start the process
        process.start()

        # add the timeout to our multiprocessing queue
        process.join(timeout)

        # when the ask returns a value or the timeout reaches zero:
        if process.is_alive():
            # if ask is still running, terminate it
            process.terminate()
            # cleanup the timeout
            process.join()
            # no solution found so return None
            return None
        else:
            # if process finished in time, get the result from the queue
            return result_queue.get() if not result_queue.empty() else None
        
    def ask_in_process(self, clause: "MazeClause", result_queue: "multiprocessing.queues.Queue[bool]"):
        """
        Function to put the ask method into our multiprocessing queue

         Parameters:
            clause "MazeClause":
                The clause we are asking the kb about
            result_queue "multiprocessing.queues.Queue[bool]"
                a multiprocessing queue for the ask method and timout to run in
        """
        # add self.kb.ask(clause) to the queue
        result = self.kb.ask(clause)
        result_queue.put(result)
    
    def CNF_to_KB(self, tiles_to_check: list[tuple[int,int]], possible_pits: int) -> None:
        """
        Uses CNF logic to tell our knowledge base where the possible pits may be
        
        Parameters:
            tiles_to_check list[tuple[int,int]]:
                The list of three locations where the pits may be
            possible_pits int:
                The number of pits which determines which set of rules we use
        """
        # match case faster than if statments
        match possible_pits:
            # If there are 2 possible pits nearby
            case 2:
                self.kb.tell(MazeClause([((Constants.PIT_BLOCK, tiles_to_check[0]), True), ((Constants.PIT_BLOCK, tiles_to_check[1]), True)]))
                self.kb.tell(MazeClause([((Constants.PIT_BLOCK, tiles_to_check[0]), True), ((Constants.PIT_BLOCK, tiles_to_check[2]), True)]))
                self.kb.tell(MazeClause([((Constants.PIT_BLOCK, tiles_to_check[1]), True), ((Constants.PIT_BLOCK, tiles_to_check[2]), True)]))
                self.kb.tell(MazeClause([((Constants.PIT_BLOCK, tiles_to_check[0]), False), ((Constants.PIT_BLOCK, tiles_to_check[1]), False), ((Constants.PIT_BLOCK, tiles_to_check[2]), False)]))

            # If there's 1 possible nearby        
            case 1:
                self.kb.tell(MazeClause([((Constants.PIT_BLOCK, tiles_to_check[0]), False), ((Constants.PIT_BLOCK, tiles_to_check[1]), False)]))
                self.kb.tell(MazeClause([((Constants.PIT_BLOCK, tiles_to_check[0]), False), ((Constants.PIT_BLOCK, tiles_to_check[2]), False)]))
                self.kb.tell(MazeClause([((Constants.PIT_BLOCK, tiles_to_check[1]), False), ((Constants.PIT_BLOCK, tiles_to_check[2]), False)]))
                self.kb.tell(MazeClause([((Constants.PIT_BLOCK, tiles_to_check[0]), True), ((Constants.PIT_BLOCK, tiles_to_check[1]), True), ((Constants.PIT_BLOCK, tiles_to_check[2]), True)]))

# Declared here to avoid circular dependency
from environment import Environment