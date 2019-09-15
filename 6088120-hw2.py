"""A module for homework 2. Version 3."""
# noqa: D413

import abc
import heapq
from collections import defaultdict

from hw1 import EightPuzzleState, EightPuzzleNode


def eightPuzzleH1(state, goal_state):
    """
    Return the number of misplaced tiles including blank.

    Parameters
    ----------
    state : EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])
    goal_state : EightPuzzleState
        a desired 8-puzzle state.

    Returns
    ----------
    int

    """
    misplace: int = 0
    for y in range(3):
        for x in range(3):
            if int(state.board[x][y]) != int(goal_state.board[x][y]):
                misplace += 1
            #print(str(state.board[x][y]) + " " + str(goal_state.board[x][y]))
    return misplace
    # TODO 1:
    pass


def eightPuzzleH2(state, goal_state):
    """
    Return the total Manhattan distance from goal position of all tiles.

    Parameters
    ----------
    state : EightPuzzleState
        an 8-puzzle state containing a board (List[List[int]])
    goal_state : EightPuzzleState
        a desired 8-puzzle state.

    Returns
    ----------
    int

    pos_val
    0 1 2
    1 2 3
    2 3 4

    goal
    1 2 3
    4 5 6
    7 8 0

    """
    man = 0
    for x in range(3):
        for y in range(3):
            pos_val = x + y
            #print("x"+str(x)+"y"+str(y))

            if state.board[x][y] == 0:
                man += abs(pos_val - 4)

            if state.board[x][y] == 1:
                man += pos_val

            if state.board[x][y] == 2:
                man += x + abs(y - 1)

            if state.board[x][y] == 3:
                man += x + abs(y - 2)

            if state.board[x][y] == 4:
                man += abs(x - 1) + y

            if state.board[x][y] == 5:
                man += abs(x - 1) + abs(y - 1)

            if state.board[x][y] == 6:
                man += abs(x - 1) + abs(y - 2)

            if state.board[x][y] == 7:
                man += abs(x - 2) + y

            if state.board[x][y] == 8:
                man += abs(x - 2) + abs(y - 1)
    return man
    # TODO 2:
    pass


class Frontier(abc.ABC):
    """An abstract class of a frontier."""

    def __init__(self):
        """Create a frontier."""
        raise NotImplementedError()

    @abc.abstractmethod
    def is_empty(self):
        """Return True if empty."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def add(self, node):
        """
        Add a node into the frontier.

        Parameters
        ----------
        node : EightPuzzleNode

        Returns
        ----------
        None

        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def next(self):
        """
        Return a node from a frontier and remove it.

        Returns
        ----------
        EightPuzzleNode

        Raises
        ----------
        IndexError
            if the frontier is empty.

        """
        raise NotImplementedError()


class DFSFrontier(Frontier):
    """An example of how to implement a depth-first frontier (stack)."""

    def __init__(self):
        """Create a frontier."""
        self.stack = []

    def is_empty(self):
        """Return True if empty."""
        return len(self.stack) == 0

    def add(self, node):
        """
        Add a node into the frontier.
        
        Parameters
        ----------
        node : EightPuzzleNode

        Returns
        ----------
        None

        """

        for n in self.stack:
            # HERE we assume that state implements __eq__() function.
            # This could be improve further by using a set() datastructure,
            # by implementing __hash__() function.
            if n.state == node.state:
                return None
        self.stack.append(node)

    def next(self):
        """
        Return a node from a frontier and remove it.

        Returns
        ----------
        EightPuzzleNode

        Raises
        ----------
        IndexError
            if the frontier is empty.

        """
        return self.stack.pop()


class GreedyFrontier(Frontier):
    """A frontier for greedy search."""

    def __init__(self, h_func, goal_state):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state, goal_state)
            a heuristic function to score a state.
        goal_state : EightPuzzleState
            a goal state used to compute h()

        """
        self.h = h_func
        self.goal = goal_state
        self.heapq = []
        # TODO: 3
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.

    def is_empty(self):
        """Return True if empty."""
        return len(self.heapq) == 0

    def add(self, node):

        for n in self.heapq:
            if n.state == node.state:
                return None
        self.heapq.append(node)
        self.heapq.sort(key=self.h)

    def next(self):

        return self.heapq.pop()

class AStarFrontier(Frontier):
    """A frontier for greedy search."""

    def __init__(self, h_func, goal_state):
        """
        Create a frontier.

        Parameters
        ----------
        h_func : callable h(state)
            a heuristic function to score a state.
        goal_state : EightPuzzleState
            a goal state used to compute h()


        """
        self.g = 0
        self.h = h_func
        self.goal = goal_state
        self.heapq = []
        # TODO: 4
        # Note that you have to create a data structure here and
        # implement the rest of the abstract methods.

    def is_empty(self):
        return len(self.heapq) == 0

    def add(self, node):

        for n in self.heapq:
            if n.state == node.state:
                return None
            self.g += node.path_cost
            self.heapq.append(node)
        self.h += self.g
        self.heapq.sort(key=self.h)

    def next(self):

        return self.heapq.pop()

def _parity(board):
    """Return parity of a square matrix."""
    inversions = 0
    nums = []
    for row in board:
        for value in row:
            nums.append(value)
    for i in range(len(nums)):
        for j in range(i+ 1, len(nums)):
            if nums[i] != 0 and nums[j] != 0 and nums[i] > nums[j]:
                inversions += 1
    return inversions % 2


def _is_reachable(board1, board2):
    """Return True if two N-Puzzle state are reachable to each other."""
    return _parity(board1) == _parity(board2)


def graph_search(init_state, goal_state, frontier):
    """
    Search for a plan to solve problem.

    Parameters
    ----------
    init_state : EightPuzzleState
        an initial state
    goal_state : EightPuzzleState
        a goal state
    frontier : Frontier
        an implementation of a frontier which dictates the order of exploreation.
    
    Returns
    ----------
    plan : List[string] or None
        A list of actions to reach the goal, None if the search fails.
        Your plan should NOT include 'INIT'.
    num_nodes: int
        A number of nodes generated in the search.

    """
    if not _is_reachable(init_state.board, goal_state.board):
        return None, 0
    if init_state.is_goal(goal_state.board):
        return [], 0
    num_nodes = 0
    solution = []
    # Perform graph search
    root_node = EightPuzzleNode(init_state, action='INIT')
    cur_node = root_node
    while frontier is not None:
        if cur_node != root_node:
            solution.append(cur_node)
        cur_node = frontier.next()
        num_nodes += 1

    # TODO: 5
    return solution, num_nodes


def test_by_hand(verbose=True):
    """Run a graph-search."""
    goal_state = EightPuzzleState([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    init_state = EightPuzzleState.initializeState()

    while not _is_reachable(goal_state.board, init_state.board):
        init_state = EightPuzzleState.initializeState()
    #eightPuzzleH1(init_state, goal_state))
    #eightPuzzleH2(init_state, goal_state)
    frontier = GreedyFrontier(eightPuzzleH2(init_state, goal_state), goal_state)  # Change this to your own implementation.
    if verbose: 
        print(init_state)
    plan, num_nodes = graph_search(init_state, goal_state, frontier)
    if verbose: 
        print(f'A solution is found after generating {num_nodes} nodes.')
    if verbose: 
        for action in plan:
            print(f'- {action}')
    return len(plan), num_nodes


def experiment(n=10000):
    """Run experiments and report number of nodes generated."""
    result = defaultdict(list)
    for __ in range(n):
        d, n = test_by_hand(False)
        result[d].append(n)
    max_d = max(result.keys())
    for i in range(max_d + 1):
        n = result[d]
        if len(n) == 0:
            continue    
        print(f'{d}, {len(n)}, {sum(n)/len(n)}')

if __name__ == '__main__':
    __, __ = test_by_hand()
    # experiment()  #  run graph search 10000 times and report result.
