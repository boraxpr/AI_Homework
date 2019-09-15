#Naipawat Poolsawat 6088120 Sec2
"""A module for homework 1."""
import random
import copy
class EightPuzzleState:
    """A class for a state of an 8-puzzle game."""

    def __init__(self, board):
        """Create an 8-puzzle state."""
        "Up,Down,Left,Right"
        self.action_space = {'u', 'd', 'l', 'r','INIT'}
        self.board = board
        for i, row in enumerate(self.board):
            for j, v in enumerate(row):
                if v == 0:
                    self.y = i
                    self.x = j
    
    
    def __repr__(self):
        """Return a string representation of a board."""
        output = []
        for row in self.board:
            row_string = ' | '.join([str(e) for e in row])
            output.append(row_string)
        return ('\n' + '-' * len(row_string) + '\n').join(output)

    def __str__(self):
        """Return a string representation of a board."""
        return self.__repr__()

    @staticmethod
    def initializeState():
        
        #Create a 2d list
        slot = [[0,1,2],[3,4,5],[6,7,8]]
        
        #Flatten the list
        flatslot = [y for x in slot for y in x]
        
        #random by shuffle the list
        random.shuffle(flatslot)
        
        #slice the list into 3 parts
        slot = [flatslot[i:i + 3] for i in range(0,len(flatslot),3)]
        
        """
        for testing
        print(slot[0][0])
        print(flatslot)
        print(slot)
        """
      
        #Return a state with a random.shuffled list
        init_state = EightPuzzleState(slot)
        
        
        return init_state
        """
        Create an 8-puzzle state with a SHUFFLED tiles.
        
        Return
        ----------
        EightPuzzleState
            A state that contain an 8-puzzle board with a type of List[List[int]]: 
            a nested list containing integers representing numbers on a board
            e.g., [[0, 1, 2], [3, 4, 5], [6, 7, 8]] where 0 is a blank tile.
        """
        # TODO: 1
        pass
    def successor(self, action):
        """
        Move a blank tile in the current state, and return a new state.

        Parameters
        ----------
        action:  string 
            Either 'u', 'd', 'l', or 'r'.

        Return
        ----------
        EightPuzzleState or None
            A resulting 8-puzzle state after performing `action`.
            If the action is not possible, this method will return None.

        Raises
        ----------
        ValueError
            if the `action` is not in the action space
        
        """    
        new_board = copy.deepcopy(self.board)
        
        #Find zero
        rows = [rows for rows in new_board if 0 in rows][0]
        #The row that has zero
        rzero = new_board.index(rows)
        #The column that has zero
        czero = rows.index(0)
        """
        #tester
        print("(",new_board.index(rows),",",rows.index(0),")")
        """
        if action == 'u':
            #If the action is not possible return none
            if rzero-1 < 0:
                return None
            #Swap the blank slot (zero) with the upper slot
            new_board[rzero][czero] ,new_board[rzero-1][czero] = new_board[rzero-1][czero], new_board[rzero][czero]
            self.board = new_board
            """ 
            for testing
            print(self.board)
            print(new_board[rzero][czero])
            """   
        if action == 'd':
            #If the action is not possible return none
            if rzero+1 > 2:
                return None
            #Swap the blank slot (zero) with the lower slot
            new_board[rzero][czero] ,new_board[rzero+1][czero] = new_board[rzero+1][czero], new_board[rzero][czero]
            self.board = new_board
            
        if action == 'l':
            #If the action is not possible return none
            if czero-1 < 0:
                return None
            #Swap the blank slot (zero) with the left slot
            new_board[rzero][czero] ,new_board[rzero][czero-1] = new_board[rzero][czero-1], new_board[rzero][czero]
            self.board = new_board
            
        if action == 'r':
            #If the action is not possible return none
            if czero+1 > 2:
                return None
            #Swap the blank slot (zero) with the right slot
            new_board[rzero][czero] ,new_board[rzero][czero+1] = new_board[rzero][czero+1], new_board[rzero][czero]
            self.board = new_board
        
        if action not in self.action_space:
            raise ValueError(f'`action`: {action} is not valid.')
        # TODO: 2
        # YOU NEED TO COPY A BOARD BEFORE MODIFYING IT
        # Thank you teacher
        return self


    def is_goal(self, goal_board=[[1, 2, 3], [4, 5, 6], [7, 8, 0]]):
        """
        Return True if the current state is a goal state.
        
        Parameters
        ----------
        goal_board (optional)
            The desired state of 8-puzzle.

        Return
        ----------
        Boolean
            True if the current state is a goal.
        
        """
        if self.board == goal_board:
            return True
        # TODO: 3
        pass


class EightPuzzleNode:
    """A class for a node in a search tree of 8-puzzle state."""
    
    def __init__(
            self, state, parent=None, action=None, cost=1):
        """Create a node with a state."""
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        if parent is not None:
            self.path_cost = parent.path_cost + self.cost
        else:
            self.path_cost = 0
            
    def trace(self):
        thisisalist = []
        """
        Return a path from the root to this node.

        Return
        ----------
        List[EightPuzzleNode]
            A list of nodes stating from the root node to the current node.
        try but fail
        node = []
        node.append(self)
        if self.state.successor(action) == 'q':
            return node
        self.trace
        """
        thisisalist.append(self)
  
        return thisisalist
       
        # TODO: 4

        pass


def test_by_hand():
    """Run a CLI 8-puzzle game."""
    state = EightPuzzleState.initializeState()
    root_node = EightPuzzleNode(state, action='INIT')
    cur_node = root_node
    print(state)
    action = input('Please enter the next move (q to quit): ')
    while action != 'q':
        thisisalist = []
        new_state = cur_node.state.successor(action)
        cur_node = EightPuzzleNode(new_state, cur_node, action)
        
        print(new_state)
        if new_state.is_goal():
            print('Congratuations!')
            break
        thisisalist.append(action)
        action = input('Please enter the next move (q to quit): ')
    
    print('Your actions are: ')
    for node in cur_node.trace():
        print(f'  - {node.action}')
    print(f'The total path cost is {cur_node.path_cost}')

if __name__ == '__main__':
    test_by_hand()
