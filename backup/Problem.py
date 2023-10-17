from Node import Node

class Problem:
    def __init__(self, board):
        self.states = set()
        #Add init state to 
        self.initialState = self.arrayToChromosome(board)
        self.states.add(self.initialState)



    def arrayToChromosome(self,board):
        chromosome = ""
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    chromosome += "X"
                else:
                    chromosome += str(board[row][col])
        return chromosome
