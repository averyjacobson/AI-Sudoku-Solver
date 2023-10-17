import copy
import math
import numpy as np
from Node import Node 
from Board import Board
from Problem import Problem
import random


class Agent:
    global swaps
    swaps = set()
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, method):
        self.method = method

    def solve(self, board):
        if (self.boardVerifier(board.get())):
            print("The board is correct")
            return board.get()
        #Simple backTracking
        elif self.method == 'B':
            print("Solving Using Simple Backtracking:")
            self.backtrackingSearch(board.get(), 0, 0)
            if (self.boardVerifier(board.get())):
                print("The board is correct")
            return board.get()
        elif self.method == "BFC":
            print("Solving Using Backtracking with Forward Checking:")
            self.backtrackingSearchFC(board.get(), 0, 0)
            if (self.boardVerifier(board.get())):
                print("The board is correct")
            return board
        elif self.method == "BAC":
            #do backtracking Arc consistency
            print("Solving Using Backtracking with Arc Consistency:")
            return board.get()
        elif self.method == "LSA":
            #do local search annealing
            print("Solving Using Local Search Annealing")
            #problem = Problem(board)
            self.simulatedAnnealing(board)
            return board
        elif self.method == "LSGA":
            #populations 20-50 should be sufficient
            #do local search genetic algorithm
            print("Solving Using Local Search Genetic Algorithm")
            return board.get

#-----------------------------------------------------------------------------------------------
#---------------------------------SEARCH FUNCTIONS----------------------------------------------
#-----------------------------------------------------------------------------------------------


    def backtrackingSearch(self, board, row, col):
        #Check if end is reached
        if (row == 8 and col == 9 ):
                return True
        #Start next row
        if col == 9:
            row = row + 1
            col = 0
        #Check if value exists in cell
        if board[row][col] > 0:
            return self.backtrackingSearch(board, row, col + 1)
        #Start backtracking
        for num in range(1, 10, 1):
            if self.validMove(board,row, col, num):
                board[row][col] = num
                if self.backtrackingSearch(board, row, col + 1):
                    return True
            board[row][col]= 0
        return False
    
    def backtrackingSearchFC(self, board, row, col):
        #Check if end is reached
        if (row == 8 and col == 9 ):
                return True
        #Start next row
        if col == 9:
            row = row + 1
            col = 0
        #Check if value exists in cell
        if board[row][col] > 0:
            return self.backtrackingSearch(board, row, col + 1)
        #Start backtracking
        for num in range(1, 10, 1):
            if self.validMove(board,row, col, num):
                board[row][col] = num
                if self.backtrackingSearch(board, row, col + 1):
                    return True
            board[row][col]= 0
        return False
    



    def simulatedAnnealing(self, board):
        #simulated annealing is inherently designed for finding a good solution but not necessarily the global optimum
        #Therefore, constantly find local optimum with around 8 constraint violations
        initState = copy.deepcopy(board)
        current = copy.deepcopy(board)

        #Begin with random board
        current = self.generateRandomState(current)
        current.printBoard()
        print("Constraint Count of CURRENT is: ",self.getConstraintCount(current.get()))

        #Set Parameters
        stuckCount = 0
        iterations = 70 #How many times a temperature is iterated
        temperature = 3 #how likely we are to accept randomness. HIGH: be random, LOW: be elitist
        coolingRate = 0.99 #How quickly randomness dwindles

        while not self.boardVerifier(current.get()):
            #Try n iterations with given temp
            previousScore = self.getConstraintCount(current.get())
            print("Constraint Count of CURRENT is: ",previousScore)
            for i in range(0, iterations):
                #give neighbor a random swap within a chunk
                neighbor = self.getNeighborLSA(current)
                acceptanceProbability = self.calculateAcceptanceProbability(current.get(), neighbor.get(), temperature)
                if (random.random() <= acceptanceProbability):
                    current = neighbor
            #If we stuck after 16 loops, abort and take answer
            if previousScore == self.getConstraintCount(current.get()):
                stuckCount += 1
                if stuckCount == 100:
                    print("Local Optimum Found at:", previousScore,"constraints")
                    break
            else:
                stuckCount = 0
            #decrement the temp using coolingRate
            temperature *= coolingRate
            initState.printBoard()
            print("-----------------------------------")
            current.printBoard()
        #set board to solution
        board = current





    





    
#-----------------------------------------------------------------------------------------------
#---------------------------------HELPER FUNCTIONS----------------------------------------------
#-----------------------------------------------------------------------------------------------


    def boardVerifier(self, board):
        if any(0 in row for row in board):
            return False
        checked = set()
        #check Rows
        for row in board:
            for value in row:
                if (checked.__contains__(value)):
                    return False
                checked.add(value)
            checked.clear()
        #check Columns
        for col in range(0,8):
            for row in range (0,8):
                if (checked.__contains__(board[row][col])):
                    return False
                checked.add(board[row][col])
            checked.clear()
        #check Chunk
        for chunk in range(9):
            for col in range(chunk%3*3, (chunk%3)*3 + 3):
                for row in range(int(chunk/3)*3, (int(chunk/3) +1) * 3):
                    if (checked.__contains__(board[row][col])):
                        return False
                    checked.add(board[row][col])
            checked.clear()
        return True

        

    def validMove(self, board, row, col, num):
        for x in range(9):
            if board[row][x] == num:
                return False 
        for x in range(9):
            if board[x][col] == num:
                return False
        startRow = row - row % 3
        startCol = col - col % 3
        for i in range(3):
            for j in range(3):
                if board[i + startRow][j + startCol] == num:
                    return False        
        return True
    
    #returns board with random state
    def generateRandomState(self, inputBoard):
        #Use unique values in each chunk, don't worry about rows/columns
        tempBoard = inputBoard
        chromosome = tempBoard.get()
        chunkVals = set()
        remainingVals = list()

        for chunk in range(9):
            #populate remainingVals
            for val in range(1,10):
                remainingVals.insert(1, val)
            
            #Get current values in chunk
            for col in range(chunk%3*3, (chunk%3)*3 + 3):
                for row in range(int(chunk/3)*3, (int(chunk/3) +1) * 3):
                    if chromosome[row][col] > 0:
                        chunkVals.add(chromosome[row][col])
                        remainingVals.remove(chromosome[row][col])

            #Assign Unique Vals to editable areas in chunk
            for col in range(chunk%3*3, (chunk%3)*3 + 3):
                for row in range(int(chunk/3)*3, (int(chunk/3) +1) * 3):
                    if chromosome[row][col] == 0:
                        chromosome[row][col] = random.choice(remainingVals)
                        remainingVals.remove(chromosome[row][col])
                        chunkVals.add(chromosome[row][col])
          
            chunkVals.clear()

        tempBoard.set(chromosome)
        return tempBoard
    

    #provides numerical value for broken constraints in puzzle
    def getConstraintCount(self, inputBoard):
        checked = set()
        constraintViolationCount = 0
        #check Rows
        for row in inputBoard:
            for value in row:
                if (checked.__contains__(value)):
                    constraintViolationCount += 1
                checked.add(value)
            checked.clear()
        #check Columns
        for col in range(0,8):
            for row in range (0,8):
                if (checked.__contains__(inputBoard[row][col])):
                    constraintViolationCount += 1
                checked.add(inputBoard[row][col])
            checked.clear()
        #check Chunk 
        for chunk in range(9):
            for col in range(chunk%3*3, (chunk%3)*3 + 3):
                for row in range(int(chunk/3)*3, (int(chunk/3) +1) * 3):
                    if (checked.__contains__(inputBoard[row][col])):
                        constraintViolationCount += 1
                    checked.add(inputBoard[row][col])
            checked.clear()
        return constraintViolationCount
    
    def getNeighborLSA(self, inputBoard):
        neighbor = Board()
        neighbor = copy.deepcopy(inputBoard)
        chunk = random.randint(0,8)
        
        #Pick random two values in chunk
        indexA = (random.randint(int(chunk/3)*3, (int(chunk/3) +1) * 3 -1), random.randint(chunk%3*3, (chunk%3)*3 + 2))
        indexB = (random.randint(int(chunk/3)*3, (int(chunk/3) +1) * 3 -1), random.randint(chunk%3*3, (chunk%3)*3 + 2))

        #make sure they are editable values
        while (indexA[0]*9 + indexA[1]) in neighbor.lockValues:
            indexA = (random.randint(int(chunk/3)*3, (int(chunk/3) +1) * 3 -1), random.randint(chunk%3*3, (chunk%3)*3 + 2))
        while (indexB[0]*9 + indexB[1]) in neighbor.lockValues:
            indexB = (random.randint(int(chunk/3)*3, (int(chunk/3) +1) * 3-1), random.randint(chunk%3*3, (chunk%3)*3 + 2))
        
        swaps.add(indexA)
        swaps.add(indexB)

        #store value of index A
        temp = neighbor.get()[indexA[0]][indexA[1]] 
        #put indexB value at indexA
        neighbor.get()[indexA[0]][indexA[1]] = neighbor.get()[indexB[0]][indexB[1]]
        #put indexA value at indexB
        neighbor.get()[indexB[0]][indexB[1]] = temp
        return neighbor
    
    def calculateAcceptanceProbability(self, current, neighbor, temperature):  #  1/(1+exp(delta/T))
        delta = self.getConstraintCount(neighbor) - self.getConstraintCount(current) # current and neighbor same thing?!?!?!?!?!?!?!?!?!?!?!
        rho = math.exp(-delta/temperature)
        return rho
    
    #helper to verify what positions have been swapped graphically
    def showSwaps(self):
        statBoard = Board()
        statBoard.chromosome = np.ones( (9, 9), dtype=int )
        for index in swaps:
            statBoard.chromosome[index[0]][index[1]] = 0
        statBoard.printBoard()





        

