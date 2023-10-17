#Imports
import copy
import math
import math
import random
from Board import Board 

#Agent class
class Agent:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, method):
        self.method = method

    #Declarations
    decisions = 0

    def solve(self, board):
 
        #Simple Backtracking
        if self.method == 'B':
            self.backtrackingSearch(board.get(), 0, 0)
            if (self.boardVerifier(board.get())):
                print("The board is correct")
            return board
        
        #Backtracking with Forward Checking
        elif self.method == "BFC":
            self.backtrackingSearchFC(board.get(), 0, 0)
            if (self.boardVerifier(board.get())):
                print("The board is correct")
            return board
        
        #Backtracking with Arc Consistency
        elif self.method == "BAC":
            self.backtrack_with_arc_consistency(board.get(), 0,0)
            return board
        
        #Local Search Annealing
        elif self.method == "LSA":
            board = self.simulatedAnnealing(board)
            if self.boardVerifier(board.get()):
                print("The board is correct")
            else:
                print("LSA was unable to solve the board")
            return board
        
        #Local Search Using a Genetic Algorithm
        elif self.method == "LSGA":
            board = self.geneticAlgorithm(board)
            return board
    
    #----------------------------------------------------------------
    #--------------------------METHODS-------------------------------
    #----------------------------------------------------------------

    #Function to Check if the Chosen Number is Valid
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
                self.decisions += 1
                if self.backtrackingSearch(board, row, col + 1):
                    return True
            board[row][col]= 0
            self.decisions += 1
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
            return self.backtrackingSearchFC(board, row, col + 1)
        #Forward check
        remaining_values = self.forward_check(board,row, col)
        #Backtracking
        
        for num in remaining_values:
            if self.validMove(board,row, col, num):
                board[row][col] = num
                self.decisions += 1
                if self.backtrackingSearchFC(board,row,col):
                    return True
            board[row][col] = 0
            self.decisions += 1
        return False
    
    def forward_check(self, board, row, col):
        remaining_values = list(range(1, 10))
        #Row check
        for i in range(9):
            if board[row][i] in remaining_values:
                remaining_values.remove(board[row][i])
            #Column check
            if board[i][col] in remaining_values:
                remaining_values.remove(board[i][col])
        #Box check
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] in remaining_values:
                    remaining_values.remove(board[i][j])
        return remaining_values
    

    def revise(self, board, row, col, num):
        for i in range(9):
            
            if ((i != col) and (board[row][i] == num)):
                board[row][i] = 0
                print("Revising")

            if i != row and board[i][col] == num:
                board[i][col] = 0
                

        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if (i, j) != (row, col) and board[i][j] == num:
                    board[i][j] = 0
                 

    def backtrack_with_arc_consistency(self, board, row, col):
        #Check if end is reached
        if (row == 8 and col == 9 ):
                return True
        #Start next row
        if col == 9:
            row = row + 1
            col = 0
        #Check if value exists in cell
        if board[row][col] > 0:
            return self.backtrack_with_arc_consistency(board, row, col + 1)

        for num in range(1, 10):
            if self.validMove(board, row, col, num):
                board[row][col] = num
                self.decisions += 1
                self.revise(board,row, col, num)

                if self.backtrack_with_arc_consistency(board,row,col):
                    return True

                # If the current configuration didn't lead to a solution, backtrack
                board[row][col] = 0
                self.decisions += 1

        return False
    
    
    def simulatedAnnealing(self, board):
        #simulated annealing is inherently designed for finding a good solution but not necessarily the global optimum
        #Therefore, constantly find local optimum with around 8 constraint violations
        initState = copy.deepcopy(board)
        current = copy.deepcopy(board)

        #Begin with random board
        current = self.generateRandomState(current)
        # current.printBoard()
        # print("Constraint Count of CURRENT is: ",self.getConstraintCount(current.get()))

        #Set Parameters
        stuckCount = 0
        iterations = 70 #How many times a temperature is iterated
        temperature = 3 #how likely we are to accept randomness. HIGH: be random, LOW: be elitist
        coolingRate = 0.99 #How quickly randomness dwindles

        while not self.boardVerifier(current.get()):
            #Try n iterations with given temp
            previousScore = self.getConstraintCount(current.get())
            # print("Constraint Count of CURRENT is: ",previousScore)
            for i in range(0, iterations):
                #give neighbor a random swap within a chunk
                neighbor = self.getNeighborLSA(current)
                acceptanceProbability = self.calculateAcceptanceProbability(current.get(), neighbor.get(), temperature)
                if (random.random() <= acceptanceProbability):
                    current = neighbor
                self.decisions += 1
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
        #return board
        return current



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

        # tempBoard.set(chromosome)
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

    #
    def boardVerifier(self, board):
        if any(0 in row for row in board):
            print("Board is incomplete")
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


    def bubbleSort(self,arr):
        n = len(arr)
        swapped = False
        for i in range(n-1):
            for j in range(0, n-i-1):
                if arr[j].fitness > arr[j + 1].fitness:
                    swapped = True
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
            if not swapped:
                return


    #Tune Menu for GA
    populationSize = 20 # 20
    tournamentSize = math.floor(.20*populationSize) # 20%
    tournamentSelected = math.floor(.05*populationSize) # 5%
    parentsAllowed = math.floor(.70*populationSize) # 70%
    mutations = math.floor(.80*populationSize) # 80%
    generationLimit = 50 #50
    mutationRate = math.floor(.80*populationSize) # 80%
    crossOverAmount = 2 #2
    keepRate = 0 #0

    stepThreshold = 0
    mutationsStep = 5 #5
    mutationRateStep = 49 #49
    keepRateStep = 1 #1

    #Declarations
    iteration = 0
    boards = []
    new_generation = []

    #Solve using GA
    def geneticAlgorithm(self, board):
        
        #Initiate Population 
        for i in range(1,self.populationSize):
            newBoard = copy.deepcopy(board)
            newBoard = self.generateRandomState(newBoard)
            self.boards.append(newBoard)
            newBoard.fitness = self.getConstraintCount(newBoard.get())

        #While Below the Generation Limit
        while(self.iteration < self.generationLimit):    
            
            #Update Fitness
            for c in self.boards:
                c.fitness = self.getConstraintCount(c.get())
            
            #Keep Most fit Canadites if Applicable
            if (self.keepRate > 0):
                #Sort boards
                self.bubbleSort(self.boards)
                for k in range (self.keepRate):
                    self.new_generation.append(self.boards[k])

            #Select New Generation with Tournament
            while (len(self.new_generation) < self.parentsAllowed):
                addGen = self.tournamentSelection(self.boards)
                for j in range(len(addGen)):
                        self.new_generation.append(addGen[j])
                
            #Replace Generation
            self.boards = []
            self.boards = self.new_generation
            self.new_generation = []
        
            #Check for Valid Board
            for x in self.boards:
                #print (x.fitness) Enable to live view fitness 
                if (x.fitness == 0) :     
                    print (x.get())
                    return x
                
            #Repopulate with crossover
            while (len(self.boards) < self.populationSize):
                for i in range(1,(math.floor((self.populationSize-2)/2))):
                    addGen = self.recombine(self.boards[i],self.boards[i + 1])
                    for j in range(1):
                        self.boards.append(addGen[j])
                    addGen = []

            #Evalute fitness
            for a in self.boards:
                a.fitness = self.getConstraintCount(a.get())

            #Check for Valid Board
            for x in self.boards:
                #print (x.fitness) Enable to live view fitness 
                if (x.fitness < self.stepThreshold):
                    self.mutations = self.mutationsStep
                    self.mutationRate = self.mutationRateStep
                    self.keepRate = self.keepRateStep
                
                if (x.fitness == 0) :
                    print (x.get())
                    return x
                                       
            #Mutate 
            for y in range (self.mutationRate):
                z = random.randint(0,self.populationSize - 1)
                self.boards[z] = self.mutation(self.boards[z])

            self.iteration += 1

        #If Generation Limit Reached Reset
        self.boards = []
        self.new_generation = []
        self.iteration = 0
        self.geneticAlgorithm(board)

    #Tournament Selection
    def tournamentSelection(self,population):
        
        selected = []
        tournament = []
        
        #Grab Contestants 
        for i in range(0, self.tournamentSize):
            tournament.append(population[random.randint(0,self.populationSize - 2)])
        
        #Sort by Fitness
        self.bubbleSort(tournament)
        
        #Select
        for i in range (0, self.tournamentSelected):
            selected.append(tournament[i])
        return selected

    #Crossover
    def recombine(self, board1, board2):
        
        #Initiate Children
        board3 = copy.deepcopy(board1)
        board4 = copy.deepcopy(board2)
        childBoards = []
        
        #Select Col and Swap
        for x in range(self.crossOverAmount):
            mutationCol = random.randint(1,8)
            for i in range(0,9):
                if board1.isEditable([i,mutationCol]):
                    board3.get()[i,mutationCol] = board2.get()[i,mutationCol]
                    board4.get()[i,mutationCol] = board1.get()[i,mutationCol]

        #Append to list and return
        childBoards.append(board3)
        childBoards.append(board4)
        return childBoards
    
    #Mutate
    def mutation(self, board):
        loop = 0
        listOfConstraints = self.returnConstraints(board)

        #Find Constraints and Change
        for x in listOfConstraints:
            
            #Random Mutate Option
            randomMutate = False        
            if (random.randint(1,10000) == 1):
                randomMutate = False

            #Normal Mutation
            if (not randomMutate):
                if (loop < self.mutations):     
                    a = x[0]
                    b = x[1]
                    board.get()[a][b] = (random.randint(1,9))
                    loop += 1
            
            #Random Mutation
            else:
                while (loop < self.mutations):
                    x  = random.randint(0,8)
                    y= random.randint(0,8)
                    if board.isEditable([x,y]):
                        board.get()[x][y] = (random.randint(1,9))
                        loop += 1
        return board
    

    #Find Constraints
    def returnConstraints(self, inputBoard):
        checked = set()
        constraints = []
        inputBoard = inputBoard.get()
        rowIndex = 0
        colIndex = 0
        
        #check Rows
        for row in inputBoard:
            for value in row:
                if (checked.__contains__(value)):
                    
                    constraints.append([rowIndex,colIndex])
                colIndex += 1

                checked.add(value)
            colIndex = 0
            checked.clear()
            rowIndex += 1
        #check Columns
      
        rowIndex = 0
        colIndex = 0
        for col in range(0,8):
            for row in range (0,8):
                if (checked.__contains__(inputBoard[row][col])):
                    
                    constraints.append([rowIndex,colIndex])
                checked.add(inputBoard[row][col])
                rowIndex += 1
            
            rowIndex = 0
                
            checked.clear()
            colIndex += 1

        
        #check Chunk 
        for chunk in range(9):
            for col in range(chunk%3*3, (chunk%3)*3 + 3):
                for row in range(int(chunk/3)*3, (int(chunk/3) +1) * 3):
                    if (checked.__contains__(inputBoard[row][col])):
                        
                        constraints.append([row,col])
                    checked.add(inputBoard[row][col])
            checked.clear()
        return constraints
            

            

