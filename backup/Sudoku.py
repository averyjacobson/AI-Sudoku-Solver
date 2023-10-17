#Imports
import csv
import os
from Board import Board
from Agent import Agent
import numpy as np

#File Declarations
difficulties = ["Easy", "Med", "Hard", "Evil"]
names = ["P1","P2","P3","P4","P5"]


#Splash
print ("Sudoku Solver\n___________________________\n")

#Read File
absolute_path = os.path.dirname(__file__)+"\\"

#Bypass method input
inputFile = absolute_path+"Med-P1.csv"
boardSeed = open(inputFile, 'r',)

#Initiate Board
board = Board(boardSeed)

#Display Board
# board.printBoard()

agent = Agent("S")

agent.method = "LSA"

agent.solve(board)







