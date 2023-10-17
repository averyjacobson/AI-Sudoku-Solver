#Imports
import csv
import os
from Board import Board
from Agent import Agent
import numpy as np
import plotly.express as px
import pandas as pd

#File Declarations
difficulties = ["Easy", "Med", "Hard", "Evil"]
names = ["P1","P2","P3","P4","P5"]
methods = ["LSGA","B","BFC","BAC","LSA"]

#Splash
print ("Sudoku Solver\n___________________________\n")
boards = []
decisions = []

#Read File
absolute_path = os.path.dirname(__file__)+"\\"

#Bypass method input
inputFile = absolute_path+"Med-P1.csv"
boardSeed = open(inputFile, 'r',)

#Initiate Board
board = Board(boardSeed)


#Try other methods
for k in methods:
    if k == "B":
        difficulties = ["Easy","Med"]
    else:
        difficulties = ["Easy", "Med", "Hard", "Evil"]
    for i in difficulties:
        for j in names:
            #Initiate board
            inputFile = absolute_path + i + "-" + j + ".csv"
            boardSeed = open(inputFile, 'r',)
            board = Board(boardSeed)
            #board.printBoard()
            agent = Agent ("S")
            agent.method = k
            agent.solve(board)
            #board.printBoard()
            boards.append("Board:" + i + j)
            decisions.append (agent.decisions)
            print(i,j,"solved by",k)


    #Print graph
    data = pd.DataFrame({'Board': boards, 'Decisions Made': decisions})
    fig = px.bar(data, x='Board', y='Decisions Made', title= (k + "Method"))
    fig.update_layout(
        xaxis_title='Board',
        yaxis_title='Decisions Made',
        xaxis_tickangle=-45,  
        yaxis_type='log',   
        yaxis_tickvals=[1, 10, 100, 1000, 10000, 100000, 1000000],
        yaxis_ticktext=["1", "10", "100", "1K", "10K", "100K", "1M"]  
    )
    fig.show()
    #Clear x and y data
    boards = []
    decisions = []
    #Print graph
 