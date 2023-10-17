import numpy as np

class Board:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, *args):
        lockValues = set()
        boardChromosome = np.zeros( (9, 9), dtype=int )

        if len(args) > 0:
            boardSeed = args[0]
            rowCount = 0
            colCount = 0
            boardSeed.read(3)
            index=0
            for row in boardSeed:
                row = row.strip()
                for value in row.split(","):
                    colCount = index%9
                    if value != "?":
                        boardChromosome[rowCount][colCount] = value
                        lockValues.add(index)
                    else:
                        boardChromosome[rowCount][colCount] = 0
                    index += 1
                rowCount += 1
            self.chromosome = boardChromosome
            #lockValues is indexes of uneditable values, csv.
            self.lockValues = lockValues
        else:
            self.chromosome = boardChromosome
            self.lockValues = lockValues

    # def __init__(self):
    #     lockValues = set()
    #     boardChromosome = np.zeros( (9, 9), dtype=int )
    #     self.chromosome = boardChromosome
    #     #lockValues is indexes of uneditable values, csv.
    #     self.lockValues = lockValues



    def printBoard(self):
        for row in self.chromosome:  
            for value in row:
                if value > 0:
                    print("[", end="")
                    print(value, end= "")
                    print("]", end = "")
                else:
                    print("[ ]", end="")
            print("")
    
    def isEditable(self, index):
        for value in self.lockValues:
            if (index[0]*9 + index[1]) == int(value) or index[0] > 8 or index[1] > 8:
                return False
        return True

    def setVal(self, index, value):
        if self.isEditable(index) and value <= 9:
            self.chromosome[index[0]][index[1]] = value
        elif value > 9:
            print(value,end="")
            print(" is not a Valid Value")
        else:
            print("[",end="")
            print(index[0], end = "")
            print("][", end="")
            print(index[1], end="")
            print("] is not an Editable Location!")

    def set(self, chromosome):
        self.chromosome = chromosome

    def get(self):

        return self.chromosome
        
            
                

    # def __repr__(self) -> str:
    #     return f"{type(self).__name__}(x={self.x}, y={self.y})"
