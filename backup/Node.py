class Node:

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
    
    def __init__(self, pos):
        self.position = pos
        self.neighbors = self.getNeighbors(pos)
        



    def getNeighbors(self, pos): #neighbors are all nodes in 3x3 grid, as well as all nodes in current row and col
        neighbors = set(tuple())
        row = pos[0]
        col = pos[1]
        #need more here
        if row -1 > -1:
            neighbors.add((row - 1, col))
        if row + 1 < 9:
            neighbors.add((row + 1, col))
        if col - 1 > -1:
            neighbors.add((row, col -1))
        if col + 1 < 9:
            neighbors.add((row, col +1))
        return neighbors

        







        