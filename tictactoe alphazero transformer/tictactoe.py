import numpy as np

dimention = 3
emptyTable = np.zeros((dimention, dimention))

def printFormmating(boardState):
    for i in range(len(boardState)):
        print(boardState[i])

def rowChecker(boardState, dimention):
    for row in range(dimention):
        totalMultiplication = 1
        for column in range(dimention):
            totalMultiplication *= boardState[row][column]

        if totalMultiplication == 2**dimention: # Row full of twos
            return 2
        
        if totalMultiplication == 1**dimention: # Row full of ones
            return 1
        
    return -1

def columnChecker(boardState, dimention):
    for column in range(dimention):
        totalMultiplication = 1
        for row in range(dimention):
            totalMultiplication *= boardState[row][column]

        if totalMultiplication == 2**dimention: # Row full of twos
            return 2
        
        if totalMultiplication == 1**dimention: # Row full of ones
            return 1
        
    return -1


def diagonalChecker(boardState, dimention):
    for corner in range(1, 3):
        totalMultiplication = 1

        if corner == 1:
            for i in range(dimention):
                totalMultiplication *= boardState[i][i]

            
            if totalMultiplication == 2**dimention: # Row full of twos
                return 2
        
            if totalMultiplication == 1**dimention: # Row full of ones
                return 1
        
        if corner == 2:
            row = 0
            totalMultiplication = 1
            for column in range(dimention - 1, -1, -1):
                totalMultiplication *= boardState[row][column]
                row += 1
            
            if totalMultiplication == 2**dimention: # Row full of twos
                return 2
        
            if totalMultiplication == 1**dimention: # Row full of ones
                return 1
    
    return -1
                
            
def winningState(boardState, dimention):
    if rowChecker(boardState, dimention) != -1 or columnChecker(boardState, dimention) != -1 or diagonalChecker(boardState, dimention) != -1:
        return True
    
    return False

def whoWins(boardState, dimention):
    if rowChecker(boardState, dimention) == 1 or columnChecker(boardState, dimention) == 1 or diagonalChecker(boardState, dimention) == 1:
        return -1
    
    if rowChecker(boardState, dimention) == 2 or columnChecker(boardState, dimention) == 2 or diagonalChecker(boardState, dimention) == 2:
        return 1
    
    if fullBoard(boardState, dimention) == True:
        return 0
    
    return 3



def fullBoard(boardState, dimention):
    zeroCounter = 0
    for row in range(dimention):
        for column in range(dimention):
            if boardState[row][column] == 0:
                zeroCounter += 1
    
    if zeroCounter == 0:
        return True
    
    return False
