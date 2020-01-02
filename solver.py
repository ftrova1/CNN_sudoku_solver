
# board = [
#             [0,0,0,2,6,0,7,0,1],
#             [6,8,0,0,7,0,0,9,0],
#             [1,9,0,0,0,4,5,0,0],
#             [8,2,0,1,0,0,0,4,0],
#             [0,0,4,6,0,2,9,0,0],
#             [0,5,0,0,0,3,0,2,8],
#             [0,0,9,3,0,0,0,7,4],
#             [0,4,0,0,5,0,0,3,6],
#             [7,0,3,0,1,8,0,0,0]
#         ]

# board = [[3,0,6,5,0,8,4,0,0],
#           [5,2,0,0,0,0,0,0,0],
#           [0,8,7,0,0,0,0,3,1],
#           [0,0,3,0,1,0,0,8,0],
#           [9,0,0,8,6,3,0,0,5],
#           [0,5,0,0,9,0,6,0,0],
#           [1,3,0,0,0,0,2,5,0],
#           [0,0,0,0,0,0,0,7,4],
#           [0,0,5,2,0,6,3,0,0]]

board = [[0,0,4,0,0,9,2,0,1],
          [6,0,0,0,8,7,0,0,4],
          [0,0,0,2,0,0,6,0,0],
          [2,0,0,5,0,0,0,0,0],
          [8,0,0,0,1,0,0,0,9],
          [0,0,0,0,0,2,0,0,5],
          [0,0,2,0,0,5,0,0,0],
          [4,0,0,3,9,0,0,0,2],
          [9,0,1,7,0,0,3,0,0]]

def find_empty_spot (board, pos):

    for row in range (0,len(board)):
        for col in range (0, len(board[row])):
            if board[row][col] == 0:
                pos[0] = row
                pos[1] = col
                return True
    return False

def is_valid (board,pos,n):
    row = pos[0]
    col = pos[1]

    if (is_valid_in_row(board,row,n) and is_valid_in_col(board,col,n) and is_valid_in_square(board,row - row%3,col - col%3,n)):
        return True
    return False


def is_valid_in_row(board, row, n):
    for col in range(0,9):
        if board[row][col] == n:
            return False
    return True

def is_valid_in_col(board, col, n):
    for row in range(0,9):
        if board[row][col] == n:
            return False
    return True

def is_valid_in_square(board, row, col, n):
    for i in range(3):
        for j in range(3):
            if(board[i+row][j+col] == n):
                return False
    return True

def solve (board):
    pos = [0,0]
    if not(find_empty_spot(board,pos)):
        #Board solved
        return True

    row = pos[0]
    #print(row)
    col = pos[1]
    #print (col)
    for n in range (1,10):
        if is_valid(board, pos, n):
            board[row][col] = n

            if solve(board):
                return True

            board[row][col] = 0
    #Backtracking
    return False


def print_board(board):
    for i in range (0,len(board)):
        print("")
        if (i%3 == 0 and i!= 0) :
            print("______________________")
        for j in range (0,len(board[i])):
            if (j%3 == 0 and j!= 0):
                print(" | ", end="")
            print (str(board[i][j]) + " " , end="")
    print("\n")



print_board(board)
print("Sto risolvendo......\n")
if (solve(board)):
    print("Risolto!")
    print_board(board)
else:
    print("Soluzione non trovata.")
