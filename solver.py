board = [
            [0,0,0,2,6,0,7,0,1],
            [6,8,0,0,7,0,0,9,0],
            [1,9,0,0,0,4,5,0,0],
            [8,2,0,1,0,0,0,4,0],
            [0,0,4,6,0,2,9,0,0],
            [0,5,0,0,0,3,0,2,8],
            [0,0,9,3,0,0,0,7,4],
            [0,4,0,0,5,0,0,3,6],
            [7,0,3,0,1,8,0,0,0]
        ]
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
