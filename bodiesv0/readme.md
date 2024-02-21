# bodies

right now im just trying out things. currently transformers with tic tac toe to see if it can learn. next is transformers with mcts. also this is rapid prototyping of ideas so far so im functioning more as a prompt engineer.

```console
Game 1: AI (Player 1) vs Random (Player 2)

Current Board:
[0. 0. 0.]
[0. 0. 0.]
[0. 0. 0.]

Current Board:
[0. 0. 0.]
[0. 1. 0.]
[0. 0. 0.]

Current Board:
[0. 2. 0.]
[0. 1. 0.]
[0. 0. 0.]

Current Board:
[0. 2. 1.]
[0. 1. 0.]
[0. 0. 0.]

Current Board:
[0. 2. 1.]
[0. 1. 0.]
[0. 0. 2.]

Current Board:
[1. 2. 1.]
[0. 1. 0.]
[0. 0. 2.]

Current Board:
[1. 2. 1.]
[0. 1. 0.]
[0. 2. 2.]

Final Board:
[1. 2. 1.]
[0. 1. 0.]
[1. 2. 2.]

Game 2: AI (Player 1) vs Random (Player 2)

Current Board:
[0. 0. 0.]
[0. 0. 0.]
[0. 0. 0.]

Current Board:
[0. 0. 0.]
[0. 0. 1.]
[0. 0. 0.]

Current Board:
[0. 0. 0.]
[0. 0. 1.]
[2. 0. 0.]

Current Board:
[0. 0. 1.]
[0. 0. 1.]
[2. 0. 0.]

Current Board:
[0. 0. 1.]
[0. 0. 1.]
[2. 2. 0.]

Current Board:
[0. 0. 1.]
[0. 1. 1.]
[2. 2. 0.]

Current Board:
[2. 0. 1.]
[0. 1. 1.]
[2. 2. 0.]

Final Board:
[2. 0. 1.]
[0. 1. 1.]
[2. 2. 1.]

Game 3: Random (Player 1) vs AI (Player 2)

Current Board:
[0. 0. 0.]
[0. 0. 0.]
[0. 0. 0.]

Current Board:
[1. 0. 0.]
[0. 0. 0.]
[0. 0. 0.]

Current Board:
[1. 0. 0.]
[2. 0. 0.]
[0. 0. 0.]

Current Board:
[1. 0. 1.]
[2. 0. 0.]
[0. 0. 0.]

Current Board:
[1. 2. 1.]
[2. 0. 0.]
[0. 0. 0.]

Current Board:
[1. 2. 1.]
[2. 0. 1.]
[0. 0. 0.]

Current Board:
[1. 2. 1.]
[2. 0. 1.]
[0. 0. 2.]

Current Board:
[1. 2. 1.]
[2. 0. 1.]
[0. 1. 2.]

Current Board:
[1. 2. 1.]
[2. 0. 1.]
[2. 1. 2.]

Final Board:
[1. 2. 1.]
[2. 1. 1.]
[2. 1. 2.]

Game 4: Random (Player 1) vs AI (Player 2)

Current Board:
[0. 0. 0.]
[0. 0. 0.]
[0. 0. 0.]

Current Board:
[0. 0. 0.]
[1. 0. 0.]
[0. 0. 0.]

Current Board:
[0. 0. 0.]
[1. 2. 0.]
[0. 0. 0.]

Current Board:
[0. 0. 1.]
[1. 2. 0.]
[0. 0. 0.]

Current Board:
[2. 0. 1.]
[1. 2. 0.]
[0. 0. 0.]

Current Board:
[2. 0. 1.]
[1. 2. 1.]
[0. 0. 0.]

Final Board:
[2. 0. 1.]
[1. 2. 1.]
[0. 0. 2.]
andrewgordienko@Andrews-MacBook-Pro bodies v1 % 
```
ai vs ai
```console
ame1

Current Board:
[0. 0. 0.]
[0. 0. 0.]
[0. 0. 0.]

Move Heat Map:
[[-0.07  0.06 -0.99]
 [ 0.56 -0.44  0.24]
 [ 0.8  -0.29  0.23]]

AI Player 1 making move...

Current Board:
[0. 1. 0.]
[0. 0. 0.]
[0. 0. 0.]

Move Heat Map:
[[-0.15  0.01 -1.01]
 [ 0.51 -0.34  0.24]
 [ 0.64 -0.16  0.21]]

AI Player 2 making move...

Current Board:
[0. 1. 0.]
[0. 2. 0.]
[0. 0. 0.]

Move Heat Map:
[[-0.14 -0.   -1.06]
 [ 0.28 -0.45 -0.01]
 [ 0.7  -0.23 -0.02]]

AI Player 1 making move...

Current Board:
[0. 1. 0.]
[0. 2. 0.]
[1. 0. 0.]

Move Heat Map:
[[-0.3   0.   -1.06]
 [ 0.31 -0.41  0.01]
 [ 0.58 -0.26  0.08]]

AI Player 2 making move...

Current Board:
[2. 1. 0.]
[0. 2. 0.]
[1. 0. 0.]

Move Heat Map:
[[-0.22  0.2  -0.93]
 [ 0.4  -0.35  0.16]
 [ 0.53 -0.26  0.18]]

AI Player 1 making move...

Current Board:
[2. 1. 0.]
[1. 2. 0.]
[1. 0. 0.]

Move Heat Map:
[[-0.16  0.14 -0.96]
 [ 0.4  -0.32  0.16]
 [ 0.61 -0.13  0.16]]

AI Player 2 making move...

Current Board:
[2. 1. 0.]
[1. 2. 0.]
[1. 2. 0.]

Move Heat Map:
[[-0.11  0.08 -0.91]
 [ 0.49 -0.36  0.16]
 [ 0.63 -0.09  0.11]]

AI Player 1 making move...

Current Board:
[2. 1. 1.]
[1. 2. 0.]
[1. 2. 0.]

Move Heat Map:
[[-0.16  0.03 -0.99]
 [ 0.42 -0.4   0.09]
 [ 0.63 -0.15  0.21]]

AI Player 2 making move...

Current Board:
[2. 1. 1.]
[1. 2. 2.]
[1. 2. 0.]

Move Heat Map:
[[-0.21  0.01 -0.97]
 [ 0.34 -0.46  0.06]
 [ 0.64 -0.2   0.09]]

AI Player 1 making move...

Final Board:
[2. 1. 1.]
[1. 2. 2.]
[1. 2. 1.]

It's a tie!
game2

Current Board:
[0. 0. 0.]
[0. 0. 0.]
[0. 0. 0.]

Move Heat Map:
[[-0.23 -0.1  -1.1 ]
 [ 0.29 -0.52  0.11]
 [ 0.83 -0.25  0.16]]

AI Player 1 making move...

Current Board:
[0. 0. 0.]
[0. 1. 0.]
[0. 0. 0.]

Move Heat Map:
[[-0.16  0.04 -1.05]
 [ 0.4  -0.5   0.05]
 [ 0.67 -0.16  0.13]]

AI Player 2 making move...

Current Board:
[0. 0. 0.]
[0. 1. 2.]
[0. 0. 0.]

Move Heat Map:
[[-0.22  0.03 -0.94]
 [ 0.46 -0.32  0.24]
 [ 0.8  -0.33 -0.03]]

AI Player 1 making move...

Current Board:
[0. 1. 0.]
[0. 1. 2.]
[0. 0. 0.]

Move Heat Map:
[[-0.19  0.11 -0.93]
 [ 0.37 -0.41  0.15]
 [ 0.66 -0.13  0.08]]

AI Player 2 making move...

Current Board:
[0. 1. 0.]
[0. 1. 2.]
[2. 0. 0.]

Move Heat Map:
[[-0.2  -0.   -0.94]
 [ 0.57 -0.34  0.32]
 [ 0.79 -0.27  0.14]]

AI Player 1 making move...

Current Board:
[0. 1. 1.]
[0. 1. 2.]
[2. 0. 0.]

Move Heat Map:
[[-0.29  0.22 -0.66]
 [ 0.4  -0.27  0.23]
 [ 0.47 -0.15  0.01]]

AI Player 2 making move...

Current Board:
[0. 1. 1.]
[0. 1. 2.]
[2. 0. 2.]

Move Heat Map:
[[-0.3   0.09 -0.85]
 [ 0.44 -0.33  0.22]
 [ 0.68 -0.27  0.09]]

AI Player 1 making move...

Current Board:
[0. 1. 1.]
[1. 1. 2.]
[2. 0. 2.]

Move Heat Map:
[[-0.21 -0.01 -1.18]
 [ 0.33 -0.5   0.03]
 [ 0.71 -0.25  0.17]]

AI Player 2 making move...

Current Board:
[2. 1. 1.]
[1. 1. 2.]
[2. 0. 2.]

Move Heat Map:
[[-0.25 -0.01 -1.1 ]
 [ 0.37 -0.46  0.15]
 [ 0.79 -0.28  0.2 ]]

AI Player 1 making move...

Final Board:
[2. 1. 1.]
[1. 1. 2.]
[2. 1. 2.]

Player 1 (AI) Wins!
game3

Current Board:
[0. 0. 0.]
[0. 0. 0.]
[0. 0. 0.]

Move Heat Map:
[[-0.21  0.03 -0.98]
 [ 0.47 -0.46  0.2 ]
 [ 0.7  -0.16  0.16]]

AI Player 1 making move...

Current Board:
[0. 1. 0.]
[0. 0. 0.]
[0. 0. 0.]

Move Heat Map:
[[-0.14  0.04 -1.07]
 [ 0.24 -0.53  0.05]
 [ 0.74 -0.14  0.17]]

AI Player 2 making move...

Current Board:
[0. 1. 0.]
[2. 0. 0.]
[0. 0. 0.]

Move Heat Map:
[[-0.17  0.14 -1.01]
 [ 0.27 -0.43  0.12]
 [ 0.62 -0.36  0.11]]

AI Player 1 making move...

Current Board:
[0. 1. 0.]
[2. 0. 0.]
[1. 0. 0.]

Move Heat Map:
[[-0.31  0.05 -1.  ]
 [ 0.39 -0.42  0.04]
 [ 0.56 -0.2   0.04]]

AI Player 2 making move...

Current Board:
[0. 1. 2.]
[2. 0. 0.]
[1. 0. 0.]

Move Heat Map:
[[-0.26  0.09 -0.88]
 [ 0.47 -0.4   0.19]
 [ 0.67 -0.16  0.19]]

AI Player 1 making move...

Current Board:
[0. 1. 2.]
[2. 0. 1.]
[1. 0. 0.]

Move Heat Map:
[[-0.25  0.03 -1.1 ]
 [ 0.42 -0.37  0.13]
 [ 0.64 -0.24  0.05]]

AI Player 2 making move...

Current Board:
[2. 1. 2.]
[2. 0. 1.]
[1. 0. 0.]

Move Heat Map:
[[-0.25  0.12 -0.9 ]
 [ 0.38 -0.39  0.23]
 [ 0.53 -0.19  0.07]]

AI Player 1 making move...

Current Board:
[2. 1. 2.]
[2. 0. 1.]
[1. 0. 1.]

Move Heat Map:
[[-0.21  0.05 -1.05]
 [ 0.48 -0.34  0.2 ]
 [ 0.73 -0.32  0.09]]

AI Player 2 making move...

Current Board:
[2. 1. 2.]
[2. 2. 1.]
[1. 0. 1.]

Move Heat Map:
[[-0.06  0.24 -0.8 ]
 [ 0.44 -0.32  0.21]
 [ 0.55  0.01  0.19]]

AI Player 1 making move...

Final Board:
[2. 1. 2.]
[2. 2. 1.]
[1. 1. 1.]

Player 1 (AI) Wins!
```
