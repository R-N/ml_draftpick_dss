import numpy as np
from ..Game import Game as _Game
from .Logic import DraftingBoard
from itertools import permutations, product


def onehot_to_label(arr):
    arr = np.array(arr)
    if len(arr.shape) > 2:
        return np.array([onehot_to_label(b) for b in arr])
    return np.where(arr)[-1]

class DraftingGame(_Game):
    """
    Use 1 for player1 and -1 for player2.
    """
    def __init__(self, n=5):
        self.board = DraftingBoard()
        self.actionSize = len(self.board.double_possible_moves)

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return DraftingBoard().get_board()

    def getActionSize(self):
        """
        Returns:
            actionSize: length of action vector
        """
        #return DraftingBoard().get_double_legal_moves()
        #return 2*120
        #return 120
        return self.actionSize

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        board = DraftingBoard().load_board(board)
        board.execute_move(action, player=player)
        return board.get_board(), board.get_round()[-1]

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        board = DraftingBoard().load_board(board)
        legal_moves = board.get_double_legal_moves(player)
        print(len(legal_moves), type(legal_moves), type(legal_moves[0]), len(legal_moves[0]), type(legal_moves[0][0]), [sum(x) for x in legal_moves])
        print(len(board.double_possible_moves), type(board.double_possible_moves), len(board.double_possible_moves[0]), type(board.double_possible_moves[0]), type(board.double_possible_moves[0][0]), [sum(x) for x in board.double_possible_moves])
        legal_moves = [1 if m in legal_moves else 0 for m in board.double_possible_moves]
        print(sum(legal_moves))
        return np.array(legal_moves)
    
    def predict_left_win(self, left, right, threshold=0.5):
        left_win = 1
        return left_win > threshold

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        board = DraftingBoard().load_board(board)
        if not board.has_game_ended():
            return 0
        left_win = self.predict_win(board.left_picks, board.right_picks)
        player_win = (1 if left_win else -1) == player
        return player_win

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. 
        """
        return DraftingBoard().load_board(board).get_board(player)

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        items = board[:4]
        items_p = [list(permutations(i, r=len(i))) for i in items]
        symmetries = list(product(*items_p))
        symmetries = [(*s, *board[4:]) for s in symmetries]
        return [(s, pi) for s in symmetries]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(np.array(board))

