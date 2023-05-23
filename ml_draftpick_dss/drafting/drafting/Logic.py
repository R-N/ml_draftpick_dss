import numpy as np

class DraftingBoard():

    def __init__(self, team_size=5, ban_size=3, hero_pool_size=120):
        self.team_size = team_size
        self.ban_size = ban_size

        self.hero_pool_size = hero_pool_size
        self.hero_pool = [
            tuple(
                1 if j == i else 0
                for j in range(hero_pool_size)
            )
            for i in range(hero_pool_size)
        ]

        self.ban_sequence = [
            (1, 1),
            (-1, 2),
            (1, 2),
            (-1, 1)
        ]
        self.pick_sequence = [
            (1, 1),
            (-1, 2),
            (1, 2),
            (-1, 2),
            (1, 2),
            (-1, 1)
        ]
        self.game_sequence = [
            *[(0, *b) for b in self.ban_sequence],
            *[(1, *p) for p in self.pick_sequence]
        ]

        self.left_picks = []
        self.right_picks = []
        self.left_bans = []
        self.right_bans = []

        self.round = 0

    def get_round(self):
        return self.game_sequence[self.round]

    def get_next_round(self):
        round = self.round+1
        if round >= len(self.game_sequence):
            return None
        return self.game_sequence[round]

    def get_picks(self, player):
        return self.left_picks if player == 1 else self.right_picks

    def get_bans(self, player):
        return self.left_bans if player == 1 else self.right_bans
    
    def get_array(self, type, player):
        return self.get_bans(player) if type == 0 else self.get_picks(player)

    def get_illegal_moves(self, player=None):
        return set().union((
            tuple(self.left_picks),
            tuple(self.right_picks),
            tuple(self.left_bans),
            tuple(self.right_bans),
        ))
    
    def get_illegal_mask(self, player=None):
        return np.sum(np.array(self.get_illegal_moves(player)), axis=-1)

    def get_legal_moves(self, player=None):
        """Returns all the legal moves for the given player.
        (1 for left, -1 for right)
        """
        return list(set(self.hero_pool) - self.get_illegal_moves(player))
    
    def get_double_legal_moves(self, player=None):
        legal_moves = self.get_legal_moves(player)
        legal_moves = [tuple(x) for x in legal_moves]
        double_legal_moves = set(
            i+j
            for i in legal_moves
            for j in legal_moves
            if i != j
        )
        return list(double_legal_moves)
    
    def get_legal_mask(self, player=None):
        illegal_mask = self.get_illegal_mask(player)
        if illegal_mask and len(list(illegal_mask)[0]) > 0:
            return 1 - self.get_illegal_mask(player)
        else:
            return np.sum(self.hero_pool, axis=-1)

    def has_legal_moves(self, player):
        return len(self.bans(player)) < self.ban_size or len(self.picks(player)) < self.team_size
    
    def has_game_ended(self):
        return len(self.left_picks) >= 5 and len(self.right_picks) >= 5

    def execute_move(self, hero, player=None, type=None):
        _type, _player, move_count = self.get_round()
        type = type or _type
        player = player or _player
        assert type == _type
        assert player == _player

        hero = tuple(int(round(x)) for x in hero)
        hero_1, hero_2 = hero[:self.hero_pool_size], hero[self.hero_pool_size:]
        #hero_1, hero_2 = hero
        #hero_1 = tuple(int(round(x)) for x in hero_1)
        #hero_2 = tuple(int(round(x)) for x in hero_2)
        illegal_moves = self.get_illegal_moves(player)
        assert hero_1 not in illegal_moves
        assert hero_2 not in illegal_moves
        if move_count < 2:
            assert sum(hero_2) == 0

        self.get_array(type, player).append(hero_1)
        if move_count > 1:
            self.get_array(type, player).append(hero_2)

        self.round += 1

    def get_board(self, player=1):
        return (
            self.get_bans(player),
            self.get_picks(player),
            self.get_bans(-player),
            self.get_picks(-player),
            self.get_legal_mask(), 
            self.get_round(),
            self.get_next_round(),
            self.round
        )
    
    def load_board(self, board):
        self.left_bans, self.left_picks, self.right_bans, self.right_picks = board[:4]
        self.round = board[-1]
