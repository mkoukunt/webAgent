import math
import random


class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # The game/decision state object
        self.parent = parent  # Parent node
        self.move = move  # The move that led to this state
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node was visited (N)
        self.wins = 0.0  # Total accumulated score/wins from simulations (W)
        self.untried_moves = state.get_legal_moves()  # Legal actions not yet expanded

    @property
    def win_rate(self):
        return self.wins / self.visits if self.visits > 0 else 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.state.is_terminal()

    def select_child(self, exploration_weight=1.414):
        """Selection Phase: Chooses a child node maximizing the UCB1 formula."""
        best_score = -float('inf')
        best_children = []

        for child in self.children:
            # UCB1 = Exploitation + Exploration
            exploitation = child.wins / child.visits
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            ucb_score = exploitation + exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_children = [child]
            elif ucb_score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def expand(self):
        """Expansion Phase: Populates the tree with an untried move."""
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        next_state = self.state.make_move(move)
        child_node = MCTSNode(state=next_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def rollout(self):
        """Simulation Phase: Plays out the game randomly to a terminal state."""
        current_rollout_state = self.state
        while not current_rollout_state.is_terminal():
            possible_moves = current_rollout_state.get_legal_moves()
            move = random.choice(possible_moves)
            current_rollout_state = current_rollout_state.make_move(move)
        return current_rollout_state.get_result()

    def backpropagate(self, result):
        """Backpropagation Phase: Updates stats upstream towards the root node."""
        self.visits += 1
        # Update reward depending on whose turn it was or general game results
        self.wins += result.get(self.state.player_to_move, 0.0)
        if self.parent:
            self.parent.backpropagate(result)


def mcts_search(root_state, iterations=1000):
    """Executes the complete MCTS loop for a specified budget of iterations."""
    root = MCTSNode(state=root_state)

    for _ in range(iterations):
        node = root

        # 1. Selection
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child()

        # 2. Expansion
        if not node.is_terminal():
            node = node.expand()

        # 3. Simulation
        result = node.rollout()

        # 4. Backpropagation
        node.backpropagate(result)

    # Return the move yielding the highest visit count
    return max(root.children, key=lambda c: c.visits).move

class TicTacToeState:
    def __init__(self, board=None, player_to_move=1):
        # 0 = Empty, 1 = X (MCTS), 2 = O (Opponent)
        self.board = board if board is not None else [0] * 9
        self.player_to_move = player_to_move

    def get_legal_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == 0]

    def make_move(self, move):
        new_board = list(self.board)
        new_board[move] = self.player_to_move
        next_player = 2 if self.player_to_move == 1 else 1
        return TicTacToeState(new_board, next_player)

    def _check_winner(self):
        win_states = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], # Columns
            [0, 4, 8], [2, 4, 6]             # Diagonals
        ]
        for line in win_states:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] != 0:
                return self.board[line[0]]
        if 0 not in self.board:
            return 0 # Draw
        return None # In progress

    def is_terminal(self):
        return self._check_winner() is not None

    def get_result(self):
        winner = self._check_winner()
        if winner == 1:
            return {1: 1.0, 2: 0.0}   # Player 1 wins
        elif winner == 2:
            return {1: 0.0, 2: 1.0}   # Player 2 wins
        else:
            return {1: 0.5, 2: 0.5}   # Tie game

    def print_board(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for i in range(0, 9, 3):
            print(f" {symbols[self.board[i]]} | {symbols[self.board[i+1]]} | {symbols[self.board[i+2]]} ")
            if i < 6: print("---+---+---")
        print()


if __name__ == "__main__":
    state = TicTacToeState()
    print("Initial Board State:")
    state.print_board()

    while not state.is_terminal():
        if state.player_to_move == 1:
            # MCTS analyzes 1000 simulated paths before making a move
            best_move = mcts_search(state, iterations=1000)
            print(f"MCTS (Player X) chooses position: {best_move}")
        else:
            # Opponent makes a completely random move
            best_move = random.choice(state.get_legal_moves())
            print(f"Random Opponent (Player O) chooses position: {best_move}")

        state = state.make_move(best_move)
        state.print_board()

    winner = state._check_winner()
    if winner == 0:
        print("Game over: It's a draw!")
    else:
        print(f"Game over: Player {'X' if winner == 1 else 'O'} wins!")
