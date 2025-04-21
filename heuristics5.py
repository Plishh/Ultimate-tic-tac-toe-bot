import time
import numpy as np
from typing import Dict, List, Tuple


import time
import numpy as np
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import utils 
from utils import State
from utils import Action
import time
from utils import ImmutableState 
from torch import tensor
from utils import convert_board_to_string

Action = Tuple[int, int, int, int]



class StudentAgent:
    def __init__(self, depth=5):
        self.depth = depth
        self.lines = [
            [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],  # Rows
            [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],  # Columns
            [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]                           # Diagonals
        ]
        self.position_weights = np.array([
            [0.3, 0.2, 0.3],  # Corners high, edges low, center highest
            [0.2, 0.4, 0.2],
            [0.3, 0.2, 0.3]
        ])
        self.corners = [(0, 0), (0, 2), (2, 0), (2, 2)]  # Define corner positions

    def choose_action(self, state: 'State') -> Action:
        start_time = time.time()
        valid_actions = state.get_all_valid_actions()
        if not valid_actions:
            return None
        if len(valid_actions) == 1:
            return valid_actions[0]

        for action in valid_actions:
            new_state = state.change_state(action)
            if new_state.is_terminal() and new_state.terminal_utility() == 1.0:
                return action  # Immediate win
        for action in valid_actions:
            inverted_state = state.invert()
            new_inverted = inverted_state.change_state(action)
            if new_inverted.is_terminal() and new_inverted.terminal_utility() == 0.0:
                return action  # Block opponent's immediate win

        best_action, _ = self.alpha_beta_search(state, start_time, valid_actions)
        return best_action if best_action is not None else valid_actions[0]

    def alpha_beta_search(self, state: 'State', start_time: float, valid_actions: List[Action]) -> Tuple[Action, float]:
        best_action = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        valid_actions.sort(key=lambda a: -self._action_priority(state, a))

        for action in valid_actions:
            new_state = state.change_state(action)
            value = self.minimax(new_state, self.depth - 1, False, alpha, beta, start_time)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
        return best_action, best_value

    def minimax(self, state: 'State', depth: int, maximizing_player: bool, 
                alpha: float, beta: float, start_time: float) -> float:
        if time.time() - start_time > 2.5:
            return self.evaluate_state(state)
        
        if state.is_terminal():
            return state.terminal_utility() * 100000
        
        if depth == 0:
            return self.evaluate_state(state)

        valid_actions = state.get_all_valid_actions()
        if not valid_actions:
            return self.evaluate_state(state)

        if maximizing_player:
            max_eval = -float('inf')
            for action in valid_actions:
                new_state = state.change_state(action)
                eval = self.minimax(new_state, depth - 1, False, alpha, beta, start_time)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            inverted_state = state.invert()
            for action in inverted_state.get_all_valid_actions():
                new_state = inverted_state.change_state(action)
                eval = self.minimax(new_state.invert(), depth - 1, True, alpha, beta, start_time)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_state(self, state: 'State') -> float:
        meta_board = state.local_board_status
        if state.is_terminal():
            if state.terminal_utility() == 1:
                return state.terminal_utility() * 100000
            else:
                return state.terminal_utility() * -100000

        weights = {
            'meta_win': 100000000,      # Game win
            'meta_almost_win': 50000,# Two in a row with open spot on meta-board
            'meta_one_two': 10000,   # One mark with two open spots on meta-board
            'small_win': 1000,       # Regular local board win
            'center_small_win': 3000,# Center local board win
            'corner_small_win': 2000,# Corner local board win (new)
            'small_almost_win': 500, # Two in a row in local board
            'small_one_two': 200,    # One mark with two open spots in local board
            'force_move': 500,       # Forces opponent to a won/full board
            'center_cell': 100,      # Center cell in small board
            'positional_meta': 100,  # Base for meta-board positional weights
            'positional_local': 10   # Base for local board positional weights
        }

        # Meta-board evaluation
        meta_score = 0.0
        meta_counts = self._analyze_lines(meta_board)
        if meta_counts['player_wins'] > 0:
            return weights['meta_win']
        elif meta_counts['opponent_wins'] > 0:
            return -weights['meta_win']
        meta_score += meta_counts['player_almost'] * weights['meta_almost_win']
        meta_score -= meta_counts['opponent_almost'] * weights['meta_almost_win']
        meta_score += meta_counts['player_one_two'] * weights['meta_one_two']
        meta_score -= meta_counts['opponent_one_two'] * weights['meta_one_two']
        meta_score += np.sum((meta_board == 1) * self.position_weights) * weights['positional_meta']
        meta_score -= np.sum((meta_board == 2) * self.position_weights) * weights['positional_meta']

        # Local board evaluation
        local_score = 0.0
        for i in range(3):
            for j in range(3):
                status = meta_board[i][j]
                if status == 1:
                    if (i, j) == (1, 1):
                        local_score += weights['center_small_win']
                    elif (i, j) in self.corners:
                        local_score += weights['corner_small_win']  # New corner win bonus
                    else:
                        local_score += weights['small_win']
                elif status == 2:
                    if (i, j) == (1, 1):
                        local_score -= weights['center_small_win']
                    elif (i, j) in self.corners:
                        local_score -= weights['corner_small_win']  # New corner loss penalty
                    else:
                        local_score -= weights['small_win']

                if status == 0:
                    local_counts = self._analyze_lines(state.board[i][j])
                    local_score += local_counts['player_almost'] * weights['small_almost_win']
                    local_score -= local_counts['opponent_almost'] * weights['small_almost_win']
                    local_score += local_counts['player_one_two'] * weights['small_one_two']
                    local_score -= local_counts['opponent_one_two'] * weights['small_one_two']
                    local_score += np.sum((state.board[i][j] == 1) * self.position_weights) * weights['positional_local']
                    local_score -= np.sum((state.board[i][j] == 2) * self.position_weights) * weights['positional_local']

                if state.board[i][j][1][1] == 1:
                    local_score += weights['center_cell']
                elif state.board[i][j][1][1] == 2:
                    local_score -= weights['center_cell']

        # Forcing move advantage
        force_score = 0.0
        if state.prev_local_action:
            target_i, target_j = state.prev_local_action
            if meta_board[target_i][target_j] in [1, 2, 3]:
                force_score += weights['force_move'] if state.fill_num == 1 else -weights['force_move']
        elif state.fill_num == 1 and state.prev_local_action is None:
            force_score += weights['force_move']

        # Composite score
        total_score = (meta_score * 0.7) + (local_score / 9 * 0.3) + (force_score * 0.2)
        return total_score

    def _analyze_lines(self, board: np.ndarray) -> Dict[str, int]:
        counts = {
            'player_wins': 0,
            'opponent_wins': 0,
            'player_almost': 0,
            'opponent_almost': 0,
            'player_one_two': 0,
            'opponent_one_two': 0
        }
        for line in self.lines:
            line_counts = self._count_line(board, line)
            if line_counts[1] == 3:
                counts['player_wins'] += 1
            elif line_counts[2] == 3:
                counts['opponent_wins'] += 1
            elif line_counts[1] == 2 and line_counts[0] == 1 and line_counts[2] == 0:
                counts['player_almost'] += 1
            elif line_counts[2] == 2 and line_counts[0] == 1 and line_counts[1] == 0:
                counts['opponent_almost'] += 1
            elif line_counts[1] == 1 and line_counts[0] == 2 and line_counts[2] == 0:
                counts['player_one_two'] += 1
            elif line_counts[2] == 1 and line_counts[0] == 2 and line_counts[1] == 0:
                counts['opponent_one_two'] += 1
        return counts

    def _count_line(self, board: np.ndarray, line: List[Tuple[int, int]]) -> Dict[int, int]:
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for i, j in line:
            counts[board[i][j]] += 1
        return counts

    def _action_priority(self, state: 'State', action: Action) -> int:
        i, j, k, l = action
        priority = 0
        
        temp_state = state.change_state(action)
        if temp_state.local_board_status[i][j] == 1 and state.local_board_status[i][j] == 0:
            if (i, j) == (1, 1):
                priority += 150  # Center win
            elif (i, j) in self.corners:
                priority += 125  # Corner win (new, slightly less than center)
            else:
                priority += 100  # Other win
        
        priority += int(self.position_weights[i][j] * 10)  # Meta-board position
        priority += int(self.position_weights[k][l] * 5)   # Local board position
        
        return priority

#opponent
    
class StudentAgent1:        
    def __init__(self):
        return

    def choose_action(self, state: State, player: int=2) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        
        return self.minimax(state, player)

    def max_value(self, state: State, alpha, beta, depth, max_depth):
        if state.is_terminal() or depth == max_depth: return self.evaluate(state), None

        value = float('-inf')
        best_action = None

        if len(state.get_all_valid_actions()) == 81:
            return [1,1,1,1]

        for action in state.get_all_valid_actions():
            next_state = state.change_state(action)
            next_value = self.min_value(next_state, alpha, beta, depth + 1, max_depth)[0]
            if next_value > value:
                value = next_value
                best_action = action

            alpha = max(alpha, value)
            if value >= beta: break

        return value, best_action

    def min_value(self, state: State, alpha, beta, depth, max_depth):
        if state.is_terminal() or depth == max_depth: return self.evaluate(state), None

        value = float('inf')
        best_action = None

        for action in state.get_all_valid_actions():
            next_state = state.change_state(action)
            next_value = self.max_value(next_state, alpha, beta, depth + 1, max_depth)[0]
            if next_value < value:
                value = next_value
                best_action = action
            
            beta = min(beta, value)
            if value <= alpha: return value, best_action

        return value, best_action

    def minimax(self, state: State, player: int = 2):
        _, best_move = self.max_value(state, float('-inf'), float('inf'), 0, 3) if player == 1 else self.min_value(state, float('-inf'), float('inf'), 0, 4)
        return best_move
    
    def local_board_evaluate(self, board):
        score = 0.0
        lines = []

        position_weights = np.array([
            [0.3, 0.2, 0.3],
            [0.2, 0.4, 0.2],
            [0.3, 0.2, 0.3]
        ])

        score += np.sum((board == 1) * position_weights) / 1.4
        score -= np.sum((board == 2) * position_weights) / 1.4

        line_score = 0.0
        lines.extend(board)
        lines.extend(board.T)
        lines.append(np.diag(board))
        lines.append(np.diag(np.fliplr(board)))

        for line in lines:
            p1_count = np.sum(line == 1)
            p2_count = np.sum(line == 2)
            empty_count = np.sum(line == 0)

            if p1_count == 3:
                return 1.0
            elif p2_count == 3:
                return -1.0
            elif p1_count == 2 and empty_count == 1:
                line_score += 0.5
            elif p1_count == 1 and empty_count == 2:
                line_score += 0.2
            elif p2_count == 2 and empty_count == 1:
                line_score -= 0.5
            elif p2_count == 1 and empty_count == 2:
                line_score -= 0.2

        score += line_score / 3.5
        return score
    
    def free_move_advantage(self, state: State):
        board_row, board_col = state.prev_local_action
        if state.local_board_status[board_row][board_col] != 0:
            return -1
        return 0
    
    def global_board_evaluate(self, state: State):
        global_score = 0.0
        lines = []

        position_weights = np.array([
            [0.3, 0.2, 0.3],
            [0.2, 0.4, 0.2],
            [0.3, 0.2, 0.3]
        ])

        board = state.local_board_status
        global_score += np.sum((board == 1) * position_weights) / 1.4
        global_score -= np.sum((board == 2) * position_weights) / 1.4

        line_score = 0.0
        lines.extend(board)
        lines.extend(board.T)
        lines.append(np.diag(board))
        lines.append(np.diag(np.fliplr(board)))
        for line in lines:
            p1_count = np.sum(line == 1)
            p2_count = np.sum(line == 2)
            empty_count = np.sum(line == 0)
            if p1_count == 3:
                return 1.0
            elif p2_count == 3:
                return -1.0
            elif p1_count == 2 and empty_count == 1:
                line_score += 0.5
            elif p1_count == 1 and empty_count == 2:
                line_score += 0.2
            elif p2_count == 2 and empty_count == 1:
                line_score -= 0.5
            elif p2_count == 1 and empty_count == 2:
                line_score -= 0.2
        
        global_score += line_score / 3.5
        
        local_score = 0.0
        for i in range(3):
            for j in range(3):
                local_board = state.board[i][j]
                local_score += self.local_board_evaluate(local_board)

        free_move_score = self.free_move_advantage(state)
        
        global_score = (global_score * 0.7) + (local_score / 9.0) * 0.3 + free_move_score * 0.2

        return global_score


    def evaluate(self, state) -> float:
        return self.global_board_evaluate(state)



    

class RandomStudentAgent(StudentAgent):
    def choose_action(self, state: State) -> Action:
        # If you're using an existing Player 1 agent, you may need to invert the state
        # to have it play as Player 2. Uncomment the next line to invert the state.
        # state = state.invert()

        # Choose a random valid action from the current game state
        return state.get_random_valid_action()

def run(your_agent: StudentAgent, opponent_agent: StudentAgent, start_num: int):
    your_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    opponent_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    turn_count = 0
    
    state = State(fill_num=start_num)
    
    while not state.is_terminal():
        turn_count += 1

        agent_name = "your_agent" if state.fill_num == 1 else "opponent_agent"
        agent = your_agent if state.fill_num == 1 else opponent_agent
        stats = your_agent_stats if state.fill_num == 1 else opponent_agent_stats

        start_time = time.time()
        action = agent.choose_action(state.clone()) if state.fill_num == 1 else agent.choose_action(state.clone(), player=2)
        end_time = time.time()
        
        random_action = state.get_random_valid_action()
        if end_time - start_time > 3 :
            print(f"{agent_name} timed out!")
            stats["timeout_count"] += 1
            #action = random_action
        if not state.is_valid_action(action):
            print(f"{agent_name} made an invalid action!")
            stats["invalid_count"] += 1
            action = random_action
                
        state = state.change_state(action)

    print(f"== {your_agent.__class__.__name__} (1) vs {opponent_agent.__class__.__name__} (2) - First Player: {start_num} ==")
        
    if state.terminal_utility() == 1:
        print("You win!")
        
    elif state.terminal_utility() == 0:
        print("You lose!")
        print(convert_board_to_string(state.board))
    else:
        print("Draw")
        print(convert_board_to_string(state.board))

    for agent_name, stats in [("your_agent", your_agent_stats), ("opponent_agent", opponent_agent_stats)]:
        print(f"{agent_name} statistics:")
        print(f"Timeout count: {stats['timeout_count']}")
        print(f"Invalid count: {stats['invalid_count']}")
        
    print(f"Turn count: {turn_count}\n")

your_agent = lambda: StudentAgent()
opponent_agent = lambda: StudentAgent1()

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)
    
# class RandomStudentAgent(StudentAgent):
#     def choose_action(self, state: State) -> Action:
#         # If you're using an existing Player 1 agent, you may need to invert the state
#         # to have it play as Player 2. Uncomment the next line to invert the state.
#         # state = state.invert()

#         # Choose a random valid action from the current game state
#         return state.get_random_valid_action()

# def run(your_agent, opponent_agent, start_num):
#     state = State(fill_num=start_num)    
#     turn_count = 0
#     your_agent_stats = {"timeout_count": 0, "invalid_count": 0}
#     opponent_agent_stats = {"timeout_count": 0, "invalid_count": 0}

#     while not state.is_terminal():
#         turn_count += 1
#         if state.fill_num == 1:
#             action = your_agent.choose_action(state)
#         else:
#             print("Your turn:")
#             while True:
#                 try:
#                     outrow = int(input("Enter outer row: "))
#                     outcol = int(input("Enter outer column: "))
#                     row = int(input("Enter row: "))
#                     col = int(input("Enter column: "))
#                     action = (outrow, outcol, row, col)
#                     if state.is_valid_action(action):
#                         break
#                     else:
#                         print("Invalid action!")
#                         your_agent_stats["invalid_count"] += 1
#                 except ValueError:
#                     print("Invalid input! Please enter integers.")
#                     your_agent_stats["invalid_count"] += 1

#         state = state.change_state(action)
#         print(convert_board_to_string(state.board))

#     print(f"== {your_agent.__class__.__name__} (1) vs {opponent_agent.__class__.__name__} (2) - First Player: {start_num} ==")
        
#     if state.terminal_utility() == 1:
#         print("You win!")
#     elif state.terminal_utility() == 0:
#         print("You lose!")
#     else:
#         print("Draw")

#     for agent_name, stats in [("your_agent", your_agent_stats), ("opponent_agent", opponent_agent_stats)]:
#         print(f"{agent_name} statistics:")
#         print(f"Timeout count: {stats['timeout_count']}")
#         print(f"Invalid count: {stats['invalid_count']}")
        
#     print(f"Turn count: {turn_count}\n")

# your_agent = lambda: StudentAgent()
# opponent_agent = lambda: RandomStudentAgent()

# run(your_agent(), opponent_agent(), 1)
# run(your_agent(), opponent_agent(), 2)
