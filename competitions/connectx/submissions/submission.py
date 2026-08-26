import numpy as np

ROWS = 6
COLS = 7
INAROW = 4
EMPTY = 0
DEPTH = 4
WIN_SCORE = 1_000_000
LOSS_SCORE = -1_000_000


def get_valid_moves(board):
    return [col for col in range(COLS) if board[0][col] == EMPTY]


def drop_piece(board, col, mark):
    new_board = board.copy()
    for row in range(ROWS - 1, -1, -1):
        if new_board[row][col] == EMPTY:
            new_board[row][col] = mark
            break
    return new_board


def is_winning_move(board, mark):
    for row in range(ROWS):
        for col in range(COLS - 3):
            if all(board[row][col + i] == mark for i in range(4)):
                return True
    for col in range(COLS):
        for row in range(ROWS - 3):
            if all(board[row + i][col] == mark for i in range(4)):
                return True
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            if all(board[row + i][col + i] == mark for i in range(4)):
                return True
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            if all(board[row - i][col + i] == mark for i in range(4)):
                return True
    return False


def evaluate_window(window, mark, opp_mark):
    score = 0
    mark_count = window.count(mark)
    opp_count = window.count(opp_mark)
    empty_count = window.count(EMPTY)
    if mark_count == 4:
        score += 1000
    elif mark_count == 3 and empty_count == 1:
        score += 50
    elif mark_count == 2 and empty_count == 2:
        score += 10
    if opp_count == 3 and empty_count == 1:
        score -= 80
    elif opp_count == 2 and empty_count == 2:
        score -= 8
    return score


def evaluate_board(board, mark, opp_mark):
    score = 0
    center_col = COLS // 2
    center_count = int(np.sum(board[:, center_col] == mark))
    score += center_count * 6
    for row in range(ROWS):
        for col in range(COLS - 3):
            window = list(board[row, col:col + 4])
            score += evaluate_window(window, mark, opp_mark)
    for col in range(COLS):
        for row in range(ROWS - 3):
            window = list(board[row:row + 4, col])
            score += evaluate_window(window, mark, opp_mark)
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            window = [board[row + i][col + i] for i in range(4)]
            score += evaluate_window(window, mark, opp_mark)
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            window = [board[row - i][col + i] for i in range(4)]
            score += evaluate_window(window, mark, opp_mark)
    return score


def minimax(board, depth, alpha, beta, is_maximizing, mark, opp_mark):
    valid_moves = get_valid_moves(board)
    if is_winning_move(board, mark):
        return WIN_SCORE + depth
    if is_winning_move(board, opp_mark):
        return LOSS_SCORE - depth
    if len(valid_moves) == 0:
        return 0
    if depth == 0:
        return evaluate_board(board, mark, opp_mark)
    ordered_moves = sorted(valid_moves, key=lambda c: abs(c - COLS // 2))
    if is_maximizing:
        max_score = -float('inf')
        for col in ordered_moves:
            new_board = drop_piece(board, col, mark)
            score = minimax(new_board, depth - 1, alpha, beta, False, mark, opp_mark)
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return max_score
    else:
        min_score = float('inf')
        for col in ordered_moves:
            new_board = drop_piece(board, col, opp_mark)
            score = minimax(new_board, depth - 1, alpha, beta, True, mark, opp_mark)
            min_score = min(min_score, score)
            beta = min(beta, score)
            if alpha >= beta:
                break
        return min_score


def agent(observation, configuration):
    board = np.array(observation.board).reshape(ROWS, COLS)
    mark = observation.mark
    opp_mark = 1 if mark == 2 else 2
    valid_moves = get_valid_moves(board)

    for col in valid_moves:
        temp_board = drop_piece(board, col, mark)
        if is_winning_move(temp_board, mark):
            return col

    for col in valid_moves:
        temp_board = drop_piece(board, col, opp_mark)
        if is_winning_move(temp_board, opp_mark):
            return col

    best_score = -float('inf')
    best_col = valid_moves[0]
    ordered_moves = sorted(valid_moves, key=lambda c: abs(c - COLS // 2))
    for col in ordered_moves:
        new_board = drop_piece(board, col, mark)
        score = minimax(new_board, DEPTH - 1, -float('inf'), float('inf'), False, mark, opp_mark)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col
