from copy import deepcopy
import random
import time

POSSIBLE_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def shuffle_board(matrix, moves = 100):
    shuffled_matrix = deepcopy(matrix)
    
    none_pos = find_empty(shuffled_matrix)
    
    for _ in range(moves):
        i, j = none_pos
        
        move = random.choice(POSSIBLE_MOVES)
        new_i, new_j = i + move[0], j + move[1]
        
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            shuffled_matrix[i][j], shuffled_matrix[new_i][new_j] = shuffled_matrix[new_i][new_j], shuffled_matrix[i][j]
            none_pos = (new_i, new_j)
    
    return shuffled_matrix

def find_empty(matrix):
    for i, row in enumerate(matrix):
        if None in row:
            return (i, row.index(None))

def generate_moves(matrix):
    moves_set = set()
    empty_row, empty_col = find_empty(matrix)
    
    for (row_offset, col_offset) in POSSIBLE_MOVES:
        new_row, new_col = empty_row + row_offset, empty_col + col_offset
        
        if 0 <= new_row < len(matrix) and 0 <= new_col < len(matrix[0]):
            new_matrix = deepcopy(matrix)
            new_matrix[empty_row][empty_col], new_matrix[new_row][new_col] = new_matrix[new_row][new_col], new_matrix[empty_row][empty_col]
            
            matrix_tuple = tuple(tuple(row) for row in new_matrix)
            moves_set.add(matrix_tuple)
    
    return [list(list(row) for row in matrix_tuple) for matrix_tuple in moves_set]

def bfs(init, goal):
    visited = set()
    queue = [(init, [])]
    max_len = len(queue)

    while queue:
        current_state, path = queue.pop(0)

        state_tuple = tuple(tuple(row) for row in current_state)
        if state_tuple in visited:
            continue
        
        current_len = len(queue)
        if current_len > max_len:
            max_len = current_len

        if current_state == goal:
            return max_len
    
        visited.add(state_tuple)

        for move in generate_moves(current_state):
            move_tuple = tuple(tuple(row) for row in move)
            
            if move_tuple not in visited:  
                new_path = path + [move] 
                queue.append((move, new_path)) 

def dfs(init, goal):
    visited = set()
    stack = [(init, [])]
    max_len = len(stack)

    while stack:
        current_state, path = stack.pop()
        state_tuple = tuple(tuple(row) for row in current_state)
        if state_tuple in visited:
            continue
        
        current_len = len(stack)
        if max_len < current_len:
            max_len = current_len

        if current_state == goal:
            return len(path)
        
        visited.add(state_tuple)  
        
        for move in generate_moves(current_state):
            move_tuple = tuple(tuple(row) for row in move)
            
            if move_tuple not in visited:  
                new_path = path + [move] 
                stack.append((move, new_path)) 

def idfs(init, goal, max_depth = 50):
    for depth in range(max_depth):
        stack = [(init, [], 0)] 
        visited = set()
        max_len = len(stack)

        while stack:
            current_state, path, current_depth = stack.pop()

            if current_depth > depth:
                continue

            state_tuple = tuple(tuple(row) for row in current_state)
            if state_tuple in visited:
                continue
            
            current_len = len(stack)
            if max_len < current_len:
                max_len = current_depth

            if current_state == goal:
                return max_len
            
            visited.add(state_tuple)

            for move in generate_moves(current_state):
                move_tuple = tuple(tuple(row) for row in move)
                
                if move_tuple not in visited:  
                    new_path = path + [move] 
                    stack.append((move, new_path, current_depth + 1)) 

def greedyTileMatching(init, goal):
    queue = [(tileMatchingHeuristic(init, goal), init, [], 0)]  
    visited = set()
    max_len = len(queue)

    while queue:
        _, current_state, path, g = queue.pop(0)

        current_len = len(queue)
        if max_len < current_len:
            max_len = current_len

        if current_state == goal:
            return max_len

        state_tuple = tuple(tuple(row) for row in current_state)
        if state_tuple in visited:
            continue

        visited.add(state_tuple)

        for move in generate_moves(current_state):
            new_path = path + [move]
            h = tileMatchingHeuristic(move, goal)
            new_g = g + 1  
            new_f = new_g - h 

            queue.append((new_f, move, new_path, new_g))

def greedyManhattan(init, goal):
    queue = [(tileMatchingHeuristic(init, goal), init, [], 0)]  
    visited = set()
    max_len = len(queue)

    while queue:
        _, current_state, path, g = queue.pop(0)

        current_len = len(queue)
        if max_len < current_len:
            max_len = current_len

        if current_state == goal:
            return max_len

        state_tuple = tuple(tuple(row) for row in current_state)
        if state_tuple in visited:
            continue

        visited.add(state_tuple)

        for move in generate_moves(current_state):
            new_path = path + [move]
            h = manhattanHeuristic(move, goal)
            new_g = g + 1  
            new_f = new_g - h 

            queue.append((new_f, move, new_path, new_g))

def hillClimbingTileMatching(init, goal):
    current = deepcopy(init)
    while True:
        neighbors = generate_moves(current)

        if not neighbors:
            break
        
        best_neighbor = None
        best_score = tileMatchingHeuristic(current, goal)
        
        for neighbor in neighbors:
            score = tileMatchingHeuristic(neighbor, goal)
            if score > best_score:
                best_score = score
                best_neighbor = neighbor
        
        if best_neighbor is None or best_score <= tileMatchingHeuristic(current, goal):
            break
        
        current = best_neighbor
    
    return current

def hillClimbingManhattan(init, goal):
    current = deepcopy(init)
    while True:
        neighbors = generate_moves(current)
        
        if not neighbors:
            break
        
        best_neighbor = None
        best_score = manhattanHeuristic(current, goal)
        
        for neighbor in neighbors:
            score = manhattanHeuristic(neighbor, goal)
            if score > best_score:
                best_score = score
                best_neighbor = neighbor
        
        if best_neighbor is None or best_score <= manhattanHeuristic(current, goal):
            break
        
        current = best_neighbor
    
    return current

def tileMatchingHeuristic(state, goal):
    correct = 0
    for i in range(len(state)):
        for j in range(len(state[i])):
            if state[i][j] == goal[i][j]:
                correct += 1

    return correct

def manhattanHeuristic(state, goal):
    distance = 0
    for i in range(len(state)):
        for j in range(len(state[i])):
            value = state[i][j]
            if value != 0:  
                for x in range(len(goal)):
                    for y in range(len(goal[x])):
                        if goal[x][y] == value:
                            distance += abs(i - x) + abs(j - y)
                            break
    return distance

if __name__ == '__main__':
    goal_matrix = [
        [1, 2, 3],
        [8, None, 4],
        [7, 6, 5]
    ]
    
    # goal_matrix = [
    #     [1, 2, 3, 4],
    #     [12, 13, 14, 5],
    #     [11, None, 15, 6],
    #     [10, 9, 8, 7]
    # ]

    init_matrix = shuffle_board(goal_matrix)

    start_time = time.time()
    result = bfs(init_matrix, goal_matrix)
    end_time = time.time()
    exec_time = end_time - start_time
    
    print('BFS')
    print(result)
    print(f'{exec_time:.6f} segundos')
    print()
    
    #
    
    start_time = time.time()
    result = dfs(init_matrix, goal_matrix)
    end_time = time.time()
    exec_time = end_time - start_time

    print('DFS')
    print(result)
    print(f'{exec_time:.6f} segundos')
    print()
    
    #
    
    start_time = time.time()
    result = idfs(init_matrix, goal_matrix)
    end_time = time.time()
    exec_time = end_time - start_time

    print('IDFS')
    print(result)
    print(f'{exec_time:.6f} segundos')
    print()

    #

    start_time = time.time()
    result = greedyTileMatching(init_matrix, goal_matrix)
    end_time = time.time()
    exec_time = end_time - start_time

    print('GREEDY TILE MATCHING')
    print(result)
    print(f'{exec_time:.6f} segundos')
    print()

    #

    start_time = time.time()
    result = greedyManhattan(init_matrix, goal_matrix)
    end_time = time.time()
    exec_time = end_time - start_time

    print('GREEDY MANHATTAN')
    print(result)
    print(f'{exec_time:.6f} segundos')
    print()

    #

    start_time = time.time()
    result = hillClimbingTileMatching(init_matrix, goal_matrix)
    end_time = time.time()
    exec_time = end_time - start_time

    print('HILL CLIMB TILE MATCHING')
    print(result)
    print(f'{exec_time:.6f} segundos')
    print()

    #

    start_time = time.time()
    result = hillClimbingManhattan(init_matrix, goal_matrix)
    end_time = time.time()
    exec_time = end_time - start_time
 
    print('HILL CLIMB MANHATTAN')
    print(result)
    print(f'{exec_time:.6f} segundos')
    print()
