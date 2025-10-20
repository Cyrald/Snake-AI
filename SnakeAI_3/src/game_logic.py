import random
from enum import Enum
from typing import List, Tuple, Optional


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class SnakeGame:
    def __init__(self, width: int = 20, height: int = 20, initial_length: int = 3):
        self.width = width
        self.height = height
        self.initial_length = initial_length
        
        self.snake: List[Tuple[int, int]] = []
        self.direction = Direction.RIGHT
        self.apple: Tuple[int, int] = (0, 0)
        self.score = 0
        self.game_over = False
        
        self.reset()
    
    def reset(self):
        center_x = self.width // 2
        center_y = self.height // 2
        
        self.snake = [(center_x - i, center_y) for i in range(self.initial_length)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.game_over = False
        self.spawn_apple()
    
    def spawn_apple(self):
        while True:
            apple = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if apple not in self.snake:
                self.apple = apple
                break
    
    def set_direction(self, direction: Direction):
        opposite_directions = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        
        if direction != opposite_directions.get(self.direction):
            self.direction = direction
    
    def step(self) -> bool:
        if self.game_over:
            return False
        
        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = (head_x + dx, head_y + dy)
        
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height or
            new_head in self.snake[:-1]):
            self.game_over = True
            return False
        
        self.snake.insert(0, new_head)
        
        if new_head == self.apple:
            self.score += 1
            self.spawn_apple()
        else:
            self.snake.pop()
        
        return True
    
    def get_state_for_ai(self) -> dict:
        return {
            'head_position': self.snake[0],
            'field_size': (self.width, self.height),
            'apple_position': self.apple,
            'initial_length': self.initial_length,
            'current_length': len(self.snake),
            'score': self.score,
            'game_over': self.game_over,
            'direction': self.direction
        }
    
    def get_snake_body(self) -> List[Tuple[int, int]]:
        return self.snake.copy()
