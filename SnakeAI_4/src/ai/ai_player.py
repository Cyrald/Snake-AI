import numpy as np
from src.ai.neural_network import NeuralNetwork
from src.game_logic import Direction


class AIPlayer:
    def __init__(self, hidden_layers: list | None = None):
        if hidden_layers is None:
            hidden_layers = [1024, 1536, 1024, 512]
        
        input_size = 8
        output_size = 4
        
        layer_sizes = [input_size] + hidden_layers + [output_size]
        self.neural_network = NeuralNetwork(layer_sizes)
    
    def decide_direction(self, game_state: dict) -> Direction:
        inputs = self.state_to_input(game_state)
        outputs = self.neural_network.forward(inputs)
        
        direction_idx = np.argmax(outputs)
        directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        
        return directions[direction_idx]
    
    def state_to_input(self, game_state: dict) -> np.ndarray:
        head_x, head_y = game_state['head_position']
        apple_x, apple_y = game_state['apple_position']
        field_width, field_height = game_state['field_size']
        
        head_x_norm = head_x / field_width
        head_y_norm = head_y / field_height
        
        apple_x_norm = apple_x / field_width
        apple_y_norm = apple_y / field_height
        
        dx = (apple_x - head_x) / field_width
        dy = (apple_y - head_y) / field_height
        
        distance = np.sqrt(dx**2 + dy**2)
        
        current_length_norm = game_state['current_length'] / (field_width * field_height)
        
        inputs = np.array([
            head_x_norm,
            head_y_norm,
            apple_x_norm,
            apple_y_norm,
            dx,
            dy,
            distance,
            current_length_norm
        ])
        
        return inputs
