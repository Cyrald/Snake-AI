import random
import os
import numpy as np
from datetime import datetime
from typing import List, Tuple
from src.ai.neural_network import NeuralNetwork
from src.ai.ai_player import AIPlayer
from src.game_logic import SnakeGame


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int = 200,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.25,
        elite_count: int = 20,
        field_width: int = 30,
        field_height: int = 30
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_count = elite_count
        self.field_width = field_width
        self.field_height = field_height
        
        self.population: List[AIPlayer] = []
        self.generation = 0
        self.best_fitness = 0
        self.best_score = 0
        
        os.makedirs('models/history', exist_ok=True)
        
        self.initialize_population()
    
    def initialize_population(self):
        self.population = []
        for _ in range(self.population_size):
            ai_player = AIPlayer()
            self.population.append(ai_player)
    
    def evaluate_fitness(self, ai_player: AIPlayer, max_steps: int = 5000) -> Tuple[float, int]:
        game = SnakeGame(self.field_width, self.field_height)
        steps = 0
        steps_without_food = 0
        max_steps_without_food = self.field_width * self.field_height * 3
        
        position_history = []
        loop_detection_window = 8
        
        while not game.game_over and steps < max_steps:
            state = game.get_state_for_ai()
            direction = ai_player.decide_direction(state)
            
            old_score = game.score
            game.set_direction(direction)
            game.step()
            
            if game.score > old_score:
                steps_without_food = 0
                position_history = []
            else:
                steps_without_food += 1
            
            if steps_without_food > max_steps_without_food:
                break
            
            position_history.append(game.snake[0])
            if len(position_history) > loop_detection_window:
                position_history.pop(0)
                if len(set(position_history)) < loop_detection_window // 2:
                    break
            
            steps += 1
        
        score = game.score
        length = len(game.snake)
        
        fitness = score * 1000 + length * 10 + steps * 0.1
        
        return fitness, score
    
    def evolve_generation(self, verbose: bool = True) -> dict:
        fitness_scores = []
        
        for ai_player in self.population:
            fitness, score = self.evaluate_fitness(ai_player)
            fitness_scores.append((fitness, score, ai_player))
        
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        best_fitness = fitness_scores[0][0]
        best_score = fitness_scores[0][1]
        avg_fitness = sum(f[0] for f in fitness_scores) / len(fitness_scores)
        avg_score = sum(f[1] for f in fitness_scores) / len(fitness_scores)
        
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_score = best_score
        
        elites = [f[2] for f in fitness_scores[:self.elite_count]]
        
        new_population = [elite for elite in elites]
        
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection(fitness_scores)
            parent2 = self.tournament_selection(fitness_scores)
            
            child_nn = NeuralNetwork.crossover(parent1.neural_network, parent2.neural_network)
            child_nn.mutate(self.mutation_rate, self.mutation_strength)
            
            child = AIPlayer()
            child.neural_network = child_nn
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        stats = {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'best_score': best_score,
            'avg_fitness': avg_fitness,
            'avg_score': avg_score,
            'best_overall_fitness': self.best_fitness,
            'best_overall_score': self.best_score
        }
        
        if verbose:
            print(f"Поколение {self.generation}: "
                  f"Лучший счёт={best_score:.0f}, "
                  f"Средний счёт={avg_score:.1f}, "
                  f"Лучший fitness={best_fitness:.1f}")
        
        return stats
    
    def tournament_selection(self, fitness_scores: List[Tuple[float, int, AIPlayer]], tournament_size: int = 3) -> AIPlayer:
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        winner = max(tournament, key=lambda x: x[0])
        return winner[2]
    
    def get_best_ai(self) -> AIPlayer:
        best_ai = self.population[0]
        best_fitness = -float('inf')
        
        for ai_player in self.population:
            fitness, _ = self.evaluate_fitness(ai_player)
            if fitness > best_fitness:
                best_fitness = fitness
                best_ai = ai_player
        
        return best_ai
    
    def save_best(self, filename: str = "best_ai.npy", auto_save_history: bool = True):
        best_ai = self.get_best_ai()
        weights = best_ai.neural_network.get_weights_flat()
        np.save(filename, weights)
        print(f"Лучший ИИ сохранён в {filename}")
        
        if auto_save_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_filename = f"models/history/gen_{self.generation}_score_{self.best_score:.0f}_{timestamp}.npy"
            np.save(history_filename, weights)
            print(f"История: {history_filename}")
    
    def load_best(self, filename: str = "best_ai.npy") -> AIPlayer:
        weights = np.load(filename)
        ai_player = AIPlayer()
        ai_player.neural_network.set_weights_flat(weights)
        return ai_player
