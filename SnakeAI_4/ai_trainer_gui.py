import pygame
import sys
import threading
import time
from typing import List, Tuple
from src.game_logic import SnakeGame, Direction
from src.ai.genetic_algorithm import GeneticAlgorithm
from src.ai.ai_player import AIPlayer


class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, color: tuple, text_color: tuple = (255, 255, 255)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.hover_color = tuple(min(c + 30, 255) for c in color)
        self.is_hovered = False
        self.enabled = True
    
    def draw(self, screen, font):
        color = self.hover_color if self.is_hovered and self.enabled else self.color
        if not self.enabled:
            color = tuple(c // 2 for c in self.color)
        
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2, border_radius=5)
        
        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and self.enabled:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class AITrainerGUI:
    def __init__(self):
        pygame.init()
        
        display_info = pygame.display.Info()
        monitor_width = display_info.current_w
        monitor_height = display_info.current_h
        
        self.screen_width = int(monitor_width * 0.7)
        self.screen_height = int(monitor_height * 0.7)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)
        pygame.display.set_caption('Тренер ИИ для Змейки')
        
        self.clock = pygame.time.Clock()
        
        font_scale = min(self.screen_width / 1200, self.screen_height / 700)
        self.font_large = pygame.font.Font(None, int(40 * font_scale))
        self.font_medium = pygame.font.Font(None, int(28 * font_scale))
        self.font_small = pygame.font.Font(None, int(22 * font_scale))
        
        self.colors = {
            'background': (20, 20, 30),
            'panel': (30, 30, 45),
            'panel_light': (40, 40, 60),
            'text': (255, 255, 255),
            'text_dim': (150, 150, 150),
            'snake': (100, 100, 255),
            'snake_head': (50, 50, 200),
            'apple': (255, 50, 50),
            'grid': (40, 40, 50),
            'green': (50, 200, 50),
            'red': (200, 50, 50),
            'blue': (50, 100, 200),
            'yellow': (200, 200, 50),
            'graph_line': (100, 200, 255),
            'graph_fill': (50, 100, 150, 100)
        }
        
        self.field_width = 30
        self.field_height = 30
        self.cell_size = 10
        
        self.ga = None
        self.training_active = False
        self.training_paused = False
        self.training_thread = None
        
        self.data_lock = threading.Lock()
        
        self.current_cycle = 0
        self.population_data = []
        self.history = {
            'cycle': [],
            'best_score': [],
            'avg_score': [],
            'best_fitness': []
        }
        
        self.session_folder = None
        self.session_start_time = None
        self.best_model_path = None
        
        self.demo_game = None
        self.demo_ai = None
        self.demo_speed = 15
        
        self.scroll_offset = 0
        self.max_scroll = 0
        
        self.setup_buttons()
        
        self.selected_individual = None
        self.visualizing = False
        
        self.auto_save_interval = 10
        
    def scale(self, value, dimension='width'):
        if dimension == 'width':
            return int(value * self.screen_width / 1200)
        else:
            return int(value * self.screen_height / 700)
    
    def setup_buttons(self):
        button_y = self.scale(650, 'height')
        button_width = self.scale(120)
        button_height = self.scale(40, 'height')
        spacing = self.scale(10)
        start_x = self.scale(900)
        
        self.buttons = {
            'start': Button(start_x, button_y, button_width, button_height, 
                          'СТАРТ', self.colors['green']),
            'pause': Button(start_x + button_width + spacing, button_y, button_width, button_height,
                          'ПАУЗА', self.colors['yellow'], (0, 0, 0)),
            'stop': Button(start_x + (button_width + spacing) * 2, button_y, button_width, button_height,
                         'СТОП', self.colors['red']),
            'save': Button(start_x + (button_width + spacing) * 3, button_y, button_width, button_height,
                         'СОХРАНИТЬ', self.colors['blue'])
        }
        
        self.buttons['pause'].enabled = False
        self.buttons['stop'].enabled = False
        self.buttons['save'].enabled = False
        
        settings_y = self.scale(710, 'height')
        small_btn_width = self.scale(40)
        small_btn_height = self.scale(30, 'height')
        self.buttons['pop_minus'] = Button(self.scale(900), settings_y, small_btn_width, small_btn_height, '-', self.colors['panel_light'])
        self.buttons['pop_plus'] = Button(self.scale(1045), settings_y, small_btn_width, small_btn_height, '+', self.colors['panel_light'])
        self.buttons['speed_minus'] = Button(self.scale(900), settings_y + small_btn_height + self.scale(10, 'height'), small_btn_width, small_btn_height, '-', self.colors['panel_light'])
        self.buttons['speed_plus'] = Button(self.scale(1045), settings_y + small_btn_height + self.scale(10, 'height'), small_btn_width, small_btn_height, '+', self.colors['panel_light'])
        
        self.population_size = 200
        self.models_per_cycle = 200
    
    def start_training(self):
        if self.training_active:
            return
        
        import os
        from datetime import datetime
        
        self.session_start_time = datetime.now()
        session_name = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        self.session_folder = f"training_sessions/{session_name}"
        os.makedirs(self.session_folder, exist_ok=True)
        
        print(f"→ Начало сессии обучения: {session_name}")
        print(f"→ Папка сессии: {self.session_folder}")
        
        self.training_active = True
        self.training_paused = False
        self.buttons['start'].enabled = False
        self.buttons['pause'].enabled = True
        self.buttons['stop'].enabled = True
        self.buttons['save'].enabled = True
        
        self.ga = GeneticAlgorithm(
            population_size=self.models_per_cycle,
            mutation_rate=0.1,
            mutation_strength=0.25,
            elite_count=max(20, self.models_per_cycle // 10),
            field_width=self.field_width,
            field_height=self.field_height
        )
        
        self.current_cycle = 0
        self.history = {
            'cycle': [],
            'best_score': [],
            'avg_score': [],
            'best_fitness': []
        }
        self.best_model_path = None
        
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()
    
    def pause_training(self):
        self.training_paused = not self.training_paused
        if self.training_paused:
            self.buttons['pause'].text = 'ПРОДОЛЖИТЬ'
            self.buttons['pause'].color = self.colors['green']
        else:
            self.buttons['pause'].text = 'ПАУЗА'
            self.buttons['pause'].color = self.colors['yellow']
    
    def stop_training(self):
        import os
        from datetime import datetime
        
        self.training_active = False
        self.training_paused = False
        
        if self.best_model_path and os.path.exists(self.best_model_path):
            session_end_time = datetime.now()
            start_str = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            end_str = session_end_time.strftime("%Y%m%d_%H%M%S")
            final_name = f"{self.session_folder}/{start_str}_{end_str}_final.npy"
            
            try:
                import shutil
                shutil.copy(self.best_model_path, final_name)
                print(f"✓ Финальная модель сохранена: {final_name}")
            except Exception as e:
                print(f"✗ Ошибка сохранения финальной модели: {e}")
        
        if self.training_thread and self.training_thread.is_alive():
            if threading.current_thread() != self.training_thread:
                self.training_thread.join(timeout=2.0)
        
        self.buttons['start'].enabled = True
        self.buttons['pause'].enabled = False
        self.buttons['stop'].enabled = False
        self.buttons['pause'].text = 'ПАУЗА'
        self.buttons['pause'].color = self.colors['yellow']
        
        print("→ Сессия обучения завершена")
    
    def save_model(self):
        if self.ga:
            try:
                self.ga.save_best('best_ai.npy', auto_save_history=True)
                print("✓ Модель сохранена в best_ai.npy")
            except Exception as e:
                print(f"✗ Ошибка при сохранении модели: {e}")
    
    def training_loop(self):
        import numpy as np
        import os
        
        try:
            current_best_ai = None
            
            print(f"→ Бесконечный цикл обучения начат (N={self.models_per_cycle} моделей за цикл)")
            
            while self.training_active:
                while self.training_paused:
                    time.sleep(0.1)
                
                if not self.training_active:
                    break
                
                self.current_cycle += 1
                print(f"\n=== Цикл {self.current_cycle} ===")
                
                if current_best_ai is None:
                    if os.path.exists('best_ai.npy'):
                        print("→ Загружаем существующую модель...")
                        current_best_ai = self.ga.load_best('best_ai.npy')
                    else:
                        print("→ Начинаем с нулевой модели")
                        current_best_ai = AIPlayer()
                
                print(f"→ Обучаем {self.models_per_cycle} моделей от текущей лучшей...")
                
                new_population = []
                from src.ai.neural_network import NeuralNetwork
                
                for i in range(self.models_per_cycle):
                    if not self.training_active:
                        break
                    
                    child = AIPlayer()
                    child.neural_network = NeuralNetwork.crossover(
                        current_best_ai.neural_network, 
                        current_best_ai.neural_network
                    )
                    child.neural_network.mutate(self.ga.mutation_rate, self.ga.mutation_strength)
                    new_population.append(child)
                
                if not self.training_active:
                    break
                
                print(f"→ Оцениваем {len(new_population)} моделей...")
                fitness_scores = []
                
                for i, ai_player in enumerate(new_population):
                    if not self.training_active:
                        break
                    
                    try:
                        fitness, score = self.ga.evaluate_fitness(ai_player)
                        fitness_scores.append({
                            'index': i,
                            'fitness': fitness,
                            'score': score,
                            'ai_player': ai_player
                        })
                    except Exception as e:
                        print(f"Ошибка при оценке модели {i}: {e}")
                        continue
                
                if not self.training_active or len(fitness_scores) == 0:
                    break
                
                fitness_scores.sort(key=lambda x: x['fitness'], reverse=True)
                
                with self.data_lock:
                    self.population_data = fitness_scores
                
                best_fitness = fitness_scores[0]['fitness']
                best_score = fitness_scores[0]['score']
                avg_fitness = sum(f['fitness'] for f in fitness_scores) / len(fitness_scores)
                avg_score = sum(f['score'] for f in fitness_scores) / len(fitness_scores)
                
                with self.data_lock:
                    self.history['cycle'].append(self.current_cycle)
                    self.history['best_score'].append(best_score)
                    self.history['avg_score'].append(avg_score)
                    self.history['best_fitness'].append(best_fitness)
                    
                    self.demo_ai = fitness_scores[0]['ai_player']
                    self.demo_game = SnakeGame(self.field_width, self.field_height)
                
                print(f"Цикл {self.current_cycle}: Лучший={best_score}, Средний={avg_score:.1f}, Best fitness={best_fitness:.1f}")
                
                cycle_model_path = f"{self.session_folder}/cycle_{self.current_cycle:04d}_score_{best_score}.npy"
                try:
                    best_weights = fitness_scores[0]['ai_player'].neural_network.get_weights_flat()
                    np.save(cycle_model_path, best_weights)
                    self.best_model_path = cycle_model_path
                    print(f"✓ Сохранена лучшая модель цикла: {cycle_model_path}")
                except Exception as e:
                    print(f"✗ Ошибка сохранения: {e}")
                
                current_best_ai = fitness_scores[0]['ai_player']
                print(f"→ Загружена лучшая модель для следующего цикла")
        
        except Exception as e:
            print(f"✗ Критическая ошибка в training_loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.training_active:
                self.stop_training()
    
    def visualize_individual(self, ai_player: AIPlayer):
        self.selected_individual = ai_player
        self.visualizing = True
        self.demo_game = SnakeGame(self.field_width, self.field_height)
        self.demo_ai = ai_player
    
    def update_demo_game(self):
        if self.demo_game and self.demo_ai and not self.demo_game.game_over:
            state = self.demo_game.get_state_for_ai()
            direction = self.demo_ai.decide_direction(state)
            self.demo_game.set_direction(direction)
            self.demo_game.step()
        elif self.demo_game and self.demo_game.game_over and self.training_active:
            self.demo_game.reset()
    
    def draw_game_field(self):
        panel_x = self.scale(20)
        panel_y = self.scale(20, 'height')
        panel_width = self.scale(340)
        panel_height = self.scale(400, 'height')
        
        pygame.draw.rect(self.screen, self.colors['panel'], 
                        (panel_x, panel_y, panel_width, panel_height), border_radius=10)
        
        title = self.font_medium.render('Лучшая особь', True, self.colors['text'])
        self.screen.blit(title, (panel_x + self.scale(10), panel_y + self.scale(10, 'height')))
        
        if self.demo_game:
            field_x = panel_x + 10
            field_y = panel_y + 50
            
            for x in range(self.field_width + 1):
                pygame.draw.line(self.screen, self.colors['grid'],
                               (field_x + x * self.cell_size, field_y),
                               (field_x + x * self.cell_size, field_y + self.field_height * self.cell_size))
            
            for y in range(self.field_height + 1):
                pygame.draw.line(self.screen, self.colors['grid'],
                               (field_x, field_y + y * self.cell_size),
                               (field_x + self.field_width * self.cell_size, field_y + y * self.cell_size))
            
            for i, (x, y) in enumerate(self.demo_game.get_snake_body()):
                color = self.colors['snake_head'] if i == 0 else self.colors['snake']
                pygame.draw.rect(self.screen, color,
                               (field_x + x * self.cell_size + 1, field_y + y * self.cell_size + 1,
                                self.cell_size - 2, self.cell_size - 2))
            
            apple_x, apple_y = self.demo_game.apple
            pygame.draw.circle(self.screen, self.colors['apple'],
                             (field_x + apple_x * self.cell_size + self.cell_size // 2,
                              field_y + apple_y * self.cell_size + self.cell_size // 2),
                             self.cell_size // 2 - 1)
            
            score_text = self.font_small.render(f'Счёт: {self.demo_game.score}', True, self.colors['text'])
            length_text = self.font_small.render(f'Длина: {len(self.demo_game.snake)}', True, self.colors['text'])
            
            self.screen.blit(score_text, (field_x, field_y + self.field_height * self.cell_size + 10))
            self.screen.blit(length_text, (field_x, field_y + self.field_height * self.cell_size + 35))
    
    def draw_statistics_panel(self):
        panel_x = self.scale(380)
        panel_y = self.scale(20, 'height')
        panel_width = self.scale(500)
        panel_height = self.scale(400, 'height')
        
        pygame.draw.rect(self.screen, self.colors['panel'],
                        (panel_x, panel_y, panel_width, panel_height), border_radius=10)
        
        title = self.font_medium.render('Статистика цикла', True, self.colors['text'])
        self.screen.blit(title, (panel_x + self.scale(10), panel_y + self.scale(10, 'height')))
        
        with self.data_lock:
            population_data_copy = self.population_data.copy() if self.population_data else []
        
        if population_data_copy:
            y_offset = panel_y + 50
            header = self.font_small.render(
                f'№   | Fitness  | Счёт | Действие',
                True, self.colors['text_dim']
            )
            self.screen.blit(header, (panel_x + 10, y_offset))
            
            y_offset += 30
            
            visible_height = panel_height - 90
            max_items = visible_height // 25
            
            start_idx = self.scroll_offset
            end_idx = min(start_idx + max_items, len(population_data_copy))
            
            for i in range(start_idx, end_idx):
                data = population_data_copy[i]
                y_pos = y_offset + (i - start_idx) * 25
                
                color = self.colors['green'] if i < 5 else self.colors['text']
                
                text = f'{i+1:3d} | {data["fitness"]:8.1f} | {data["score"]:4d} | '
                text_surface = self.font_small.render(text, True, color)
                self.screen.blit(text_surface, (panel_x + 10, y_pos))
                
                view_button = pygame.Rect(panel_x + 320, y_pos, 80, 20)
                pygame.draw.rect(self.screen, self.colors['blue'], view_button, border_radius=3)
                view_text = self.font_small.render('Просмотр', True, self.colors['text'])
                view_text_rect = view_text.get_rect(center=view_button.center)
                self.screen.blit(view_text, view_text_rect)
                
                data['view_button'] = view_button
            
            if len(population_data_copy) > max_items:
                scrollbar_height = max(20, visible_height * max_items // len(population_data_copy))
                scrollbar_y = y_offset + (visible_height - scrollbar_height) * self.scroll_offset // max(1, len(population_data_copy) - max_items)
                
                pygame.draw.rect(self.screen, self.colors['panel_light'],
                               (panel_x + panel_width - 15, y_offset, 10, visible_height))
                pygame.draw.rect(self.screen, self.colors['blue'],
                               (panel_x + panel_width - 15, scrollbar_y, 10, scrollbar_height))
    
    def draw_graph(self):
        panel_x = self.scale(20)
        panel_y = self.scale(440, 'height')
        panel_width = self.scale(860)
        panel_height = self.scale(250, 'height')
        
        pygame.draw.rect(self.screen, self.colors['panel'],
                        (panel_x, panel_y, panel_width, panel_height), border_radius=10)
        
        title = self.font_medium.render('Прогресс обучения', True, self.colors['text'])
        self.screen.blit(title, (panel_x + self.scale(10), panel_y + self.scale(10, 'height')))
        
        with self.data_lock:
            history_copy = {
                'cycle': self.history['cycle'].copy() if 'cycle' in self.history else [],
                'best_score': self.history['best_score'].copy(),
                'avg_score': self.history['avg_score'].copy()
            }
        
        if len(history_copy['cycle']) > 1:
            graph_x = panel_x + 50
            graph_y = panel_y + 50
            graph_width = panel_width - 80
            graph_height = panel_height - 70
            
            pygame.draw.rect(self.screen, self.colors['panel_light'],
                           (graph_x, graph_y, graph_width, graph_height))
            
            max_score = max(history_copy['best_score']) if history_copy['best_score'] else 1
            max_gen = len(history_copy['cycle'])
            
            points_best = []
            points_avg = []
            
            for i, gen in enumerate(history_copy['cycle']):
                x = graph_x + (i / max(1, max_gen - 1)) * graph_width
                y_best = graph_y + graph_height - (history_copy['best_score'][i] / max(1, max_score)) * graph_height
                y_avg = graph_y + graph_height - (history_copy['avg_score'][i] / max(1, max_score)) * graph_height
                
                points_best.append((x, y_best))
                points_avg.append((x, y_avg))
            
            if len(points_avg) > 1:
                pygame.draw.lines(self.screen, self.colors['text_dim'], False, points_avg, 2)
            if len(points_best) > 1:
                pygame.draw.lines(self.screen, self.colors['graph_line'], False, points_best, 3)
            
            legend_x = graph_x + graph_width - 150
            legend_y = graph_y + 10
            
            pygame.draw.line(self.screen, self.colors['graph_line'], 
                           (legend_x, legend_y + 5), (legend_x + 30, legend_y + 5), 3)
            legend_text = self.font_small.render('Лучший', True, self.colors['text'])
            self.screen.blit(legend_text, (legend_x + 40, legend_y))
            
            pygame.draw.line(self.screen, self.colors['text_dim'],
                           (legend_x, legend_y + 25), (legend_x + 30, legend_y + 25), 2)
            legend_text = self.font_small.render('Средний', True, self.colors['text'])
            self.screen.blit(legend_text, (legend_x + 40, legend_y + 20))
    
    def draw_control_panel(self):
        panel_x = self.scale(900)
        panel_y = self.scale(20, 'height')
        panel_width = self.scale(280)
        panel_height = int(self.screen_height * 0.95)
        
        pygame.draw.rect(self.screen, self.colors['panel'],
                        (panel_x, panel_y, panel_width, panel_height), border_radius=10)
        
        title = self.font_large.render('Управление', True, self.colors['text'])
        self.screen.blit(title, (panel_x + self.scale(20), panel_y + self.scale(20, 'height')))
        
        y = panel_y + 80
        
        cycle_text = self.font_medium.render(f'Цикл обучения: {self.current_cycle}', 
                                          True, self.colors['text'])
        self.screen.blit(cycle_text, (panel_x + 20, y))
        
        if self.session_folder:
            session_text = self.font_small.render(f'Сессия: {self.session_folder.split("/")[-1]}', 
                                                  True, self.colors['text_dim'])
            self.screen.blit(session_text, (panel_x + 20, y + 30))
        
        y += 80
        
        with self.data_lock:
            population_data_copy = self.population_data.copy() if self.population_data else []
        
        if population_data_copy:
            stats_title = self.font_medium.render('Текущая статистика:', True, self.colors['text'])
            self.screen.blit(stats_title, (panel_x + 20, y))
            y += 35
            
            best = population_data_copy[0]
            avg_fitness = sum(d['fitness'] for d in population_data_copy) / len(population_data_copy)
            avg_score = sum(d['score'] for d in population_data_copy) / len(population_data_copy)
            
            stats = [
                f'Лучший счёт: {best["score"]}',
                f'Лучший fitness: {best["fitness"]:.1f}',
                f'Средний счёт: {avg_score:.1f}',
                f'Средний fitness: {avg_fitness:.1f}',
                f'Популяция: {len(population_data_copy)}'
            ]
            
            for stat in stats:
                stat_text = self.font_small.render(stat, True, self.colors['text'])
                self.screen.blit(stat_text, (panel_x + 30, y))
                y += 25
        
        y += 20
        
        settings_title = self.font_medium.render('Настройки:', True, self.colors['text'])
        self.screen.blit(settings_title, (panel_x + 20, y))
        y += 35
        
        models_text = self.font_small.render(f'Моделей за цикл: {self.models_per_cycle}', True, self.colors['text'])
        self.screen.blit(models_text, (panel_x + 30, y + 5))
        self.buttons['pop_minus'].rect.y = y
        self.buttons['pop_plus'].rect.y = y
        y += 40
        
        speed_text = self.font_small.render(f'Скорость визуализации: {self.demo_speed} FPS', True, self.colors['text'])
        self.screen.blit(speed_text, (panel_x + 30, y + 5))
        self.buttons['speed_minus'].rect.y = y
        self.buttons['speed_plus'].rect.y = y
        y += 50
        
        for button in self.buttons.values():
            button.draw(self.screen, self.font_small)
        
        if self.training_active:
            status_text = 'ПАУЗА' if self.training_paused else 'ОБУЧЕНИЕ...'
            status_color = self.colors['yellow'] if self.training_paused else self.colors['green']
        else:
            status_text = 'ГОТОВ К ЗАПУСКУ'
            status_color = self.colors['text_dim']
        
        status = self.font_medium.render(status_text, True, status_color)
        self.screen.blit(status, (panel_x + 20, panel_y + panel_height - 50))
    
    def handle_scroll(self, event):
        if event.type == pygame.MOUSEWHEEL:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            stats_panel = pygame.Rect(380, 20, 500, 400)
            
            if stats_panel.collidepoint(mouse_x, mouse_y):
                self.scroll_offset -= event.y * 3
                
                visible_height = 400 - 90
                max_items = visible_height // 25
                self.max_scroll = max(0, len(self.population_data) - max_items)
                self.scroll_offset = max(0, min(self.scroll_offset, self.max_scroll))
    
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop_training()
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.stop_training()
                        running = False
                
                self.handle_scroll(event)
                
                if self.buttons['start'].handle_event(event):
                    self.start_training()
                if self.buttons['pause'].handle_event(event):
                    self.pause_training()
                if self.buttons['stop'].handle_event(event):
                    self.stop_training()
                if self.buttons['save'].handle_event(event):
                    self.save_model()
                
                if self.buttons['pop_minus'].handle_event(event) and not self.training_active:
                    self.models_per_cycle = max(10, self.models_per_cycle - 10)
                if self.buttons['pop_plus'].handle_event(event) and not self.training_active:
                    self.models_per_cycle = self.models_per_cycle + 10
                
                if self.buttons['speed_minus'].handle_event(event):
                    self.demo_speed = max(5, self.demo_speed - 5)
                if self.buttons['speed_plus'].handle_event(event):
                    self.demo_speed = min(60, self.demo_speed + 5)
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    for data in self.population_data:
                        if 'view_button' in data and data['view_button'].collidepoint(mouse_pos):
                            self.visualize_individual(data['ai_player'])
            
            self.update_demo_game()
            
            self.screen.fill(self.colors['background'])
            
            self.draw_game_field()
            self.draw_statistics_panel()
            self.draw_graph()
            self.draw_control_panel()
            
            pygame.display.flip()
            self.clock.tick(self.demo_speed)
        
        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    print("=" * 60)
    print("ГРАФИЧЕСКИЙ ТРЕНЕР ИИ ДЛЯ ЗМЕЙКИ")
    print("=" * 60)
    print("\nИнструкция:")
    print("1. Нажмите СТАРТ для начала обучения")
    print("2. Наблюдайте за прогрессом в реальном времени")
    print("3. Используйте ПАУЗА для приостановки")
    print("4. Нажмите СОХРАНИТЬ для сохранения лучшей модели")
    print("5. Нажмите на 'Просмотр' чтобы увидеть конкретную особь")
    print("\nУправление:")
    print("- ESC - выход")
    print("- Прокрутка мыши - скролл списка особей")
    print("\nЗапуск...\n")
    
    trainer = AITrainerGUI()
    trainer.run()
