import pygame
import sys
import random
from src.game_logic import SnakeGame, Direction
from src.ai.ai_player import AIPlayer


class SnakeGameGUI:
    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        cell_size: int = 20,
        ai_player: AIPlayer | None = None
    ):
        pygame.init()
        
        self.game = SnakeGame(width, height)
        self.cell_size = cell_size
        self.ai_player = ai_player
        
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size + 50
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        if ai_player:
            pygame.display.set_caption('Змейка - Режим ИИ')
        else:
            pygame.display.set_caption('Змейка - Игрок')
        
        self.clock = pygame.time.Clock()
        self.fps = 10 if not ai_player else 15
        
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        self.colors = {
            'background': (0, 0, 0),
            'snake': (0, 255, 0) if not ai_player else (100, 100, 255),
            'head': (0, 200, 0) if not ai_player else (50, 50, 200),
            'apple': (255, 0, 0),
            'text': (255, 255, 255),
            'grid': (40, 40, 40),
            'ai_mode': (100, 100, 255)
        }
    
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if not self.ai_player:
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        self.game.set_direction(Direction.UP)
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        self.game.set_direction(Direction.DOWN)
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        self.game.set_direction(Direction.LEFT)
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        self.game.set_direction(Direction.RIGHT)
                
                if event.key == pygame.K_SPACE and self.game.game_over:
                    self.game.reset()
                elif event.key == pygame.K_ESCAPE:
                    return False
        
        return True
    
    def draw_grid(self):
        for x in range(0, self.screen_width, self.cell_size):
            pygame.draw.line(self.screen, self.colors['grid'],
                           (x, 0), (x, self.screen_height - 50))
        for y in range(0, self.screen_height - 50, self.cell_size):
            pygame.draw.line(self.screen, self.colors['grid'],
                           (0, y), (self.screen_width, y))
    
    def draw(self):
        self.screen.fill(self.colors['background'])
        
        self.draw_grid()
        
        for i, (x, y) in enumerate(self.game.get_snake_body()):
            color = self.colors['head'] if i == 0 else self.colors['snake']
            pygame.draw.rect(self.screen, color,
                           (x * self.cell_size, y * self.cell_size,
                            self.cell_size - 2, self.cell_size - 2))
        
        apple_x, apple_y = self.game.apple
        pygame.draw.circle(self.screen, self.colors['apple'],
                         (apple_x * self.cell_size + self.cell_size // 2,
                          apple_y * self.cell_size + self.cell_size // 2),
                         self.cell_size // 2 - 2)
        
        score_text = self.font.render(f'Счёт: {self.game.score}', True, self.colors['text'])
        self.screen.blit(score_text, (10, self.screen_height - 45))
        
        state = self.game.get_state_for_ai()
        mode_text = "ИИ" if self.ai_player else "Игрок"
        mode_color = self.colors['ai_mode'] if self.ai_player else self.colors['text']
        
        info_text = self.small_font.render(
            f'{mode_text} | Поле: {state["field_size"][0]}x{state["field_size"][1]} | Длина: {state["current_length"]}',
            True, mode_color
        )
        self.screen.blit(info_text, (self.screen_width - 350, self.screen_height - 40))
        
        if self.game.game_over:
            game_over_text = self.font.render('ИГРА ОКОНЧЕНА! Нажмите ПРОБЕЛ',
                                             True, self.colors['text'])
            text_rect = game_over_text.get_rect(center=(self.screen_width // 2,
                                                         (self.screen_height - 50) // 2))
            
            pygame.draw.rect(self.screen, self.colors['background'],
                           (text_rect.x - 10, text_rect.y - 10,
                            text_rect.width + 20, text_rect.height + 20))
            pygame.draw.rect(self.screen, self.colors['text'],
                           (text_rect.x - 10, text_rect.y - 10,
                            text_rect.width + 20, text_rect.height + 20), 3)
            
            self.screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()
    
    def run(self):
        running = True
        
        while running:
            running = self.handle_input()
            
            if not self.game.game_over:
                if self.ai_player:
                    state = self.game.get_state_for_ai()
                    direction = self.ai_player.decide_direction(state)
                    self.game.set_direction(direction)
                
                self.game.step()
            
            self.draw()
            self.clock.tick(self.fps)
        
        pygame.quit()
        sys.exit()


def show_menu():
    pygame.init()
    screen = pygame.display.set_mode((700, 500))
    pygame.display.set_caption('Змейка - Выбор режима')
    
    font_title = pygame.font.Font(None, 64)
    font_option = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)
    
    colors = {
        'background': (0, 0, 0),
        'text': (255, 255, 255),
        'highlight': (0, 255, 0),
        'ai_highlight': (100, 100, 255)
    }
    
    selected = 0
    options = ['Играть человеком', 'Смотреть ИИ', 'Обучить ИИ (GUI)', 'Обучить ИИ (консоль)']
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    pygame.quit()
                    return selected
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        
        screen.fill(colors['background'])
        
        title = font_title.render('ЗМЕЙКА + ИИ', True, colors['text'])
        title_rect = title.get_rect(center=(350, 60))
        screen.blit(title, title_rect)
        
        for i, option in enumerate(options):
            if i == selected:
                color = colors['ai_highlight'] if i > 0 else colors['highlight']
            else:
                color = colors['text']
            
            text = font_option.render(option, True, color)
            text_rect = text.get_rect(center=(350, 150 + i * 60))
            screen.blit(text, text_rect)
        
        help_text = font_small.render('↑/↓ - выбор, ENTER - начать, ESC - выход', True, colors['text'])
        help_rect = help_text.get_rect(center=(350, 450))
        screen.blit(help_text, help_rect)
        
        pygame.display.flip()
        clock.tick(30)


if __name__ == '__main__':
    choice = show_menu()
    
    field_width = random.randint(10, 25)
    field_height = random.randint(10, 25)
    
    if choice == 0:
        print(f'Запуск игры Змейка с полем {field_width}x{field_height}')
        print('Управление: стрелки или WASD')
        print('ПРОБЕЛ - перезапуск после окончания игры')
        print('ESC - выход')
        
        gui = SnakeGameGUI(width=field_width, height=field_height, cell_size=20)
        gui.run()
    
    elif choice == 1:
        import os
        import numpy as np
        
        print('Режим просмотра ИИ')
        
        ai_player = AIPlayer()
        
        ai_file = 'best_ai.npy'
        if os.path.exists(ai_file):
            print(f'Загружаем обученного ИИ из {ai_file}...')
            try:
                weights = np.load(ai_file)
                ai_player.neural_network.set_weights_flat(weights)
                print('✓ Обученный ИИ загружен успешно!')
            except Exception as e:
                print(f'⚠ Ошибка загрузки: {e}')
                print('Используем случайного ИИ')
        else:
            print(f'Файл {ai_file} не найден')
            print('Используем случайного ИИ (обучите сначала через тренер ИИ)')
        
        print(f'Запуск игры с ИИ на поле {field_width}x{field_height}')
        print('ПРОБЕЛ - перезапуск игры')
        print('ESC - выход')
        
        gui = SnakeGameGUI(
            width=field_width,
            height=field_height,
            cell_size=20,
            ai_player=ai_player
        )
        gui.run()
    
    elif choice == 2:
        print('Запуск графического тренера ИИ...')
        from ai_trainer_gui import AITrainerGUI
        trainer = AITrainerGUI()
        trainer.run()
    
    elif choice == 3:
        print('Запуск консольного обучения ИИ...')
        print('Используйте: python train_ai.py')
        print('\nПараметры:')
        print('  --generations N  - количество поколений')
        print('  --population N   - размер популяции')
        print('  --width N        - ширина поля')
        print('  --height N       - высота поля')
        sys.exit()
