import sys
import os
from src.ai.genetic_algorithm import GeneticAlgorithm


def train_ai(
    generations: int = 50,
    population_size: int = 50,
    field_width: int = 15,
    field_height: int = 15,
    save_interval: int = 10
):
    print("=" * 60)
    print("ОБУЧЕНИЕ ИИ ДЛЯ ИГРЫ ЗМЕЙКА")
    print("=" * 60)
    print(f"\nПараметры обучения:")
    print(f"  Поколений: {generations}")
    print(f"  Размер популяции: {population_size}")
    print(f"  Размер поля: {field_width}x{field_height}")
    print(f"  Сохранение каждые {save_interval} поколений")
    print("\nНачинаем обучение...\n")
    
    ga = GeneticAlgorithm(
        population_size=population_size,
        mutation_rate=0.1,
        mutation_strength=0.5,
        elite_count=5,
        field_width=field_width,
        field_height=field_height
    )
    
    best_overall_score = 0
    
    try:
        for gen in range(generations):
            stats = ga.evolve_generation(verbose=True)
            
            if stats['best_score'] > best_overall_score:
                best_overall_score = stats['best_score']
                print(f"  🎉 Новый рекорд! Счёт: {best_overall_score}")
            
            if (gen + 1) % save_interval == 0:
                filename = f"best_ai_gen_{gen + 1}.npy"
                ga.save_best(filename)
            
            if (gen + 1) % 10 == 0:
                print("-" * 60)
        
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 60)
        print(f"\nЛучший общий результат:")
        print(f"  Счёт: {ga.best_score}")
        print(f"  Fitness: {ga.best_fitness:.1f}")
        
        ga.save_best("best_ai.npy")
        print(f"\nЛучший ИИ сохранён в best_ai.npy")
        print("\nЗапустите main.py и выберите 'Смотреть ИИ' для просмотра результата")
        
    except KeyboardInterrupt:
        print("\n\nОбучение прервано пользователем")
        print("Сохраняем текущего лучшего ИИ...")
        ga.save_best("best_ai_interrupted.npy")
        print("Сохранено в best_ai_interrupted.npy")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение ИИ для игры Змейка')
    parser.add_argument('--generations', type=int, default=50,
                       help='Количество поколений для обучения (по умолчанию: 50)')
    parser.add_argument('--population', type=int, default=50,
                       help='Размер популяции (по умолчанию: 50)')
    parser.add_argument('--width', type=int, default=15,
                       help='Ширина поля (по умолчанию: 15)')
    parser.add_argument('--height', type=int, default=15,
                       help='Высота поля (по умолчанию: 15)')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Интервал сохранения (по умолчанию: каждые 10 поколений)')
    
    args = parser.parse_args()
    
    train_ai(
        generations=args.generations,
        population_size=args.population,
        field_width=args.width,
        field_height=args.height,
        save_interval=args.save_interval
    )
