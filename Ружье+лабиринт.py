import numpy as np
import matplotlib
# Используем Agg, чтобы окна не открывались и не блокировали скрипт
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from scipy.signal import convolve2d
from scipy import ndimage

# --- 1. Базовые настройки ---
N = 200  # Размер поля (достаточно большой для всех фигур)
ON = 1
OFF = 0

# --- 2. Ядро свёртки и правила игры ---
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

def step(grid):
    """Один шаг игры «Жизнь» (векторизованный)"""
    neighbors = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)
    birth = (grid == 0) & (neighbors == 3)
    survival = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    return (birth | survival).astype(int)

# --- 3. Функция цветной визуализации ---
def get_rgb(grid):
    """Превращает бинарную сетку в цветную RGB-картинку"""
    rgb = np.zeros((N, N, 3))
    rgb[:, :] = [0.04, 0.04, 0.08] # Темный фон
    
    labeled_array, num_features = ndimage.label(grid, structure=np.ones((3,3)))
    cmap = plt.cm.tab20
    
    for i in range(1, num_features + 1):
        mask = (labeled_array == i)
        color = cmap((i - 1) % 20)[:3]
        color = np.clip(np.array(color) * 1.5, 0, 1) # Усиливаем яркость
        rgb[mask] = color
    return rgb

# --- 4. Функция запуска и сохранения GIF ---
def run_simulation(initial_grid, title, filename, frames=120):
    print(f"Генерация: {title}...")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0a16')
    ax.set_facecolor('#0a0a16')
    ax.axis('off')
    
    # Стильный заголовок
    text_obj = ax.set_title(title, color='#00ffff', fontsize=20, 
                            fontfamily='monospace', fontweight='bold', pad=20)
    text_obj.set_path_effects([pe.withStroke(linewidth=4, foreground='#0066ff')])
    
    grid = initial_grid.copy()
    rgb_image = get_rgb(grid)
    mat = ax.imshow(rgb_image, animated=True)
    
    def update(frame):
        nonlocal grid
        grid = step(grid)
        mat.set_data(get_rgb(grid))
        return [mat]
        
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
    ani.save(filename, writer='pillow', fps=15)
    plt.close(fig)
    print(f"Сохранено: {filename}\n")

# --- 5. Генераторы начальных позиций ---

# ЦИКЛ 1: Рандом
def make_random():
    grid = np.zeros((N, N))
    # Заполняем центр поля случайными значениями
    grid[50:150, 50:150] = np.random.choice([ON, OFF], size=(100, 100), p=[0.3, 0.7])
    return grid

# ЦИКЛ 2: Неподвижные структуры (Still Lifes)
def make_still_lifes():
    grid = np.zeros((N, N))
    patterns = {
        'Block': [(0,0), (0,1), (1,0), (1,1)],
        'Beehive': [(1,0), (2,0), (0,1), (3,1), (1,2), (2,2)],
        'Loaf': [(1,0), (2,0), (0,1), (3,1), (1,2), (3,2), (2,3)],
        'Tub': [(1,0), (0,1), (2,1), (1,2)]
    }
    
    # Расставляем их в шахматном порядке
    y_pos = 30
    for name, coords in patterns.items():
        x_pos = 30
        for _ in range(5): # Повторяем 5 раз каждую фигуру
            for dx, dy in coords:
                if x_pos + dx < N and y_pos + dy < N:
                    grid[y_pos + dy, x_pos + dx] = ON
            x_pos += 30
        y_pos += 40
    return grid

# ЦИКЛ 3: Осцилляторы
def make_oscillators():
    grid = np.zeros((N, N))
    patterns = {
        'Blinker': [(0,0), (1,0), (2,0)],
        'Toad': [(1,0), (2,0), (3,0), (0,1), (1,1), (2,1)],
        'Beacon': [(0,0), (1,0), (0,1), (1,1), (2,2), (3,2), (2,3), (3,3)],
        'Pentadecathlon': [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0), (4,-1), (5,-1), (4,1), (5,1)]
    }
    
    y_pos = 30
    for name, coords in patterns.items():
        x_pos = 30
        for _ in range(4):
            for dx, dy in coords:
                if x_pos + dx < N and y_pos + dy < N:
                    grid[y_pos + dy, x_pos + dx] = ON
            x_pos += 40
        y_pos += 50
    return grid

# ЦИКЛ 4: Космические корабли (Spaceships)
def make_spaceships():
    grid = np.zeros((N, N))
    patterns = {
        'Glider': [(0,1), (1,2), (2,0), (2,1), (2,2)],
        'LWSS': [(1,0), (4,0), (0,1), (0,2), (4,2), (0,3), (1,3), (2,3), (3,3)],
        'MWSS': [(1,0), (4,0), (0,1), (0,2), (5,2), (0,3), (1,3), (2,3), (3,3), (4,3), (5,3)],
        'HWSS': [(1,0), (5,0), (0,1), (0,2), (6,2), (0,3), (1,3), (2,3), (3,3), (4,3), (5,3), (6,3)]
    }
    
    y_pos = 30
    for name, coords in patterns.items():
        x_pos = 30
        for _ in range(3):
            for dx, dy in coords:
                if x_pos + dx < N and y_pos + dy < N:
                    grid[y_pos + dy, x_pos + dx] = ON
            x_pos += 50
        y_pos += 40
    return grid

# ЦИКЛ 5: Ружьё Госпера (Gun)
def make_gun():
    grid = np.zeros((N, N))
    coordinates = [
        (1, 5), (1, 6), (2, 5), (2, 6),          
        (35, 3), (35, 4), (36, 3), (36, 4),      
        (25, 1), (23, 2), (25, 2),               
        (21, 3), (22, 3), (21, 4), (22, 4), (21, 5), (22, 5),
        (23, 6), (25, 6), (25, 7),
        (13, 3), (14, 3),                        
        (12, 4), (16, 4),
        (11, 5), (17, 5),
        (11, 6), (15, 6), (17, 6), (18, 6),
        (11, 7), (17, 7),
        (12, 8), (16, 8),
        (13, 9), (14, 9)
    ]
    for x, y in coordinates:
        grid[y + 10, x + 10] = ON
    return grid

# --- 6. ЗАПУСК ВСЕХ 5 ЦИКЛОВ ---
if __name__ == "__main__":
    print("Начинаем генерацию 5 GIF-файлов...\n")
    
    # Цикл 1
    run_simulation(make_random(), "ЦИКЛ 1: РАНДОМ", "cycle_1_random.gif", frames=150)
    
    # Цикл 2
    run_simulation(make_still_lifes(), "ЦИКЛ 2: НЕПОДВИЖНЫЕ (STILL LIFES)", "cycle_2_still_lifes.gif", frames=60)
    
    # Цикл 3
    run_simulation(make_oscillators(), "ЦИКЛ 3: ОСЦИЛЛЯТОРЫ", "cycle_3_oscillators.gif", frames=100)
    
    # Цикл 4
    run_simulation(make_spaceships(), "ЦИКЛ 4: КОСМИЧЕСКИЕ КОРАБЛИ", "cycle_4_spaceships.gif", frames=100)
    
    # Цикл 5
    run_simulation(make_gun(), "ЦИКЛ 5: РУЖЬЁ ГОСПЕРА", "cycle_5_gun.gif", frames=200)
    
    print("Все 5 GIF-файлов успешно созданы и сохранены в папке с проектом!")