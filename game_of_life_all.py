import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from scipy.signal import convolve2d
from scipy import ndimage

# --- 1. Базовые настройки ---
N = 200  
ON = 1
OFF = 0

# --- 2. Ядро свёртки и правила игры ---
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

def step(grid):
    neighbors = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)
    birth = (grid == 0) & (neighbors == 3)
    survival = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    return (birth | survival).astype(int)

# --- 3. Цветная визуализация ---
def get_rgb(grid):
    rgb = np.zeros((N, N, 3))
    rgb[:, :] = [0.04, 0.04, 0.08]
    labeled_array, num_features = ndimage.label(grid, structure=np.ones((3,3)))
    cmap = plt.cm.tab20
    for i in range(1, num_features + 1):
        mask = (labeled_array == i)
        color = cmap((i - 1) % 20)[:3]
        rgb[mask] = np.clip(np.array(color) * 1.5, 0, 1)
    return rgb

# --- 4. RLE-декодер (гарантирует правильные паттерны кораблей) ---
def rle_to_grid(rle_str):
    rle = rle_str.strip().replace('!', '')
    rows = rle.split('$')
    grid_rows = []
    for row in rows:
        decoded = []
        i = 0
        while i < len(row):
            num = ''
            while i < len(row) and row[i].isdigit():
                num += row[i]
                i += 1
            count = int(num) if num else 1
            if i < len(row):
                if row[i] == 'o':
                    decoded.append('1' * count)
                elif row[i] == 'b':
                    decoded.append('0' * count)
                i += 1
        grid_rows.append(''.join(decoded))
    max_w = max(len(r) for r in grid_rows)
    grid_rows = [r.ljust(max_w, '0') for r in grid_rows]
    return np.array([[int(c) for c in row] for row in grid_rows], dtype=int)

# --- 5. Функция запуска и сохранения ---
def run_simulation(initial_grid, title, filename, frames=120):
    print(f"\n>>> Генерация: {title}...")
    print("    (Программа ждет, пока вы закроете окно)\n")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0a16')
    ax.set_facecolor('#0a0a16')
    ax.axis('off')
    
    text_obj = ax.set_title(title, color='#00ffff', fontsize=14, 
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
    print(f"    Сохранено: {filename}")
    print(f"    Воспроизведение... Закройте окно, чтобы продолжить.\n")
    
    # Сбрасываем на начальное состояние, чтобы окно показало старт
    grid = initial_grid.copy()
    mat.set_data(get_rgb(grid))
    
    def update_disp(frame):
        nonlocal grid
        grid = step(grid)
        mat.set_data(get_rgb(grid))
        return [mat]
    
    ani2 = animation.FuncAnimation(fig, update_disp, frames=frames,
                                    interval=100, blit=True, repeat=False)
    plt.show()  # ЖДЕТ закрытия окна
    plt.close(fig)
    print(f"<<< Цикл '{title}' завершен. Переходим к следующему.\n")

# --- 6. Генераторы начальных позиций ---

# ЦИКЛ 1: Рандом
def make_random():
    grid = np.zeros((N, N))
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
    positions = {'Block': (60, 60), 'Beehive': (140, 60), 
                 'Loaf': (60, 140), 'Tub': (140, 140)}
    for name, (cx, cy) in positions.items():
        coords = patterns[name]
        min_x = min(c[0] for c in coords); max_x = max(c[0] for c in coords)
        min_y = min(c[1] for c in coords); max_y = max(c[1] for c in coords)
        start_x = cx - (max_x - min_x) // 2
        start_y = cy - (max_y - min_y) // 2
        for dx, dy in coords:
            grid[start_y + dy, start_x + dx] = ON
    return grid

# ЦИКЛ 3: Осцилляторы
def make_oscillators():
    grid = np.zeros((N, N))
    pulsar_coords = [
        (2,0), (3,0), (4,0), (8,0), (9,0), (10,0),
        (0,2), (5,2), (7,2), (12,2), (0,3), (5,3), (7,3), (12,3),
        (0,4), (5,4), (7,4), (12,4),
        (2,5), (3,5), (4,5), (8,5), (9,5), (10,5),
        (2,7), (3,7), (4,7), (8,7), (9,7), (10,7),
        (0,8), (5,8), (7,8), (12,8), (0,9), (5,9), (7,9), (12,9),
        (0,10), (5,10), (7,10), (12,10),
        (2,12), (3,12), (4,12), (8,12), (9,12), (10,12)
    ]
    patterns = {
        'Blinker': [(0,0), (1,0), (2,0)],
        'Toad': [(1,0), (2,0), (3,0), (0,1), (1,1), (2,1)],
        'Beacon': [(0,0), (1,0), (0,1), (1,1), (2,2), (3,2), (2,3), (3,3)],
        'Pentadecathlon': [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (6,0), (7,0), (8,0), (9,0), (4,-1), (5,-1), (4,1), (5,1)],
        'Pulsar': pulsar_coords
    }
    positions = {
        'Blinker': (50, 60), 'Toad': (100, 60), 'Beacon': (150, 60),
        'Pentadecathlon': (75, 140), 'Pulsar': (125, 140)
    }
    for name, (cx, cy) in positions.items():
        coords = patterns[name]
        min_x = min(c[0] for c in coords); max_x = max(c[0] for c in coords)
        min_y = min(c[1] for c in coords); max_y = max(c[1] for c in coords)
        start_x = cx - (max_x - min_x) // 2
        start_y = cy - (max_y - min_y) // 2
        for dx, dy in coords:
            grid[start_y + dy, start_x + dx] = ON
    return grid

# ЦИКЛ 4: Космические корабли (ИНТЕГРИРОВАНЫ ПРАВИЛЬНЫЕ ПАТТЕРНЫ)
def make_spaceships():
    grid = np.zeros((N, N))
    
    # === RLE-паттерны из LifeWiki (проверенные!) ===
    GLIDER = rle_to_grid('bo$2bo$3o!')                 # 5 клеток, по диагонали
    LWSS   = rle_to_grid('bo2bo$o4b$o3bo$4o!')         # 9 клеток
    MWSS   = rle_to_grid('3bo2b$bo3bo$o5b$o4bo$5o!')  # 11 клеток
    HWSS   = rle_to_grid('3b2o2b$bo4bo$o6b$o5bo$6o!') # 13 клеток
    
    # !!! Отражаем горизонтально, чтобы корабли летели ВПРАВО !!!
    LWSS = np.fliplr(LWSS)
    MWSS = np.fliplr(MWSS)
    HWSS = np.fliplr(HWSS)
    
    # === Размещаем все ТРИ горизонтальных корабля в одной линии (Y=100) ===
    Y = 100
    
    # LWSS на x=20
    h, w = LWSS.shape
    grid[Y:Y+h, 20:20+w] = LWSS
    
    # MWSS на x=70
    h, w = MWSS.shape
    grid[Y:Y+h, 70:70+w] = MWSS
    
    # HWSS на x=130
    h, w = HWSS.shape
    grid[Y:Y+h, 130:130+w] = HWSS
    
    # === Глайдер отдельно (летит по диагонали) ===
    h, w = GLIDER.shape
    grid[20:20+h, 20:20+w] = GLIDER
    
    return grid

# ЦИКЛ 5: Ружьё Госпера
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

# --- 7. ЗАПУСК ВСЕХ 5 ЦИКЛОВ ПОСЛЕДОВАТЕЛЬНО ---
if __name__ == "__main__":
    print("=" * 60)
    print("ГЕНЕРАЦИЯ 5 GIF-ФАЙЛОВ")
    print("После каждого цикла программа ЖДЕТ, пока вы закроете окно,")
    print("и только потом запускает следующий цикл.")
    print("=" * 60)
    
    run_simulation(make_random(), 
                   "ЦИКЛ 1: РАНДОМ", 
                   "cycle_1_random.gif", frames=150)
    
    run_simulation(make_still_lifes(), 
                   "ЦИКЛ 2: НЕПОДВИЖНЫЕ (STILL LIFES)", 
                   "cycle_2_still_lifes.gif", frames=60)
    
    run_simulation(make_oscillators(), 
                   "ЦИКЛ 3: ОСЦИЛЛЯТОРЫ (5 ВИДОВ)", 
                   "cycle_3_oscillators.gif", frames=100)
    
    run_simulation(make_spaceships(), 
                   "ЦИКЛ 4: КОСМИЧЕСКИЕ КОРАБЛИ", 
                   "cycle_4_spaceships.gif", frames=150)
    
    run_simulation(make_gun(), 
                   "ЦИКЛ 5: РУЖЬЁ ГОСПЕРА", 
                   "cycle_5_gun.gif", frames=200)
    
    print("=" * 60)
    print("Все 5 GIF-файлов успешно созданы!")
    print("=" * 60)