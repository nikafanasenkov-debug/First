import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as pe
from scipy.signal import convolve2d
from scipy import ndimage

# --- 1. Настройки ---
N = 150                
ON = 255               
OFF = 0                

# --- 2. Точная функция добавления ружья Госпера ---
def addGosperGliderGun(grid, offset_x=5, offset_y=5):
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
        grid[y + offset_y, x + offset_x] = ON

# --- 3. Инициализация поля ---
grid = np.zeros((N, N))
addGosperGliderGun(grid)

# --- 4. Ядро свёртки ---
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

# --- 5. Настройка анимации ---
BG_COLOR = '#0a0a16'      
GRID_COLOR = '#151530'    

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor(BG_COLOR) 
ax.set_facecolor(BG_COLOR)        

# Освобождаем место сверху для заголовка
fig.subplots_adjust(top=0.85)

initial_img = np.full((N, N, 3), 0.04) 
mat = ax.imshow(initial_img, animated=True)

# --- 6. Стилизация заголовка ---
title_text = "РУЖЬЁ ГОСПЕРА"
text_obj = fig.text(0.5, 0.92, title_text, 
                    color='#00ffff', fontsize=28, fontfamily='monospace', 
                    fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.6", facecolor='#0a0a16', 
                              edgecolor='#00ffff', linewidth=2, alpha=0.9))
text_obj.set_path_effects([pe.withStroke(linewidth=4, foreground='#0066ff')])

ax.axis('off') 

# Добавляем стильную сетку
for i in range(0, N, 10):
    ax.axhline(i - 0.5, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    ax.axvline(i - 0.5, color=GRID_COLOR, linewidth=0.5, alpha=0.5)

# --- 7. Счетчик выстрелов (ПЕРЕМЕЩЕН В ЛЕВЫЙ НИЖНИЙ УГОЛ) ---
# Изменили y с 0.95 на 0.05 (низ) и va с 'top' на 'bottom'
shot_text = ax.text(0.02, 0.05, "ВЫСТРЕЛОВ: 0", transform=ax.transAxes,
                    color='#00ffff', fontsize=14, fontfamily='monospace',
                    fontweight='bold', va='bottom', ha='left',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='#0a0a16', 
                              edgecolor='#00ffff', linewidth=1, alpha=0.8))

# Выбираем палитру
cmap = plt.cm.tab20 

# --- 8. Векторизованная функция обновления ---
def update(frame):
    global grid
    grid_bool = grid / 255
    
    neighbors = convolve2d(grid_bool, kernel, mode='same', boundary='fill', fillvalue=0)
    
    birth = (grid_bool == 0) & (neighbors == 3)
    survival = (grid_bool == 1) & ((neighbors == 2) | (neighbors == 3))
    
    new_grid_bool = birth | survival
    grid = new_grid_bool.astype(float) * 255
    
    # --- МАГИЯ ЦВЕТА ---
    labeled_array, num_features = ndimage.label(new_grid_bool, structure=np.ones((3,3)))
    
    rgb_image = np.zeros((N, N, 3))
    rgb_image[:, :] = [0.04, 0.04, 0.08] 
    
    for i in range(1, num_features + 1):
        mask = (labeled_array == i)
        color = cmap((i - 1) % 20)[:3] 
        color = np.clip(np.array(color) * 1.5, 0, 1)
        rgb_image[mask] = color
        
    mat.set_data(rgb_image)
    
    # --- ОБНОВЛЕНИЕ СЧЕТЧИКА ---
    shots = frame // 30
    shot_text.set_text(f"ВЫСТРЕЛОВ: {shots}")
    
    return [mat, shot_text]

# --- 9. Запуск ---
ani = animation.FuncAnimation(fig, update, interval=40, blit=True, save_count=2000)

plt.show()