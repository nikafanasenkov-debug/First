import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
from scipy import ndimage

# --- Настройки ---
N = 150
kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

def step(grid):
    n = convolve2d(grid, kernel, mode='same', boundary='fill', fillvalue=0)
    b = (grid == 0) & (n == 3)
    s = (grid == 1) & ((n == 2) | (n == 3))
    return (b | s).astype(int)

def get_rgb(grid):
    rgb = np.zeros((N, N, 3))
    rgb[:, :] = [0.04, 0.04, 0.08]
    lbl, num = ndimage.label(grid, structure=np.ones((3,3)))
    cmap = plt.cm.tab20
    for i in range(1, num+1):
        mask = (lbl == i)
        color = cmap((i-1) % 20)[:3]
        rgb[mask] = np.clip(np.array(color)*1.5, 0, 1)
    return rgb

# --- RLE декодер (гарантирует правильные паттерны) ---
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

# --- Паттерны из LifeWiki (точные RLE) ---
# Все три космических корабля летят ВПРАВО
GLIDER = rle_to_grid('bo$2bo$3o!')              # летит по диагонали
LWSS   = rle_to_grid('bo2bo$o4b$o3bo$4o!')      # летит вправо
MWSS   = rle_to_grid('3bo2b$bo3bo$o5b$o4bo$5o!')# летит вправо
HWSS   = rle_to_grid('3b2o2b$bo4bo$o6b$o5bo$6o!')# летит вправо
LWSS = np.fliplr(LWSS)
MWSS = np.fliplr(MWSS)
HWSS = np.fliplr(HWSS)

def run_ship(pattern, title, filename, frames=150):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Форма паттерна: {pattern.shape}  (высота x ширина)")
    print(f"  Живых клеток:   {pattern.sum()}")
    print(f"  Паттерн:\n{pattern}\n")

    initial_grid = np.zeros((N, N))
    h, w = pattern.shape
    y0 = N//2 - h//2
    x0 = 5
    initial_grid[y0:y0+h, x0:x0+w] = pattern

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0a16')
    ax.set_facecolor('#0a0a16')
    ax.axis('off')
    ax.set_title(title, color='#00ffff', fontsize=16,
                 fontfamily='monospace', fontweight='bold', pad=20)

    grid = initial_grid.copy()
    mat = ax.imshow(get_rgb(grid), animated=True)

    def update(frame):
        nonlocal grid
        grid = step(grid)
        mat.set_data(get_rgb(grid))
        return [mat]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=80, blit=True)
    ani.save(filename, writer='pillow', fps=15)
    print(f"  Сохранено: {filename}")

    # !!! Сбрасываем на НАЧАЛЬНОЕ состояние, чтобы окно показало старт, а не конец !!!
    grid = initial_grid.copy()
    mat.set_data(get_rgb(grid))

    def update_disp(frame):
        nonlocal grid
        grid = step(grid)
        mat.set_data(get_rgb(grid))
        return [mat]

    ani2 = animation.FuncAnimation(fig, update_disp, frames=frames,
                                    interval=80, blit=True, repeat=False)
    plt.show()  # ЖДЕТ, пока вы не закроете окно
    plt.close(fig)

if __name__ == "__main__":
    ships = [
        (GLIDER, "1/4: ГЛАЙДЕР (5 клеток, вниз-вправо)", "ship_1_glider.gif"),
        (LWSS,   "2/4: LWSS (9 клеток, вправо)",        "ship_2_lwss.gif"),
        (MWSS,   "3/4: MWSS (11 клеток, вправо)",       "ship_3_mwss.gif"),
        (HWSS,   "4/4: HWSS (13 клеток, вправо)",       "ship_4_hwss.gif"),
    ]
    for pat, title, fn in ships:
        run_ship(pat, title, fn)
    print("\n" + "="*60)
    print("  ВСЕ 4 КОРАБЛЯ ПОКАЗАНЫ!")
    print("="*60)