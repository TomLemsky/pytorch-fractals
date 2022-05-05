import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import colors, animation
from scipy.signal import convolve2d

overflow_kernel = np.array([
    [0,1,0],
    [1,0,1],
    [0,1,0]])
cmap = colors.ListedColormap(['b','g','y','r'])



n = 101
grid = np.zeros((n,n), dtype=int)
iterations = 10000

fig = plt.figure()
ax = plt.axes()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
plt.axis('off')
fig.set_size_inches(5, 5, True)
plt.rcParams['savefig.bbox'] = 'tight'
img = ax.imshow(grid, cmap=cmap, vmin=0, vmax=4, animated=True)
print(type(img))

def init():
    grid = np.zeros((n,n), dtype=int)
    img.set_array(grid)
    return (img,)

def animate(frame):
    global grid
    print(frame)
    #grid = np.zeros((n,n), dtype=int)
    #for i in range(frame):
    grid[n//2,n//2] += 1
    overflow = (grid >= 4)
    grid -= 4*overflow
    grid += convolve2d(overflow, overflow_kernel, mode='same')
    img.set_array(grid)
    return (img,)


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=5000, interval=4, blit=True, save_count=200)
mywriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
anim.save('basic_animation.mp4', writer=mywriter)
