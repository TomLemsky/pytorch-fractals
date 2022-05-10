import numpy as np
import torch
import torch.nn.functional as func
from torchvision.utils import save_image
from matplotlib import colors
from PIL import Image
import matplotlib.pyplot as plt

cmap = colors.ListedColormap(['b','g','y','r'])

overflow_kernel = torch.tensor([
    [0,1,0],
    [1,0,1],
    [0,1,0]])

overflow_kernel = overflow_kernel.view((1,1,3,3))

save_every = 100
n = 250
iterations = 100000

grid = torch.zeros((1,1,n,n), dtype=int)

for i in range(iterations):
    grid[:,:,n//2,n//2] += 1
    # numbers four or higher spill over to their neighbors
    overflow = (grid >= 4).long()
    # decreae overflowing pixels
    grid -= 4*overflow
    # add 1 to neighbors of overflowing pixels
    grid += func.conv2d(overflow, overflow_kernel,stride=(1,), padding=(1,)).long()
    if i%save_every==0:
        print(f"{i:08}/{iterations}")
        colored_image = cmap(grid[0,0,:,:])
        plt.imsave(f"sandpile_animation/{i//save_every:08}.png",colored_image)
        #save_image(torch.Tensor(colored_image), )
