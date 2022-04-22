import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.utils import save_image

def buddhabrot(center, diameter, resolution, iterations=20):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # complex64 is way faster, but less accurate.
    # Could it be faster not to use the complex type at all?
    float_type = torch.float64
    complex_float_type = torch.complex128

    x_center = center[0]
    y_center = center[1]

    pixel_width = diameter/resolution

    x = torch.linspace(x_center-diameter, x_center+diameter, resolution, dtype=float_type)
    y = torch.linspace(y_center-diameter, y_center+diameter, resolution, dtype=float_type)

    xv, yv = torch.meshgrid(x, y)
    # randomize within the pixel's box
    xv = xv + torch.rand_like(xv)*pixel_width
    yv = yv + torch.rand_like(yv)*pixel_width

    # complex matrix defined by coordinates
    C = xv + 1j*yv

    current = torch.zeros_like(C, dtype=complex_float_type)
    past_positions = []
    #output  = torch.zeros_like(C, dtype=torch.int64)
    counts  = torch.zeros(resolution**2)

    C = C.to(device)
    current = current.to(device)

    #output = output.to(device)
    counts  = counts.to(device)

    for i in range(1,iterations+1): #200):
        current = current**2 + C
        past_positions.append(current)

    for i,p in enumerate(past_positions):
        print(i)
        # zero out pixels that don't escape
        p[torch.absolute(current)<2*2] = 0

        # align all pixels so that the pixel with smallest Real and Imag part is the first in the matrix
        p_aligned = p - (x_center-diameter+1j*(y_center-diameter))
        # make everything integers
        p_aligned = p_aligned/(2*pixel_width)
        # round to integer coordinates and restrict to image size
        real = torch.clamp(p_aligned.real.int(),min=0,max=resolution)
        imag = torch.clamp(p_aligned.imag.int(),min=0,max=resolution)
        # assign a number for each coordinate
        coords = real + imag*resolution
        # count occurances of each coordinate
        counts += torch.bincount(coords.flatten(), weights=None, minlength=resolution**2)[:resolution**2]

    return counts.cpu()

#output = np.absolute(current)
#output = np.clip(output,0,5)

resolution = 1000
center = (-0.7,0) #(-0.451,0) #
diameter = 2 # 3
iterations = 20
samples_per_pixel = 20
counts  = torch.zeros(resolution**2)
for i in range(1,samples_per_pixel):
    print(f"{i}-th round of samples")
    counts += buddhabrot(center=center, diameter=diameter, resolution=resolution,iterations=iterations)

    image = counts.reshape(resolution,resolution)
    save_image(image, f"buddhabrot.png")
