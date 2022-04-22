import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.utils import save_image

def mandelbrot(center, diameter, resolution):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # complex64 is way faster, but less accurate.
    # Could it be faster not to use the complex type at all?
    float_type = torch.float64 
    complex_float_type = torch.complex128

    x_center = center[0]
    y_center = center[1]

    x = torch.linspace(x_center-diameter, x_center+diameter, resolution, dtype=float_type)
    y = torch.linspace(y_center-diameter, y_center+diameter, resolution, dtype=float_type)

    xv, yv = torch.meshgrid(x, y)

    # complex matrix defined by coordinates
    C = xv + 1j*yv

    current = torch.zeros_like(C, dtype=complex_float_type)
    output  = torch.zeros_like(C, dtype=torch.int64)

    C = C.to(device)
    current = current.to(device)
    output = output.to(device)

    for i in range(1,200):
        current = current**2 + C

        output[torch.absolute(current)>2*2] = i

    output = torch.swapaxes(output,0,1)
    return output.cpu()

#output = np.absolute(current)
#output = np.clip(output,0,5)



#image = mandelbrot(center=(-0.7,0), diameter=3, resolution=2000)

center = (-0.743643887037158704752191506114774, 0.131825904205311970493132056385139)

for i in range(1,200):
    print(i)
    image = mandelbrot(center=center, diameter=3/(1.1**i), resolution=2000)
    image = image/image.max()
    save_image(image, f"mandelbrot_animation/{i:04}.png")
