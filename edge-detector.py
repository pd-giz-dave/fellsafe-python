from PIL import Image, ImageDraw
import numpy as np
from math import sqrt

# Load image:
input_image = Image.open("blob-0-photo-365-oblique.jpg")
input_pixels = input_image.load()
width, height = input_image.width, input_image.height

# Create output image
output_image = Image.new("RGB", input_image.size)
draw = ImageDraw.Draw(output_image)

# Convert to grayscale
intensity = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        #intensity[x, y] = sum(input_pixels[x, y]) / 3    # colour image
        intensity[x, y] = input_pixels[x, y]             # b/w image

# Compute convolution between intensity and kernels
max_colour = 0
min_colour = 255
buf = [[0 for _ in range(input_image.height)] for _ in range(input_image.width)]
for x in range(1, input_image.width - 1):
    for y in range(1, input_image.height - 1):
        """ this is a sobel on
                      0,  0,  0            0, -1,  0
                Gx = -1,  0, +1  and Gy =  0,  0,  0
                      0,  0,  0            0, +1,  0
            """
        magx = 0 #intensity[x + 1, y] - intensity[x - 1, y]
        magy = intensity[x, y + 1] - intensity[x, y - 1]

        # Draw in black and white the magnitude
        color = int(sqrt(magx ** 2 + magy ** 2))
        if color > max_colour:
            max_colour = color
        if color < min_colour:
            min_colour = color
        buf[x][y] = color
scale = (max_colour - min_colour) >> 1
print('Scale {}, max {}, min {}'.format(scale,max_colour,min_colour))
for x in range(1, input_image.width - 1):
    for y in range(1, input_image.height - 1):
        color = min(int(((buf[x][y] - min_colour) / scale) * 256), 255)
        draw.point((x, y), (color, color, color))

output_image.save("edge.jpg")
output_image.show()
