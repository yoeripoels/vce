"""CUDA/GPU-accelerated functions to convert shape data to images.
"""
import math
from numba import cuda


# creating a single image
@cuda.jit(device=True)
def create_image_kernel(shapes, sf, sf_base, output, index):
    h = output.shape[1]
    w = output.shape[2]
    NUM_LINES = shapes.shape[1]
    MAX_POINTS = sf.shape[1]
    EDGE_FALLOFF = 0.02  # constant for now

    for y in range(h):
        for x in range(w):
            # normalize
            n_x = x / w
            n_y = y / h
            min_distance = 2  # anything above ~1.41 works
            # find closest line
            for i in range(NUM_LINES):
                l_x1 = shapes[index][i][0]  # no tuple assignment in nopython :(
                l_y1 = shapes[index][i][1]
                l_x2 = shapes[index][i][2]
                l_y2 = shapes[index][i][3]
                if l_x1 == -1:  # wildcard, no more lines to parse
                    break

                # check if point is in between two points of line
                t = (l_x1 - n_x) * (l_x2 - l_x1) + (l_y1 - n_y) * (l_y2 - l_y1)
                t /= -(math.pow(l_x2 - l_x1, 2) + math.pow(l_y2 - l_y1, 2))
                # if in between, get perpendicular distance
                if 0 <= t <= 1:
                    distance = abs(
                        (l_y2 - l_y1) * n_x - (l_x2 - l_x1) * n_y + l_x2 * l_y1 - l_y2 * l_x1)
                    distance /= math.sqrt(math.pow(l_y2 - l_y1, 2) + math.pow(l_x2 - l_x1, 2))
                else:
                    # otherwise, return distance to closest point
                    distance = min(math.sqrt(math.pow(l_x2 - n_x, 2) + math.pow(l_y2 - n_y, 2)),
                                   math.sqrt(math.pow(l_x1 - n_x, 2) + math.pow(l_y1 - n_y, 2)))
                min_distance = min(min_distance, distance)

            # calculate stroke for this position
            weight_sum = sf_base[index][1]
            stroke_sum = sf_base[index][0] * weight_sum
            for i in range(MAX_POINTS):
                # calculate addition per point
                x1 = sf[index][i][0]
                y1 = sf[index][i][1]
                s = sf[index][i][2]
                thickness = sf[index][i][3]
                exp = sf[index][i][4]
                if x1 == -1:  # wildcard, no more points to parse
                    break
                stroke_distance = math.sqrt(math.pow(x1 - n_x, 2) + math.pow(y1 - n_y, 2))
                strength = s * math.exp(-exp * stroke_distance)
                weight_sum += strength
                stroke_sum += strength * thickness
            stroke = stroke_sum / weight_sum

            # calculate final brightness for this pixel
            if min_distance < (stroke + EDGE_FALLOFF):
                if min_distance < stroke:
                    output[index][y][x] = 1
                else:
                    output[index][y][x] = (EDGE_FALLOFF - (min_distance - stroke)) / EDGE_FALLOFF
            else:
                output[index][y][x] = 0

# creating all images
@cuda.jit
def create_images(shapes, sf, sf_base, output):
    N = shapes.shape[0]
    startx = cuda.blockDim.x * cuda.blockIdx.x + cuda.threadIdx.x
    starty = cuda.blockDim.y * cuda.blockIdx.y + cuda.threadIdx.y
    gridx = cuda.gridDim.x * cuda.blockDim.x
    gridy = cuda.gridDim.y * cuda.blockDim.y
    for i in range(startx, N, gridx):
        if starty == 0:
            create_image_kernel(shapes, sf, sf_base, output, i)
