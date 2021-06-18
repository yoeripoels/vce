"""Visualizing the MNIST-augmentation process
"""
import numpy as np
from data.mnist.augment import threshold_image, thin_image, to_graph, get_starting_points, split_strokes, \
    map_strokes_to_image

colors = [(27, 158, 119), (217, 95, 2), (117, 112, 179), (231, 41, 138), (102, 166, 30), (230, 171, 2), (166, 118, 29), (102, 102, 102)]


def create_stroke_overlay(stroke_map):
    h, w = stroke_map.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if stroke_map[y][x] != -1:
                color_idx = stroke_map[y][x]
                if color_idx >= len(colors):
                    color_idx = color_idx % (len(colors))
                r, g, b = colors[color_idx]
                overlay[y][x][0], overlay[y][x][1], overlay[y][x][2] = r, g, b
                overlay[y][x][3] = 0.5
    return overlay


def overlay_image(image, stroke_map):
    rgba_image = np.repeat(image[:, :, np.newaxis], 4, axis=2)
    rgba_image[:, :, 3] = 255
    overlay = create_stroke_overlay(stroke_map)
    combined_image = rgba_image * 0.5 + overlay * 0.5
    h, w, _ = combined_image.shape
    for y in range(h):
        for x in range(w):
            alpha = combined_image[y][x][3]
            offset = 255 / alpha # times this to fix
            combined_image[y][x][3] = combined_image[y][x][3] * offset
    combined_image = combined_image.astype(np.uint8)
    return combined_image


def visualize_process(image):
    thresh = threshold_image(image)
    thin = thin_image(thresh)
    graph = to_graph(thin)
    starting_points = get_starting_points(graph)
    # get image of starting points
    starting_image = thin.copy()
    starting_image_rgb = np.zeros((*thin.shape[0:2], 3))
    for y in range(thin.shape[0]):
        for x in range(thin.shape[1]):
            for i in range(3):
                starting_image_rgb[y][x][i] = thin[y][x]
            if (y, x) in starting_points:
                starting_image_rgb[y][x][0] = 1
                starting_image_rgb[y][x][1] = 0
                starting_image_rgb[y][x][2] = 0
    for (y, x) in starting_points:
        starting_image[y][x] = 128
    stroke, angles = split_strokes(graph, starting_points)
    stroke_map_thin = map_strokes_to_image(thin, stroke)
    stroke_image_thin = overlay_image(thin, stroke_map_thin)
    stroke_map = map_strokes_to_image(image, stroke)
    stroke_image = overlay_image(image, stroke_map)
    return [image, thresh, thin, starting_image_rgb, stroke_image_thin, stroke_image]


