"""Functions to augment MNIST to separate digits
"""
import cv2
import numpy as np
from numba import jit
import math
import random


###########################
# HELPER VARIABLES / FUNCTIONS
###########################
max_dif = 45/180 * math.pi


def calc_angle(rect):
    y0, x0 = rect[0]
    y1, x1 = rect[-1]
    return math.atan2(y1-y0, x1-x0)


def get_bbox(points):
    x0, x1 = math.inf, -math.inf
    y0, y1 = math.inf, -math.inf
    for (y, x) in points:
        x0 = min(x, x0)
        x1 = max(x, x1)
        y0 = min(y, y0)
        y1 = max(y, y1)
    return (y0, x0), (y1, x1)


def dist(p0, p1):
    y0, x0 = p0
    y1, x1 = p1
    return math.sqrt((x0-x1)**2 + (y0-y1)**2)


###########################
# IMAGE PRE-PROCESS FUNCTIONS (IMAGE -> GRAPH)
###########################


# image thresholding (as done in https://arxiv.org/abs/1611.03068)
def threshold_image(img):
    # threshold until number of connected components changes
    num_con_4, _ = cv2.connectedComponents(img, connectivity=4)
    num_con_8, _ = cv2.connectedComponents(img, connectivity=8)
    num_pixels_min = round(cv2.countNonZero(img) / 2)
    done = False
    step_size = 25
    threshold = 0
    _, c_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    while not done:
        # increase threshold
        threshold += step_size
        _, new_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        num_con_4_new, _ = cv2.connectedComponents(new_img, connectivity=4)
        num_con_8_new, _ = cv2.connectedComponents(new_img, connectivity=8)
        num_pixels_new = cv2.countNonZero(new_img)
        # verify we still fit the conditions
        if (num_pixels_new < num_pixels_min) or (num_con_4 != num_con_4_new or num_con_8 != num_con_8_new) or \
                threshold >= 250:
            done = True
            break
        c_img = new_img
    return c_img


# zhang-suen thinning (https://rosettacode.org/wiki/Zhang-Suen_thinning_algorithm)
# adapted from https://github.com/bsdnoobz/zhang-suen-thinning
@jit(nopython=True)
def _thinningIteration(im, iter_type):
    M = im.copy()
    h, w = im.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            p2 = im[i - 1][j]
            p3 = im[i - 1][j + 1]
            p4 = im[i][j + 1]
            p5 = im[i + 1][j + 1]
            p6 = im[i + 1][j]
            p7 = im[i + 1][j - 1]
            p8 = im[i][j - 1]
            p9 = im[i - 1][j - 1]
            A = int(p2 == 0 and p3 == 1) + int(p3 == 0 and p4 == 1) + int(p4 == 0 and p5 == 1) + int(
                p5 == 0 and p6 == 1) + int(p6 == 0 and p7 == 1) + int(p7 == 0 and p8 == 1) + int(
                p8 == 0 and p9 == 1) + int(p9 == 0 and p2 == 1)
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            if iter_type == 0:
                m1_white = p2 * p4 * p6 == 0
                m2_white = p4 * p6 * p8 == 0
            else:
                m1_white = p2 * p4 * p8 == 0
                m2_white = p2 * p6 * p8 == 0
            if im[i][j] == 1 and A == 1 and 2 <= B <= 6 and m1_white and m2_white:
                M[i][j] = 0
    return M


def thin_image(src):
    dst = src.copy() / 255
    prev = np.zeros(src.shape[:2], np.uint8)
    while True:
        dst = _thinningIteration(dst, 0)
        dst = _thinningIteration(dst, 1)
        diff = np.absolute(dst - prev)
        prev = dst.copy()
        if np.sum(diff) == 0:
            break

    return dst * 255


# graph representation of image
def to_graph(img):
    # convert white pixels to (x,y) and connect to neighbours
    graph = {}
    h, w = img.shape
    surrounding = [-1, 0, 1]
    for y in range(h):
        for x in range(w):
            if img[y][x] > 0:  # found candidate pixel
                neighbours = set()
                # check surrounding pixels
                for ex_y in surrounding:
                    for ex_x in surrounding:
                        if ex_y == 0 and ex_x == 0:
                            continue
                        new_y = y+ex_y
                        new_x = x+ex_x
                        if 0 <= new_y < h and 0 <= new_x < w:
                            if img[new_y][new_x] > 0:
                                neighbours.add((new_y, new_x))
                graph[(y, x)] = neighbours
    return graph


###########################
# STROKE-CREATION FUNCTIONS
###########################


# get starting points to draw strokes from
def get_starting_points(graph, delete_outlier=True):
    # rules: all neighbours must include the given pixel + another neighbour
    # outliers: both neighbours connect to each other and other pixels, can remove pixel
    starting_points = set()
    for pixel, neighbours in graph.items():
        if len(neighbours) > 1:
            continue
        for neighbour in neighbours:
            n_n = graph[neighbour]
            # mask out irrelevant pixels
            other_n = n_n.copy()
            other_n.remove(pixel)
            own_n = neighbours.copy()
            own_n.remove(neighbour)
            # verify subset and smaller
            if own_n.issubset(other_n) and len(own_n) < len(other_n):
                # extra check: neighbour must not have 2+ other neighbours (or it's an outlier)
                if len(other_n) < 2 or not delete_outlier:
                    starting_points.add(pixel)
                    break
    if len(starting_points) == 0 and delete_outlier:  # none found, try with less restrictions
        starting_points = get_starting_points(graph, delete_outlier=False)
    if len(starting_points) == 0: # still none found
        # pick with most neighbours
        max_neighbour = 0
        for neighbours in graph.values():
            max_neighbour = max(max_neighbour, len(neighbours))
        for pixel, neighbours in graph.items():
            if len(neighbours) == max_neighbour:
                starting_points.add(pixel)
    return starting_points


def clean_up(graph, just_marked):
    max_cluster = 3  # if pixel groups of x or less are 'surrounded' by just marked pixels, add them to cluster
    clusters = []
    for point, neighbours in graph.items():
        if point in just_marked:
            continue  # skip just marked pixel
        for n in neighbours:
            if n in just_marked:  # found candidate
                clusters.append(({point}, {point}))  # total cluster and just added
    for i in range(max_cluster - 1):  # for cluster size
        for j in range(len(clusters)):  # go through all the clusters
            points, to_expand = clusters[j]
            just_added = set()  # refresh just added
            for p in to_expand:
                for n in graph[p]:  # get all neighbours
                    if n in just_marked or n in points:  # skip already marked
                        continue
                    points.add(n)  # add new points
                    just_added.add(n)
            clusters[j] = (points, just_added)
    # after we are done expanding, check if 'just added' has any neighbours not in cluster yet and not marked
    to_add = set()
    for i in range(len(clusters)):
        points, just_added = clusters[i]
        can_add = True
        for p in just_added:
            for n in graph[p]:
                if n not in just_marked and n not in points:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            for p in points:
                to_add.add(p)  # add all points
    return to_add


def get_stroke(graph, already_seen, starting_point):
    # filter out unnecessary points, don't mark double
    graph_use = {}
    for key, value in graph.items():
        if key not in already_seen:
            # compute neighbours that we have not seen
            neighbour = set()
            for n in value:
                if n not in already_seen:
                    neighbour.add(n)
            graph_use[key] = neighbour
    # start making line
    # outlier: no neighbours
    if len(graph_use[starting_point]) == 0:
        return {starting_point}, []
    # get random neighbour as we have no info yet (usually starting point only has 1)
    next_p = random.sample(graph_use[starting_point], 1)[0]
    cur_points = {starting_point, next_p}
    cur_rect = [starting_point, next_p]
    # add next point to rect to make 3, then start iterating (moving avg over 3 points)
    cur_angle = calc_angle(cur_rect)
    all_angles = [cur_angle]
    min_dif = math.inf
    sel_angle = math.inf
    candidate_point = -1
    for n in graph_use[next_p]:
        if n in cur_points:
            continue  # don't use points twice
        new_angle = calc_angle((cur_rect[0], n))
        dif = abs(cur_angle - new_angle)
        if dif < min_dif:
            min_dif = dif
            candidate_point = n
            sel_angle = new_angle
    if candidate_point == -1:  # no new neighbours to add
        return cur_points, all_angles
    cur_rect.append(candidate_point)
    cur_points.add(candidate_point)
    all_angles.append(sel_angle)
    # have first rect, now keep repeating until angle dif too big
    can_extend = True
    sel_angle = math.inf
    while can_extend:  # loop until we can't find
        # see if we can find candidate
        last_point = cur_rect[-1]
        cur_angle = calc_angle(cur_rect)
        min_dif = max_dif
        candidate_point = -1
        for n in graph_use[last_point]:
            if n in cur_points:
                continue  # don't repeat
            new_angle = calc_angle((cur_rect[1], n))
            dif = abs(cur_angle - new_angle)
            if dif < min_dif:
                min_dif = dif
                candidate_point = n
                sel_angle = new_angle
        if candidate_point != -1:  # we found next point
            cur_points.add(candidate_point)
            del cur_rect[0]
            cur_rect.append(candidate_point)
            all_angles.append(sel_angle)
        else:
            can_extend = False
    return cur_points, all_angles


def merge_strokes(graph, strokes):
    # deep copy strokes for safety
    new_s = []
    for s in strokes:
        new_s.append(s.copy())
    strokes = new_s

    # mark for to-add
    to_merge = []  # (i, j) where we merge i into j
    for i, s in enumerate(strokes):
        if len(s) <= 3:  # 3 or smaller get swalled up in bigger strokes if possible
            # check neighbouring strokes (if exist)
            candidates = set()
            for p in s:
                for n in graph[p]:
                    for j, s_j in enumerate(strokes):  # check other strokes
                        if j == i:
                            continue  # don't try to merge with itself or other too small stroke
                        if n in s_j:  # found candidate
                            candidates.add(j)
            # now that we have candidates, select best fitting (first priority for big stroke, then smallest angle)
            min_dif = math.inf
            cur_angle = calc_angle(get_bbox(s))
            candidate_stroke = -1
            large_stroke = False
            for c_i in candidates:
                stroke_candidate = strokes[c_i]
                new_angle = calc_angle(get_bbox(stroke_candidate))
                dif = abs(cur_angle - new_angle)
                if dif < min_dif or (not large_stroke and len(stroke_candidate) > 3):
                    min_dif = dif
                    candidate_stroke = c_i
                    if len(stroke_candidate) > 3:
                        large_stroke = True
            if candidate_stroke != -1:
                to_merge.append((i, candidate_stroke))
    # create new strokes
    new_strokes = []
    to_merge_total = [x for (x, y) in to_merge]
    for i, s in enumerate(strokes):
        if i in to_merge_total:  # ignore strokes we want to delete
            continue
        cur_s = s.copy()  # start handling stroke
        for to_delete, to_add_to in to_merge:  # check if we are merging into this
            if i == to_add_to:  # merge
                cur_s = cur_s.union(strokes[to_delete])
        new_strokes.append(cur_s)

    if len(strokes) != len(new_strokes):  # we made changes, see if we can remove more
        new_strokes = merge_strokes(graph, new_strokes)
    return new_strokes


def split_strokes(graph, starting_points):
    # pick random starting point
    start = random.sample(starting_points, 1)[0]
    already_seen = set()
    done = False
    strokes = []
    angle_map = []
    while not done:
        # handle stroke
        stroke, angles = get_stroke(graph, already_seen, start)
        # map angle to a point in stroke so we can easily map back
        angle_map.append((angles, next(iter(stroke))))

        # clean noise
        extra_add = clean_up(graph, stroke)
        full_stroke = stroke.union(extra_add - already_seen)  # make sure we don't double add points
        # stroke done
        strokes.append(full_stroke)
        already_seen = already_seen.union(full_stroke)  # save points in already added
        # check if we are done, i.e., all pixels marked
        if len(already_seen) == len(graph.keys()):
            done = True
            break

        # not done: find new starting point
        starting_points = starting_points - already_seen
        if len(starting_points) > 0:
            start = random.sample(starting_points, 1)[0]
        else:
            # pick random connected pixel
            found = False
            for p in already_seen:
                for n in graph[p]:  # neighbours of pixel
                    if n not in already_seen:  # new pixel
                        found = True
                        start = n
            if not found:  # all connected are already handled: different piece not marked as starting point (outlier)
                # pick random outlier and start from there
                start = random.sample(graph.keys() - already_seen, 1)[0]
    # merge tiny strokes, i.e. remove noise

    strokes = merge_strokes(graph, strokes)

    # map angles to strokes
    angle_stroke = []
    for i in range(len(strokes)):
        angle_stroke.append([])
    for i in range(len(angle_map)):
        angles, point = angle_map[i]
        for j in range(len(strokes)):
            stroke = strokes[j]
            if point in stroke:
                angle_stroke[j].append(angles)
    return strokes, angle_stroke


def map_strokes_to_image(img, strokes):
    h, w = img.shape
    stroke_map = np.full((h, w), -1, dtype=np.int8)
    for y in range(h):
        for x in range(w):
            if img[y][x] == 0:
                continue  # skip black pixels
            nearest_stroke = -1
            min_dist = math.inf
            for i, s in enumerate(strokes):
                for p in s:
                    cur_dist = dist((y, x), p)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        nearest_stroke = i
            stroke_map[y][x] = nearest_stroke
    return stroke_map


def hide_stroke(img, stroke_map, idx):
    if not isinstance(idx, list):
        idx = [idx]
    max_stroke = np.amax(stroke_map)
    for i in idx:
        if i < 0 or i > max_stroke:
            raise ValueError('Invalid index specified')
    out_img = img.copy()
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if stroke_map[y][x] in idx:
                out_img[y][x] = 0
    return out_img


def highlight_strokes(img, stroke_map, idx):
    if not isinstance(idx, list):
        idx = [idx]
    max_stroke = np.amax(stroke_map)
    for i in idx:
        if i < 0 or i > max_stroke:
            raise ValueError('Invalid index specified')
    out_img = img.copy()
    h, w = img.shape
    for y in range(h):
        for x in range(w):
            if stroke_map[y][x] not in idx:
                out_img[y][x] = 0
    return out_img


def get_strokes(img):
    """Input = image
    Output = Strokes
    """
    img = threshold_image(img)
    img = thin_image(img)
    graph = to_graph(img)
    starting_points = get_starting_points(graph)
    stroke, angles = split_strokes(graph, starting_points)
    return stroke, angles


def split_digit(img):
    """Input = image
    Output = Stroke map
    """
    stroke, angles = get_strokes(img)
    stroke_map = map_strokes_to_image(img, stroke)
    return stroke_map
