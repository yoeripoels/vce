"""Classes/functions used to generate synthetic data
"""
import math
import numpy as np
import random
import util.vis as vis

class LineSegment:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def distance(self, x0, y0):
        # check if point is in between two points of line
        t = (self.x1 - x0) * (self.x2 - self.x1) + (self.y1 - y0) * (self.y2 - self.y1)
        t /= -(math.pow(self.x2 - self.x1, 2) + math.pow(self.y2 - self.y1, 2))
        # if in between, get perpendicular distance
        if 0 <= t <= 1:
            distance = abs((self.y2 - self.y1) * x0 - (self.x2 - self.x1) * y0 + self.x2 * self.y1 - self.y2 * self.x1)
            distance /= math.sqrt(math.pow(self.y2 - self.y1, 2) + math.pow(self.x2 - self.x1, 2))
        else:
            # otherwise, return distance to closest point
            distance = min(math.sqrt(math.pow(self.x2 - x0, 2) + math.pow(self.y2 - y0, 2)),
                           math.sqrt(math.pow(self.x1 - x0, 2) + math.pow(self.y1 - y0, 2)))
        return distance

    def __repr__(self):
        return '[({},{})-({},{})]'.format(self.x1, self.y1, self.x2, self.y2)

    def return_data(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])


class Shape:
    def __init__(self, name=''):
        self.name = name
        self.points = []
        self.lines = []
        self.new_line = set()

    def add_point(self, x1, y1, new_line=False):
        if new_line:
            self.new_line.add(len(self.points))
        self.points.append((x1, y1))
        if not new_line and len(self.points) >= 2:
            self.lines.append(LineSegment(*self.points[-2], *self.points[-1]))

    def rebuild_lines(self):
        self.lines = []
        for i in range(len(self.points) - 1):
            if i not in self.new_line:
                self.lines.append(LineSegment(*self.points[i], *self.points[i + 1]))

    def delete_point(self, index):
        if index < 0 or index > len(self.lines) - 1:
            raise ValueError('Index to delete out of bounds')
        del self.points[index]
        self.rebuild_lines()

    def distort_norm_range(self, vf, amount):
        mc = vf.max_contribution()
        o_min_x, o_max_x, o_min_y, o_max_y = math.inf, -math.inf, math.inf, -math.inf
        n_min_x, n_max_x, n_min_y, n_max_y = math.inf, -math.inf, math.inf, -math.inf
        for (x, y) in self.points:
            vx, vy = vf.contribution(x, y)
            nx = x + vx / mc * amount
            ny = y + vy / mc * amount
            o_min_x = min(o_min_x, x)
            o_max_x = max(o_max_x, x)
            o_min_y = min(o_min_y, y)
            o_max_y = max(o_max_y, y)
            n_min_x = min(n_min_x, nx)
            n_max_x = max(n_max_x, nx)
            n_min_y = min(n_min_y, ny)
            n_max_y = max(n_max_y, ny)
        return o_min_x, o_max_x, o_min_y, o_max_y, n_min_x, n_max_x, n_min_y, n_max_y

    def distort(self, vf, amount, normalize=False, normalize_info=None):
        mc = vf.max_contribution()
        distorted_shape = Shape(self.name + '_distort')
        if normalize:
            o_min_x, o_max_x, o_min_y, o_max_y = math.inf, -math.inf, math.inf, -math.inf
            n_min_x, n_max_x, n_min_y, n_max_y = math.inf, -math.inf, math.inf, -math.inf
        for i, (x, y) in enumerate(self.points):
            vx, vy = vf.contribution(x, y)
            nx = x + vx / mc * amount
            ny = y + vy / mc * amount
            distorted_shape.add_point(nx, ny, i in self.new_line)
            if normalize:
                o_min_x = min(o_min_x, x)
                o_max_x = max(o_max_x, x)
                o_min_y = min(o_min_y, y)
                o_max_y = max(o_max_y, y)
                n_min_x = min(n_min_x, nx)
                n_max_x = max(n_max_x, nx)
                n_min_y = min(n_min_y, ny)
                n_max_y = max(n_max_y, ny)
        if not normalize:
            return distorted_shape
        else:
            distorted_shape_normal = Shape(distorted_shape.name)
            if normalize_info:
                o_min_x, o_max_x, o_min_y, o_max_y, n_min_x, n_max_x, n_min_y, n_max_y = normalize_info
            o_xrange = o_max_x - o_min_x
            n_xrange = n_max_x - n_min_x
            o_yrange = o_max_y - o_min_y
            n_yrange = n_max_y - n_min_y
            for i, (x, y) in enumerate(distorted_shape.points):
                xp = (x - n_min_x) / n_xrange * o_xrange + o_min_x
                yp = (y - n_min_y) / n_yrange * o_yrange + o_min_y
                distorted_shape_normal.add_point(xp, yp, i in self.new_line)
            return distorted_shape_normal

    def rotate(self, angle, cx=0.5, cy=0.5):
        rotated_shape = Shape(self.name + '_rotate')
        for i, (x, y) in enumerate(self.points):
            px = math.cos(angle) * (x - cx) - math.sin(angle) * (y - cy) + cx
            py = math.sin(angle) * (x - cx) + math.cos(angle) * (y - cy) + cy
            rotated_shape.add_point(px, py, i in self.new_line)
        return rotated_shape


class VectorPoint:
    def __init__(self, x1, y1, vx, vy, s, normalize=True):
        self.x1, self.y1, self.vx, self.vy, self.s = x1, y1, vx, vy, s
        if normalize:
            self.vx /= math.sqrt(vx * vx + vy * vy)
            self.vy /= math.sqrt(vx * vx + vy * vy)

    def contribution(self, x0, y0):
        distance = math.sqrt(math.pow(self.x1 - x0, 2) + math.pow(self.y1 - y0, 2))
        strength = self.s / (distance + 1)  # max out at s
        return self.vx * strength, self.vy * strength

    def __repr__(self):
        return '[p({},{}),v({},{}),s={}]'.format(self.x1, self.y1, self.vx, self.vy, self.s)


class VectorField:
    def __init__(self, name='', default_points=True):
        self.name = name
        self.points = []
        if default_points:
            self.add_point(-0.5, -0.5, 0, 0, 1, normalize=False)
            self.add_point(0.5, -0.5, 0, 0, 1, normalize=False)
            self.add_point(-0.5, 0.5, 0, 0, 1, normalize=False)
            self.add_point(0.5, 0.5, 0, 0, 1, normalize=False)

    def add_point(self, x1, y1, vx, vy, s, normalize=True):
        self.points.append(VectorPoint(x1, y1, vx, vy, s, normalize))

    def max_contribution(self):
        c = 0
        for p in self.points:
            c += p.s
        return c

    def contribution(self, x0, y0):
        vx, vy = 0, 0
        num = len(self.points)
        for p in self.points:
            pvx, pvy = p.contribution(x0, y0)
            vx += pvx
            vy += pvy
        return vx, vy


class StrokePoint:
    def __init__(self, x1, y1, s, t, exp):
        self.x1, self.y1, self.s, self.t, self.exp = x1, y1, s, t, exp

    def contribution(self, x0, y0):
        distance = math.sqrt(math.pow(self.x1 - x0, 2) + math.pow(self.y1 - y0, 2))
        strength = self.s * math.exp(-self.exp * distance)
        return strength

    def thickness(self):
        return self.t

    def __repr__(self):
        return '[p({},{}),s={},t={},exp={}]'.format(self.x1, self.y1, self.s, self.t, self.exp)

    def return_data(self):
        return np.array([self.x1, self.y1, self.s, self.t, self.exp])


class StrokeField:
    def __init__(self, name='', base_stroke=0.02, base_weight=1):
        self.name = name
        self.base_stroke = base_stroke
        self.base_weight = base_weight
        self.points = []

    def add_point(self, x1, y1, s, t, exp=2):
        self.points.append(StrokePoint(x1, y1, s, t, exp))

    def stroke(self, x0, y0):
        weight_sum = self.base_weight
        stroke = self.base_stroke * self.base_weight
        for p in self.points:
            s = p.contribution(x0, y0)
            weight_sum += s
            stroke += s * p.thickness()
        stroke /= weight_sum
        return stroke


class ShapeParser:
    # Default modification distributions (instance properties so can be overwritten)
    def rv(self):  # random vector from [0.5, 2] or [-2, -0.5]
        bottom_range, top_range = 0.5, 2
        if random.choice([True, False]):  # check if negative or positive
            return np.random.uniform(-top_range, -bottom_range)
        else:
            return np.random.uniform(bottom_range, top_range)

    rp = lambda _: np.random.uniform(0, 1)  # random position [0, 1]
    rw = lambda _: np.random.uniform(1, 2)  # random weight [1, 2]
    rwv = lambda _: np.random.uniform(3, 6)  # random weight for vectors
    re = lambda _: np.random.randint(2, 4)  # random exponent [2, 4]
    rs = lambda _: np.random.uniform(0.001, 0.05)  # random stroke [0.001, 0.05]
    rnp = lambda _: np.random.randint(3, 6)  # 3-5 points
    rd = lambda _: np.random.uniform(0.3, 0.6)  # random distortion amount
    ra = lambda _: np.random.normal(loc=0, scale=0.25) * (
                2 * math.pi) / 20  # random rotation, [-18°, 18°] = [-2SD, 2SD]

    def __init__(self, w=32, h=32):
        self.w = 32
        self.h = 32

    def random_modification(self, shape, w=None, h=None, no_image=False):
        w = w if w is not None else self.w
        h = h if h is not None else self.h
        vf = VectorField()
        n_v = self.rnp()
        for i in range(n_v):
            vf.add_point(self.rp(), self.rp(), self.rwv(), self.rv(), self.rv())
        distortion_amount = self.rd()
        distorted_shape = shape.distort(vf, distortion_amount, normalize=True)
        rot = self.ra()
        rotated_shape = distorted_shape.rotate(rot)
        base_stroke = self.rs()
        base_weight = self.rw() / 10
        sf = StrokeField(base_stroke=base_stroke, base_weight=base_weight)
        n_s = self.rnp()
        for i in range(n_s):
            stroke = self.rs()
            sf.add_point(self.rp(), self.rp(), self.rw(), stroke, exp=self.re())
        if no_image:
            return rotated_shape, sf
        else:
            image = self.shape_to_image(rotated_shape, w, h, sf=sf)
            return image

    def random_modification_multi(self, shapes, w=None, h=None, no_image=False):
        w = w if w is not None else self.w
        h = h if h is not None else self.h
        vf = VectorField()
        n_v = self.rnp()
        for i in range(n_v):
            vf.add_point(self.rp(), self.rp(), self.rwv(), self.rv(), self.rv())
        distortion_amount = self.rd()
        rotation_amount = self.ra()
        return_shapes = []

        # calculate normalization info to be shared between shapes
        o_min_x,   o_max_x,  o_min_y,   o_max_y,  n_min_x,   n_max_x,  n_min_y,   n_max_y = \
        math.inf, -math.inf, math.inf, -math.inf, math.inf, -math.inf, math.inf, -math.inf
        for in_shape in shapes:
            normalize_info = in_shape.distort_norm_range(vf, distortion_amount)
            o_min_x = min(o_min_x, normalize_info[0])
            o_max_x = max(o_max_x, normalize_info[1])
            o_min_y = min(o_min_y, normalize_info[2])
            o_max_y = max(o_max_y, normalize_info[3])
            n_min_x = min(n_min_x, normalize_info[4])
            n_max_x = max(n_max_x, normalize_info[5])
            n_min_y = min(n_min_y, normalize_info[6])
            n_max_y = max(n_max_y, normalize_info[7])
        normalize_info = o_min_x, o_max_x, o_min_y, o_max_y, n_min_x, n_max_x, n_min_y, n_max_y
        for in_shape in shapes:
            shape = in_shape.distort(vf, distortion_amount, normalize=True, normalize_info=normalize_info)
            shape = shape.rotate(rotation_amount)
            return_shapes.append(shape)

        base_stroke = self.rs()
        base_weight = self.rw() / 10
        sf = StrokeField(base_stroke=base_stroke, base_weight=base_weight)
        n_s = self.rnp()
        for i in range(n_s):
            stroke = self.rs()
            sf.add_point(self.rp(), self.rp(), self.rw(), stroke, exp=self.re())

        if no_image:
            return return_shapes, sf
        else:
            out_image = [self.shape_to_image(s, w, h, sf=sf) for s in return_shapes]
            return out_image

    def shape_to_image(self, shape, w=None, h=None, stroke=0.02, sf=None, edge_fade=0.02):
        w = w if w is not None else self.w
        h = h if h is not None else self.h
        image = np.zeros((h, w))
        for x in range(w):
            for y in range(h):
                # normalize
                n_x = x / w
                n_y = y / h
                dist = 2  # anything above ~1.41 will do
                for line in shape.lines:
                    dist = min(dist, line.distance(n_x, n_y))
                if sf:
                    stroke = sf.stroke(n_x, n_y)
                if dist < (stroke + edge_fade):
                    if dist < stroke:
                        image[y][x] = 1
                    else:
                        image[y][x] = (edge_fade - (dist - stroke)) / edge_fade
                else:
                    image[y][x] = 0
        return image


def lines_to_shape(lines, name=''):
    s = Shape(name)
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            x, y = lines[i][j]
            new_line = i > 0 and j == 0
            s.add_point(x, y, new_line=new_line)
    return s


if __name__ == '__main__':
    #  some helper functions to create data
    def create_line_interp(p0, p1, num_point):
        x0, y0 = p0
        x1, y1 = p1
        line = []
        for i in range(num_point):
            ratio = i / (num_point - 1)
            line.append((x0 * (1 - ratio) + x1 * ratio, y0 * (1 - ratio) + y1 * ratio))
        return line

    def idx_to_lines(lines, idx):
        return [lines[i] for i in idx]

    def idx_to_shape(lines, idx):
        l = [lines[i] for i in idx]
        return lines_to_shape(l)

    sm = ShapeParser(w=32, h=32)

    line_left = create_line_interp((0.2, 0.2), (0.2, 0.8), 20)
    line_right = create_line_interp((0.8, 0.2), (0.8, 0.8), 20)
    line_top = create_line_interp((0.2, 0.2), (0.8, 0.2), 20)
    line_bottom = create_line_interp((0.2, 0.8), (0.8, 0.8), 20)
    cross_a = create_line_interp((0.2, 0.2), (0.8, 0.8), 20)
    cross_b = create_line_interp((0.2, 0.8), (0.8, 0.2), 20)
    hor = create_line_interp((0.2, 0.5), (0.8, 0.5), 20)
    ver = create_line_interp((0.5, 0.2), (0.5, 0.8), 20)
    #            0          1           2         3            4        5        6    7
    all_lines = [line_left, line_right, line_top, line_bottom, cross_a, cross_b, hor, ver]
    classes = [[6, 7], [6], [0, 1, 2, 3], [2, 3, 4, 5], [0, 1, 4, 5], [2, 3, 6], [0, 1, 7], [0, 2, 5], [1, 3, 5]]
    class_9 = [[2, 3, 4], [2, 3, 5]]
    classes_lines = [idx_to_lines(all_lines, classes[i]) for i in range(len(classes))]
    class_9_lines = [idx_to_lines(all_lines, class_9[i]) for i in range(len(class_9))]

    classes_shape = [lines_to_shape(l) for l in classes_lines]
    class_9_shape = [lines_to_shape(l) for l in class_9_lines]

    images_clean = []
    for i in range(len(classes)):
        images_clean.append(sm.shape_to_image(classes_shape[i], edge_fade=0.04))
    for i in range(len(class_9)):
        images_clean.append(sm.shape_to_image(class_9_shape[i], edge_fade=0.04))

    images_distort = []
    for i in range(len(classes)):
        images_distort.append(sm.random_modification(classes_shape[i]))
    for i in range(len(class_9)):
        s = idx_to_shape(all_lines, class_9[i])
        images_distort.append(sm.random_modification(class_9_shape[i]))
    vis.plot_images([images_clean, images_distort])

