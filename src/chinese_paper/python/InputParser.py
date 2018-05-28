import cv2
import numpy as np
import math


def get_top_n_key_points(img, n):
    feature_detector = cv2.xfeatures2d.SURF_create()
    key_points = feature_detector.detect(img, None)
    key_points.sort(key=lambda kp: -kp.response)  # Sort in decreasing feature response order
    return key_points[0:n]  # Get the top N features


def get_1d_boundaries(img_shape, coordinates, patch_size, axis):
    n_points = len(coordinates)
    shift_array = np.full(n_points, math.floor(patch_size / 2))

    center_coords = coordinates.transpose()[axis]

    lo_coords = center_coords - shift_array
    hi_coords = center_coords + shift_array

    shift_forward = (lo_coords < 0) * lo_coords * -1
    aux = hi_coords - img_shape[axis] + 1
    shift_back = (aux > 0) * aux

    lo_coords = lo_coords - shift_back + shift_forward
    hi_coords = hi_coords - shift_back + shift_forward

    return lo_coords, hi_coords


def get_2d_boundaries(img_shape, coordinates, patch_size):
    x_lo, x_hi = get_1d_boundaries(img_shape, coordinates, patch_size, 0)
    y_lo, y_hi = get_1d_boundaries(img_shape, coordinates, patch_size, 1)
    return x_lo, x_hi, y_lo, y_hi


def get_vectorized_patches_from_key_points(img, key_points, patch_size):
    mapper = np.vectorize(lambda kp: (round(kp.pt[0]), round(kp.pt[1])))
    coordinates = np.array(mapper(key_points)).transpose()

    x_lo, x_hi, y_lo, y_hi = get_2d_boundaries(img.shape, coordinates, patch_size)

    n_cols = len(key_points)
    vector_patches = np.empty((n_cols, patch_size ** 2), dtype=int)
    for i in range(n_cols):
        patch = img[x_lo[i]:x_hi[i], y_lo[i]:y_hi[i]]
        vector_patch = patch.reshape((1, patch_size ** 2))
        vector_patches[i] = vector_patch

    return vector_patches


class InputParser:
    def __init__(self, n_patches, patch_size):
        self.n_patches = n_patches
        self.patch_size = patch_size

    def calculate(self, image):
        key_points = get_top_n_key_points(image, self.n_patches)
        vector_patches = get_vectorized_patches_from_key_points(image, key_points, self.patch_size)
        normalized_vector_patches = vector_patches / 255.0
        return normalized_vector_patches

    def calculate_all(self, images):
        batch = np.empty((self.n_patches, self.patch_size ** 2, len(images)))
        for i, image in enumerate(images):
            batch[0, 0, i] = self.calculate(image)
        return batch
