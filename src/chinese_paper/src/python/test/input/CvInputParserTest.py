from pathlib import Path

import cv2
import numpy as np
import pytest
from mockito import mock

from main.input.CvInputParser import get_1d_boundaries, get_2d_boundaries, get_vectorized_patches_from_key_points, \
    Parser

root = Path('../../')


def test_get_1d_boundaries_all_conditions():
    rect_shape = [10, 10]
    patch_size = 3
    center_points = np.array([[0, 0], [4, 0], [9, 0]])
    # noinspection PyTupleAssignmentBalance
    lo, hi = get_1d_boundaries(rect_shape, center_points, patch_size, axis=0)
    expected_lo = np.array([0, 3, 7])
    assert np.array_equal(lo, expected_lo)
    expected_hi = np.array([2, 5, 9])
    assert np.array_equal(hi, expected_hi)


def test_get_1d_boundaries_all_conditions_2():
    rect_shape = [20, 20]
    patch_size = 5
    center_points = np.array([[0, 0], [9, 0], [19, 0]])
    # noinspection PyTupleAssignmentBalance
    lo, hi = get_1d_boundaries(rect_shape, center_points, patch_size, axis=0)
    expected_lo = np.array([0, 7, 15])
    assert np.array_equal(lo, expected_lo)
    expected_hi = np.array([4, 11, 19])
    assert np.array_equal(hi, expected_hi)


def test_get_1d_boundaries_invalid_patch_size():
    with pytest.raises(ValueError) as exception:
        rect_shape = [20, 20]
        patch_size = 4
        center_points = np.array([[0, 0], [9, 0], [19, 0]])
        get_1d_boundaries(rect_shape, center_points, patch_size, axis=0)
        assert 'Invalid patch size' in str(exception.value)


def test_get_1d_boundaries_invalid_rect_shape():
    with pytest.raises(ValueError) as exception:
        rect_shape = [20]
        patch_size = 5
        center_points = np.array([[0, 0], [9, 0], [19, 0]])
        get_1d_boundaries(rect_shape, center_points, patch_size, axis=0)
        assert 'Invalid rect shape' in str(exception.value)


def test_get_1d_boundaries_invalid_center_points():
    with pytest.raises(ValueError) as exception:
        rect_shape = [20, 20]
        patch_size = 5
        center_points = np.array([[[0, 0], [9, 0], [19, 0]]])
        get_1d_boundaries(rect_shape, center_points, patch_size, axis=0)
        assert 'Invalid center points' in str(exception.value)


def test_get_1d_boundaries_invalid_center_points_2():
    with pytest.raises(ValueError) as exception:
        rect_shape = [20, 20]
        patch_size = 5
        center_points = np.array([[0, 0, 0], [9, 0, 0], [19, 0, 0]])
        get_1d_boundaries(rect_shape, center_points, patch_size, axis=0)
        assert 'Invalid center points' in str(exception.value)


def test_get_2d_boundaries_all_conditions():
    rect_shape = [10, 10]
    patch_size = 3
    center_points = np.array([[0, 0], [0, 4], [0, 9], [4, 0], [4, 4], [4, 9], [9, 0], [9, 4], [9, 9]])
    x_lo, x_hi, y_lo, y_hi = get_2d_boundaries(rect_shape, center_points, patch_size)
    expected_x_lo = np.array([0, 0, 0, 3, 3, 3, 7, 7, 7])
    assert np.array_equal(x_lo, expected_x_lo)
    expected_x_hi = np.array([2, 2, 2, 5, 5, 5, 9, 9, 9])
    assert np.array_equal(x_hi, expected_x_hi)
    expected_y_lo = np.array([0, 3, 7, 0, 3, 7, 0, 3, 7])
    assert np.array_equal(y_lo, expected_y_lo)
    expected_y_hi = np.array([2, 5, 9, 2, 5, 9, 2, 5, 9])
    assert np.array_equal(y_hi, expected_y_hi)


def test_get_vectorized_patches_from_keypoints():
    mock_points = []
    n_keypoints = 3
    for i in range(n_keypoints):
        pt = mock({'pt': [float(i), float(i)]})
        mock_points.append(pt)
    img = np.arange(0, 100).reshape(10, 10)
    patch_size = 3
    vectorized_patches = get_vectorized_patches_from_key_points(img, mock_points, patch_size)
    assert vectorized_patches.shape[0] == n_keypoints
    assert vectorized_patches.shape[1] == patch_size ** 2
    expected = np.array([[0, 1, 2, 10, 11, 12, 20, 21, 22], [0, 1, 2, 10, 11, 12, 20, 21, 22],
                         [11, 12, 13, 21, 22, 23, 31, 32, 33]])
    assert np.array_equal(vectorized_patches, expected)


def test_parser_calculate():
    n_patches = 5
    patch_size = 3
    parser = Parser(n_patches, patch_size)
    img_path = root.joinpath('resources/sample.jpg')
    img = cv2.imread(str(img_path.resolve()), cv2.IMREAD_GRAYSCALE)
    parsed_1 = parser.calculate(img)
    expected_shape = [n_patches, patch_size ** 2]
    assert np.array_equal(parsed_1.shape, expected_shape)
    parsed_2 = parser.calculate_from_path(img_path)
    assert np.array_equal(parsed_2.shape, expected_shape)
    assert np.array_equal(parsed_1, parsed_2)


def test_parser_calculate_all():
    n_patches = 5
    patch_size = 3
    n_images = 4
    parser = Parser(n_patches, patch_size)

    img_path = root.joinpath('resources/sample.jpg')
    path_array = []
    img_array = []
    for i in range(n_images):
        path = str(img_path.resolve())
        path_array.append(path)
        img_array.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    parsed_1 = parser.calculate_all(img_array)
    parsed_2 = parser.calculate_all_from_path(path_array)
    expected_shape = [n_patches * n_images, patch_size ** 2]
    assert np.array_equal(parsed_1.shape, expected_shape)
    assert np.array_equal(parsed_2.shape, expected_shape)
    assert np.array_equal(parsed_1, parsed_2)
