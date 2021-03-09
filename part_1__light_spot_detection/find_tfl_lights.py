import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter


def find_tfl_lights(c_image: np.ndarray):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    kernel = np.array([[-3 - 3j,     0 - 10j,    3 - 3j],
                       [-10 + 0j,    0 + 0j,     10 + 0j],
                       [-3 + 3j,     0 + 10j,    3 + 3j]])

    ######## convolve
    # red
    black_white_image = c_image[:, :, 0]
    red_grad = convolve2d(black_white_image.T, kernel, boundary='symm', mode='same')

    # green
    black_white_image = c_image[:, :, 1]
    green_grad = convolve2d(black_white_image.T, kernel, boundary='symm', mode='same')

    ######## maximum filter
    max_filter_red = maximum_filter(np.real(red_grad), size=10)
    max_filter_red = np.argwhere(max_filter_red == red_grad)
    max_filter_green = maximum_filter(np.real(green_grad), size=10)
    max_filter_green = np.argwhere(max_filter_green == red_grad)

    ######## filter
    # red
    red = np.argwhere(red_grad > 1650)
    red = np.array([i for i in red if i in max_filter_red])
    red_x, red_y = np.array(red[:, :1]).ravel(), np.array(red[:, 1:]).ravel()

    # green
    green = np.argwhere(green_grad > 1650)
    green = np.array([i for i in green if i in max_filter_green])
    green_x, green_y = np.array(green[:, :1]).ravel(), np.array(green[:, 1:]).ravel()

    return red_x, red_y, green_x, green_y