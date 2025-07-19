import numpy as np
from scipy.interpolate import RectBivariateSpline


def disp2coords(
    displacements: np.ndarray, img_shape: tuple[int], xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray]:
    spline_dx = RectBivariateSpline(ys, xs, displacements[0])
    spline_dy = RectBivariateSpline(ys, xs, displacements[1])
    XX, YY = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]))
    disp_x = spline_dx(YY[:, 0], XX[0, :])
    disp_y = spline_dy(YY[:, 0], XX[0, :])
    return (XX + disp_x).ravel(), (YY + disp_y).ravel()
