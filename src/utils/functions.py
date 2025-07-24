import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage.metrics import structural_similarity as ssim


def disp2coords(
    displacements: np.ndarray, img_shape: tuple[int], xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray]:
    spline_dx = RectBivariateSpline(ys, xs, displacements[0])
    spline_dy = RectBivariateSpline(ys, xs, displacements[1])
    XX, YY = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]))
    disp_x = spline_dx(YY[:, 0], XX[0, :])
    disp_y = spline_dy(YY[:, 0], XX[0, :])
    return (XX + disp_x).ravel(), (YY + disp_y).ravel()

def compute_ssim(pred_tensor, target_tensor):
    pred = pred_tensor.squeeze(1).detach().cpu().numpy()
    target = target_tensor.squeeze(1).detach().cpu().numpy()

    ssim_scores = []
    for p, t in zip(pred, target):
        ssim_scores.append(ssim(p, t, data_range=1.0))
    return np.mean(ssim_scores)