import numpy as np
import torch

from DeepGaze.deepgaze_pytorch import DeepGazeIIE
from scipy.ndimage import zoom
from scipy.special import logsumexp

from common.viridis_cm import apply_viridis_cm


class DeepGazeWrapper:
    CENTERBIAS_PATH = "./DeepGaze/centerbias_mit1003.npy"

    def __init__(self, device):
        self.device = device
        self.deepgaze = DeepGazeIIE().to(device=device, dtype=torch.float32)
        self.centerbias_template = np.load(DeepGazeWrapper.CENTERBIAS_PATH)

    def get_centerbias_for_image(self, h, w):
        # rescale centerbias to match image size
        centerbias = self.centerbias_template
        centerbias = zoom(centerbias, (h / centerbias.shape[0], w / centerbias.shape[1]), order=0, mode='nearest')
        centerbias -= logsumexp(centerbias)  # renormalize log density
        return centerbias

    def compute_gaze(self, image):
        image = torch.permute(image, (0, 3, 1, 2))  # B, C, H, W

        centerbias = self.get_centerbias_for_image(image.shape[2], image.shape[3])
        centerbias_tensor = torch.from_numpy(centerbias).unsqueeze(0).to(self.device)

        log_gaze_prob = self.deepgaze(image, centerbias_tensor)

        return log_gaze_prob


if __name__ == "__main__":
    import cv2
    with torch.no_grad():
        device = torch.device('cpu')
        d = DeepGazeWrapper(device)
        image = cv2.imread("image.jpg")
        w = 540
        s = (1280 - w) // 2
        image = image[s:s+w]  # resize from 720x1280 to 540x720
        img = torch.from_numpy(image).unsqueeze(0).to(device=device)  # 1, H, W, C
        log_gaze = d.compute_gaze(img).squeeze().numpy()
        gaze = np.exp(log_gaze)

    # gaze -= gaze.min()
    gaze /= gaze.max()

    heatmap = gaze - gaze.min()
    heatmap /= heatmap.max()
    heatmap = apply_viridis_cm(heatmap)

    gaze = gaze.reshape(*gaze.shape, 1)

    print(gaze.shape)
    print('gaze', gaze.min(), gaze.mean(), gaze.max())

    cv2.imshow('image', image)
    cv2.imshow('gaze', gaze)
    cv2.imshow('gaze image', image / 255 * gaze)
    cv2.imshow('gaze heatmap', heatmap)
    cv2.waitKey()
