import numpy as np
import Config
import matplotlib.pyplot as plt
from Misc import Image
import cv2

class MaskList:
    def __init__(self, file_path):
        self.file_path = file_path

    def __getitem__(self, key):
        video_id, scene_id = key
        mask = np.load(self.file_path + '/mask_' + str(video_id) + '_' + str(scene_id) + '.npy')
        return mask

if __name__ == '__main__':
    list = MaskList(Config.data_path + '/masks')
    mask = list[(1, 1)]
    mask = (mask * 255).astype(int)
    mask = np.dstack((mask, mask, mask))
    im = cv2.cvtColor(Image.load(Config.data_path + '/average_image/1/average145.jpg'), cv2.COLOR_BGR2RGB)
    im = cv2.addWeighted(im.astype(int), 1, mask, 0.3, 0.0)
    plt.imshow(im)
    plt.show()