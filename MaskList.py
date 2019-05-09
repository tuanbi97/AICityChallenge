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
    list = MaskList(Config.data_path + '/masksv3')
    video_id = 2
    scene_id = 1
    frame_id = 192
    mask = list[(video_id, 1)]
    mask = (mask * 255).astype(int)
    mask = np.dstack((mask, mask, mask))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(mask)
    im = cv2.cvtColor(Image.load(Config.data_path + '/output5/' + str(video_id) + '/' + str(scene_id) + '/' + 'events' + format(frame_id, '03d') + '.jpg'), cv2.COLOR_BGR2RGB)
    im = cv2.addWeighted(im.astype(int), 1, mask, 0.4, 0.0)
    ax2.imshow(im)
    plt.show()