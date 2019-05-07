import cv2
import Config
import matplotlib.pyplot as plt
import numpy as np
import json

class RoadMask:
    def __init__(self, mask_path, scene_path, im_path):
        self.mask_path = mask_path
        self.scene_path = scene_path
        self.im_path = im_path
        with open(scene_path, 'r') as f:
            self.stableList = json.load(f)
        self.refineMasks()

    def getMask(self, video_id, scene_id):
        mask = np.load(self.file_path + '/mask_' + str(video_id) + '_' + str(scene_id) + '.npy')
        return mask

    def refineMask(self, im, mask):
        mask = (mask * 255).astype(int)
        mask = np.dstack((mask, mask, mask))


    def refineMasks(self):
        for video_id in range(1, 2):
            stableIntervals = self.stableList[str(video_id)]
            for scene_id in range(1, len(stableIntervals) + 1):
                l, r = stableIntervals[scene_id - 1]
                mask = self.getMask(video_id, scene_id)
                im = cv2.cvtColor(cv2.imread(self.im_path + '/' + str(video_id) + '/average' + str(l) + '.jpg'), cv2.COLOR_BGR2RGB)
                self.refineMask(im, mask)

        print(self.stableList)

if __name__ == '__main__':
    list = RoadMask(Config.data_path + '/masks', Config.data_path + '/unchanged_scene_periods.json', Config.data_path + '/average_image')
    # video_id = 1
    # mask = list[(video_id, 1)]
    # mask = (mask * 255).astype(int)
    # mask = np.dstack((mask, mask, mask))
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(mask)
    # im = cv2.cvtColor(cv2.imread(Config.data_path + '/average_image/' + str(video_id) + '/average145.jpg'),
    #                   cv2.COLOR_BGR2RGB)
    # im = cv2.addWeighted(im.astype(int), 1, mask, 0.5, 0.0)
    # ax2.imshow(im)
    # plt.show()