import cv2
import Config
import matplotlib.pyplot as plt
import numpy as np

class RoadMask:
     def __init__(self):
         pass

if __name__ == '__main__':

    video_id = 1
    base_image = cv2.imread(Config.data_path + '/average_image/' + str(video_id) + '/average10.jpg')
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(base_image.shape, np.float)
    cap = cv2.VideoCapture(Config.dataset_path + '/' + str(video_id) + '.mp4')
    while (True):
        ret, frame = cap.read()
        if (ret):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask += abs(frame - base_image)
        else:
            break
    cap.release()
    plt.imshow(((mask / np.max(mask)) * 255).astype(int))
    plt.show()