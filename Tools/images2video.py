import numpy as np
import Config
import cv2
import os
import glob
import matplotlib.pyplot as plt

def mkdir(path):
    if not os.path.exists(output_path):
        os.mkdir(path)

def sort(images):
    for i in range(0, len(images)):
        for j in range(i + 1, len(images)):
            if (images[i] > images[j]):
                tmp = images[i]
                images[i] = images[j]
                images[j] = tmp

input_path = Config.output_path
videos = [20, 28]
video_size = (800, 410)

for video in videos:
    output_path = input_path + '/video_%03d' % (video)
    mkdir(output_path)

    writer = cv2.VideoWriter(output_path + '/anomaly.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)
    images = glob.glob(input_path + '/' + str(video) + '/1/anomaly*')
    sort(images)
    print(images)
    for image in images:
        im = cv2.imread(image)
        writer.write(im)
    writer.release()

    writer = cv2.VideoWriter(output_path + '/origin.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)
    images = glob.glob(input_path + '/' + str(video) + '/1/origin_*')
    sort(images)
    print(images)
    for image in images:
        im = cv2.imread(image)
        writer.write(im)
    writer.release()

    writer = cv2.VideoWriter(output_path + '/events.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)
    images = glob.glob(input_path + '/' + str(video) + '/1/events_*')
    sort(images)
    for image in images:
        im = cv2.imread(image)
        writer.write(im)
    writer.release()

    writer = cv2.VideoWriter(output_path + '/events2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, video_size)
    images = glob.glob(input_path + '/' + str(video) + '/1/events2_*')
    sort(images)
    for image in images:
        im = cv2.imread(image)
        writer.write(im)
    writer.release()

    # writer = cv2.VideoWriter(output_path + '/graph.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (800, 410))
    # txtfile = input_path + '/' + str(video) + '/' + str(video) + '_anomaly.txt'
    # f = open(txtfile, 'r')
    # lines = f.readlines()
    # x = []
    # y = []
    # for i in range(0, len(lines) - 1):
    #     line = lines[i]
    #     split = line.split(' ')
    #     x.append(int(split[0]))
    #     y.append(float(split[1]))
    #     fig = plt.figure(figsize=(8, 4.101))
    #     ax = fig.add_subplot(1, 1, 1)
    #     ax.plot(x, y, linewidth = 8)
    #     ax.yaxis.grid(linewidth = 4)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     ax.spines["left"].set_visible(False)
    #     ax.axis([0, 900, 0, 1.1])
    #     ax.set_yticks(np.arange(0, 1.1, 0.2))
    #     ax.tick_params(axis='both', which='major', labelsize=14)
    #     ax.set_title('Anomaly Detection', fontdict={'fontsize': 20})
    #     fig.savefig('temp.png')
    #     im = cv2.imread('temp.png')
    #     writer.write(im)
    #     # plt.show()
    #     plt.close()
    #
    # writer.release()