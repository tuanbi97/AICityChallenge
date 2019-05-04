import numpy as np
import cv2
from Misc import BoundingBox
import Config

class AnomalyEvent:
    def __init__(self, box, time):
        self.region = box
        self.start_time = time
        self.latest_update = time
        self.boxes = [box] #list of bounding box in anomaly event
        self.count = 1 #appear times
        self.status = 0 #0 / 1 = suspect / anomaly

    def getConf(self):
        return self.region.score

    def boxIntersect(self, box1, box2):
        xmax = max(box1.x1, box2.x1)
        xmin = min(box1.x2, box2.x2)
        ymax = max(box1.y1, box2.y1)
        ymin = min(box1.y2, box2.y2)
        if (xmax > xmin) or (ymax > ymin):
            return 0
        else:
            return (xmin - xmax) * (ymin - ymax)

    def overlapRatio(self, box):
        return self.boxIntersect(self.region, box) / box.area()

    def radiusRestrict(self, box):
        x1 = (self.region.x1 + self.region.x2) / 2.0
        y1 = (self.region.y1 + self.region.y2) / 2.0
        x2 = (box.x1 + box.x2) / 2.0
        y2 = (box.y1 + box.y2) / 2.0
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        radius = (((self.region.x2 - self.region.x1) ** 2 + (self.region.y2 - self.region.y1) ** 2) ** 0.5) / 2 \
                + (((box.x2 - box.x1) ** 2 + (box.y2 - box.y1) ** 2) ** 0.5) / 2
        if dist <= radius: return True
        return False

    def checkContains(self, box):
        if (self.overlapRatio(box) > Config.aevent_overlap_ratio) or (self.radiusRestrict(box)):
            return True
        else:
            return False

    def expandRegion(self, box):
        self.boxes.append(box)
        self.region.x1 = min(self.region.x1, box.x1)
        self.region.y1 = min(self.region.y1, box.y1)
        self.region.x2 = max(self.region.x2, box.x2)
        self.region.y2 = max(self.region.y2, box.y2)
        self.region.score = 1.0 - (1.0 - self.region.score) * (1.0 - box.score)

    def addBox(self, box, time):
        self.expandRegion(box)
        self.latest_update = time
        self.count += 1

class AnomalyDetector:
    def __init__(self):
        self.nextId = 0
        self.events = {}  # list of anomaly event

    def addBoxes(self, boxes, time):
        for box in boxes:
            lc = 0
            max_overlap = -1.0
            max_event = None
            for key in self.events.keys():
                event = self.events[key]
                if event.checkContains(box):
                    lc = 1
                    if event.overlapRatio(box) > max_overlap:
                        max_overlap = event.overlapRatio(box)
                        max_event = event

            if lc == 1:
                max_event.addBox(box, time)
            if lc == 0:
                self.events[self.nextId] = AnomalyEvent(box, time)
                self.nextId += 1

    def examineEvents(self, video_id, scene_id, time, isEnd, file):
        ret = []
        keys = [key for key in self.events.keys()]
        for key in keys:
            event = self.events[key]
            if time - event.latest_update > Config.threshold_anomaly_finish \
                    or (time > event.start_time and event.count / (time - event.start_time) < Config.threshold_anomaly_freq) \
                    or isEnd:
                if event.status == 1: #anomaly event
                    #format: video_id scene_id start_time end_time confident
                    file.write(str(video_id) + ' ' + str(scene_id) + ' ' + str(event.start_time) + ' ' + str(time) + ' ' + str(event.getConf()) + '\n')
                self.events.pop(key)
            else:
                if time - event.start_time > Config.threshold_anomaly_least_time:
                    event.status = 1
                    ret.append(event)
        return ret

    def drawEvents(self, im):
        for key in self.events.keys():
            event = self.events[key]
            if event.status == 0:
                #draw proposal
                im = cv2.rectangle(im, (event.region.x1, event.region.y1), (event.region.x2, event.region.y2),
                                   (255, 153, 51), 3)
                im = cv2.putText(im, "%d proposal" % (key), (event.region.x2 + 10, event.region.y1),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 153, 51), 2, cv2.LINE_AA)
            else:
                if event.status == 1:
                    im = cv2.rectangle(im, (event.region.x1, event.region.y1), (event.region.x2, event.region.y2), (255, 0, 0), 3)
                    im = cv2.putText(im, "%d %.2f" % (key, event.region.score), (event.region.x2 + 10, event.region.y1),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                    im = cv2.putText(im, "start: %.2f" % (event.start_time), (event.region.x1, event.region.y2 + 20),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        return im