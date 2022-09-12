import cv2
from argparse import ArgumentParser
from cvzone.SelfiSegmentationModule import SelfiSegmentation

class BackgroundRemover:
    def __init__(self):
        self.segmentor = SelfiSegmentation()

    def process(self, img, threshold=0.50):
        # img = cv2.resize(img, (640, 480))
        color = (255, 255, 255)
        return self.segmentor.removeBG(img, color, threshold)



