
import cv2
import numpy as np
import json
import os

class FaceExtractor:
    def __init__(self):
        self.prototype = {
            "PATH"    : "face_detector_prototype",
            "PROTO"   : "deploy.prototxt",
            "WEIGHTS" : "res10_300x300_ssd_iter_140000.caffemodel"
        }
        self.detector = self.__getDetectorFromPath()

    def process(self, img):
        detect_result_box = self.detectFaceRegion(img)
        face_img = self.__cropImageFromDetection(img, detect_result_box)
        return face_img

    def __cropImageFromDetection(self, img, rect):
        """
        This methods crop out the image using the rect data to extract the
        face region from the given image. After that, it will resize the image
        to size x size. The default is 128 x 128

        Params:
        -------
        img : numpy.ndarray
            The image that has face
        rect : array
            an array that stores x1, y1, x2, y2 position to crop the image.
            It's the position from the top left corner of the image to the
            lower right corner.

        Return:
        -------
            numpy.ndarray
                The cropped face with the dimension size x size
        """
        #Extract features inside the detected area
        [x1, y1, x2, y2] = rect
        #Extract, resize and convert to gray scale for this face region
        face_img  = img[y1 : y2, x1 : x2]

        # try:
        #     face_img  = cv2.resize(face_img, (size, size))
        # except Exception:
        #     print("Cannot extract face")

        return face_img


    def detectFaceRegion(self, img):
        """
        This relies on the open cv dnn deep learning model to detect the face
        location from the image. The detector require a blob, so that, the
        method cv2.dnn.blobFromImage is used.

        Params:
        -------
        detector: deep learning model
            The open cv face detection model
        img : numpy.ndarray
            The image that has face
        Return:
        -------
            numpy.ndarray
                The cropped face with the dimension size x size
        """
        detector           = self.detector
        scale              = 1.0
        size               = (300, 300) #Input size Size taken from the deploy.prototxt
        mean_substract_val = (104, 177, 123)
        #mean_subtract_val for human face docs
        #https://github.com/opencv/opencv/tree/master/samples/dnn
        try:
            ### To use the detector above, we need to pass in an image as a blob, which can
            ### be created using cv2.dnn.blobFromImage
            blob  = cv2.dnn.blobFromImage(img, scale, size, mean_substract_val)
            #input the image and get face detection result
            detector.setInput(blob)
            detections = detector.forward()
            #This return an 4D array. Read more:
            #https://stackoverflow.com/questions/67355960/what-does-the-4d-array-returned-by-net-forward-in-opencv-dnn-means-i-have-lit
        except Exception:
            return None


        #Read data from this neural network
        #https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
        h, w = img.shape[:2] # height x width because of num rows x num cols
        list_boxes = []
        scaled_box = None
        for i in range(detections.shape[2]):
            #Confidence probability of this detection
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                #this box is normalized
                #box return startX, startY, endX, endY
                box         = detections[0, 0, i, 3:7]
                #Return here

                scaled_box  = box * [w, h, w, h]
                list_boxes.append(scaled_box)

                #Only take take the image that has 1 face only
                if len(list_boxes) > 1:
                    print("More than 1 face was detected")
                    return None
                if len(list_boxes) == 0:
                    print("No face detected")
                    return None

            if type(scaled_box) == type(None):
                return None

        return scaled_box.astype("int") #Convert to integer array for slicing

        
    def __getDetectorFromPath(self):
        """
        Load the pre-trained model from open cv dnn face detection module. It
        will require a path that has a proto file "deploy.prototxt" and the net
        "res10_300x300_ssd_iter_140000.caffemodel"

        Params:
        -------
            filepath: str
                The path to a folder that has the 2 file for this detection
                module
        Return:
        -------
            open cv dnn Net:
                The deep learning model for face detection
        """
        prototype = self.prototype
        dnn_model_dir = os.path.join(os.getcwd(), prototype["PATH"])
        prototxt_path = os.path.join(dnn_model_dir, prototype["PROTO"])
        weights_path  = os.path.join(dnn_model_dir, prototype["WEIGHTS"])
        detector      = cv2.dnn.readNet(prototxt_path, weights_path)

        return detector

