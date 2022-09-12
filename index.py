from FaceExtractor import FaceExtractor
from BackgroundRemover import BackgroundRemover
from Cartoonifier import Cartoonifier

def process(img):
    face_extractor = FaceExtractor()
    bg_remover     = BackgroundRemover()
    cartoonifier   = Cartoonifier()

    result_img = img
    for processor in [face_extractor, cartoonifier, bg_remover]:
        result_img = processor.process(result_img)

    return result_img

if __name__ == "__main__":
    import cv2
    filename = "face1.jpeg"
    img = cv2.imread(filename)

    cv2.imwrite(f"Extracted_face_{filename}", process(img))