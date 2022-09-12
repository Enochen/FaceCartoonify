import requests
import cv2
import numpy as np

class Cartoonifier:
    def __init__(self):
        self.headers = {
            'accept': 'image/png'
        }
        self.url = 'https://master-white-box-cartoonization-psi1104.endpoint.ainize.ai/predict'

    def process(self, img):
        encoded_img = cv2.imencode('.png', img)[1]
        image_bytes = encoded_img.tobytes()

        files = {
            'file_type': (None, 'image'),
            'source': ('', image_bytes, 'image/png')
        }

        response = requests.post(self.url, headers=self.headers, files=files)

        buff = np.fromstring(response.content, np.uint8)

        return cv2.imdecode(buff, cv2.IMREAD_COLOR)

if __name__ == "__main__":
    filename = "face1.jpeg"
    img = cv2.imread(filename)
    cartoonify = Cartoonify()
    print(cartoonify.process(img))
# return cv2.imdecode(response.content, cv2.1)
# with open('img.png', 'wb') as f:
#     f.write(response.content)
# files = { 
#     'file_type': (None, 'image'),
#     'source': (filename, open(filename, 'rb'), 'image/png')
# }

# response = requests.post('https://master-white-box-cartoonization-psi1104.endpoint.ainize.ai/predict', headers=headers, files=files)

# with open('img.png', 'wb') as f:
#     f.write(response.content)

# print(response.text)