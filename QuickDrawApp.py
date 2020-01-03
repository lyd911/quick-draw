import cv2
from keras.models import load_model
import numpy as np

model = load_model('QuickDraw.h5')


def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    # print('pred_probab is: ', pred_probab)
    pred_class = list(pred_probab).index(max(pred_probab))
    # print('pred_class is: ', pred_class)
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


def main():
    categories = ['candle', 'door', 'lightning', 'moon', 'mountain', 'shoes', 'sword', 't-shirt', 'telephone', 'train']
    img_name = 'test_shoes_1.png'
    img = cv2.imread(img_name, 0)
    img = 255 - img
    pred_proba, pred_class = keras_predict(model, img)
    print('Prediction for', img_name, 'is:', categories[pred_class])

# keras_predict(model, np.zeros((50, 50, 1), dtype=np.uint8))
if __name__ == '__main__':
    main()
