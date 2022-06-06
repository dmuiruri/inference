#! /usr/bin/env python

"""A module to carry out inference tests on object detection models

"""
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
from time import time
import requests

def load_image_into_tensor(path):
    """Load an image from file into a tensor

    An image can be opened as a file object which will result in
    string (byte string) or an image can be openeded as an array. In
    this case we open the image into a numpy array and convert it into
    a tensor.

    """
    image_np = np.array(Image.open(path))
    print('file opened')
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
    print('tensor created')
    return input_tensor.numpy().tolist()

def test_requests_arr():
    """
    Perform a single complete request to the server
    """
    img_inf = load_image_into_tensor('./image3.jpg')
    start = time()
    json_data = {
        "signature_name": 'serving_default',
        "instances": img_inf
    }
    with requests.Session() as sess:
        req = requests.Request("post", 'http://128.214.252.11:8501/v1/models/centernet_hg_1024:predict',
                               json=json_data)
        prepared_req = sess.prepare_request(req)
        response = sess.send(prepared_req)
    # print(f'prediction: {[np.argmax(i) for i in response.json()["predictions"]]}')
    # print(response.json())
    perf = float(time() - start)
    print(f'{perf}')
    sys.stdout.write('.')
    sys.stdout.flush()
    return perf

def perform_multiple_arr_requests(batch):
    """Send multiple requests.

    For statistical stability of results, we perform multiple requests
    to get a general distribution of the performance.

    """
    batch = create_arr_batch(batch)
    json_data = {
        "signature_name": 'predict_images',
        "instances": batch
    }
    arr_res = [test_requests_arr(json_data) for _ in range(number_of_tests)]
    return arr_res

if __name__ == '__main__':
    # print(load_image_into_tensor('./image1.jpg'))
    print(test_requests_arr())
