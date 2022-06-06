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
import sys

def load_image_into_tensor(path):
    """Load an image from file into a tensor

    An image can be opened as a file object which will result in
    string (byte string) or an image can be openeded as an array. In
    this case we open the image into a numpy array and convert it into
    a tensor.

    """
    image_np = np.array(Image.open(path))
    # img_batch = np.repeat(image_np, batch_size, axis=0)#.tolist() How to perform batch inference in these models.
    print(f'image file opened')
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
    print('tensor created')
    return input_tensor.numpy().tolist()

def test_requests_arr():
    """
    Perform a single complete request to the server

    Different model URLs
    http://128.214.252.11:8501/v1/models/centernet_hg_1024:predict
    http://128.214.252.11:8501/v1/models/centernet_hg_512:predict
    http://128.214.252.11:8501/v1/models/centernet_resnet50_512:predict
    """
    img_inf = load_image_into_tensor('./image3.jpg')
    json_data = {
        "signature_name": 'serving_default',
        "instances": img_inf
    }
    start = time()
    with requests.Session() as sess:
        req = requests.Request("post", 'http://128.214.252.11:8501/v1/models/centernet_resnet50_512:predict',
                               json=json_data)
        prepared_req = sess.prepare_request(req)
        response = sess.send(prepared_req)
    # print(f'prediction: {[np.argmax(i) for i in response.json()["predictions"]]}')
    # print(response.json())
    perf = float(time() - start)
    print(f'{perf}')
    sys.stdout.write('.')
    sys.stdout.flush()
    return response

def perform_multiple_arr_requests(number_of_reqs):
    """Send multiple requests.

    For statistical stability of results, we perform multiple requests
    to get a general distribution of the performance.

    """
    arr_res = [test_requests_arr() for _ in range(number_of_reqs)]
    return arr_res

if __name__ == '__main__':
    # image = load_image_into_tensor('./image1.jpg')
    # print(test_requests_arr())
    perform_multiple_arr_requests(2) # warmup and test test request
