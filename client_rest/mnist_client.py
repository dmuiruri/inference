# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python

import sys
import threading
import requests
import numpy as np
import pandas as pd
import asyncio
import aiohttp
import timeit
import base64
import json

from time import time

import mnist_input_data

work_dir = '/tmp'
#response_times = list()
batch_size = 16

test_data_set = mnist_input_data.read_data_sets(work_dir).test
img, label = test_data_set.next_batch(batch_size)
batch = img.tolist()

with open('0.png', 'rb') as payload:
    img_str = np.repeat(payload.read(), batch_size, axis=0).tolist()

def test_serialize_arr():
    start = time()
    json_data = {
        "signature_name": 'predict_images',
        "instances": batch
    }
    with requests.Session() as sess:
        req = requests.Request("post", 'http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
        prepared_req = sess.prepare_request(req)
    return float(time() - start)

def test_serialize_string():
    start = time()
    json_data = {
        "signature_name": 'predict_images',
        "instances":[{'b64': base64.b64encode(s).decode('utf-8')} for s in img_str]
    }
    with requests.Session() as sess:
        req = requests.Request("post", 'http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
        prepared_req = sess.prepare_request(req)
    return float(time() - start)

def predict_string():
    # payload = data.read()
    # encoded_image = base64.b64encode(payload).decode('utf-8')
    # instance = [{"b64": encoded_image}]
    # return json.dumps({"instances": instance})

    with open('0.png', 'rb') as payload:
        img = payload.read()
    img_encoded = base64.b64encode(img).decode('utf-8')
    json_data = {
        "signature_name": 'predict_images',
        "instances": [{'b64': img_encoded}]
    }
    response = requests.post('http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
    print(response.json())
    return response.elapsed.total_seconds()
    
    
def prepare_data():
    test_data = {
        "signature_name": 'predict_images',
        "instances": batch
    }
    return test_data

def get_predictions():
    """Inference querying
    
    Predict returns the probabilities of the classes 0-9, so we need
    to pick the highest probability

    number = np.argmax(response_prediction.json()['predictions'][0])
    """
    json_data = prepare_data()
    response = requests.post('http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
    print(response.json())
    return response.elapsed.total_seconds()

def run_performance_tests_no_serialization(iterations=10):
    return [get_predictions() for _ in range(iterations)]

if __name__ == '__main__':
    # print(predict_string())
    # print(run_performance_tests_no_serialization(iterations=1))
    print(img.shape)
    print([test_serialize_arr() for _ in range(100)])
    print('>>>')
    print([test_serialize_string() for _ in range(100)])
