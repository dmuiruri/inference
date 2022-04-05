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
from threading import Thread
import concurrent.futures
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
batch_size = 1

test_data_set = mnist_input_data.read_data_sets(work_dir).test
def create_arr_batch(batch_size):
    img, label = test_data_set.next_batch(batch_size)
    batch = img.tolist()
    return batch

def create_str_batch(batch_size):
    with open('0.png', 'rb') as payload:
        img_str = np.repeat(payload.read(), batch_size, axis=0).tolist()
    return img_str

### Array tests
def test_serialize_arr(json_data):
    start = time()
    with requests.Session() as sess:
        req = requests.Request("post", 'http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
        prepared_req = sess.prepare_request(req)
    return float(time() - start)

def arr_serialization_tests(batch):
    print(f'arr batch size: {batch}') 
    batch = create_arr_batch(batch)
    json_data = {
        "signature_name": 'predict_images',
        "instances": batch
    }
    arr_res = [test_serialize_arr(json_data) for _ in range(500)]
    return arr_res

def run_arr_serialization_tests():
    df = pd.DataFrame(index=range(500))
    for batchsize in [4, 16, 64, 256, 1024]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(arr_serialization_tests, b) for b in [batchsize]] # , 16, 64, 256, 1024
            res = [f.result() for f in concurrent.futures.as_completed(futures)]
        df[f'{batchsize}']=np.reshape(res,(500, 1)) #columns=[4, 16, 64, 256, 1024]
    print('saving results to csv file...')
    df.to_csv('arr_batch.csv')
    return

### String tests
def test_serialize_string(json_data):
    start = time()
    with requests.Session() as sess:
        req = requests.Request("post", 'http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
        prepared_req = sess.prepare_request(req)
    return float(time() - start)

def str_serialization_tests(batch_size):
    print(f'str batch size: {batch_size}')
    batch = create_str_batch(batch_size)
    json_data = {
        "signature_name": 'predict_images',
        "instances":[{'b64': base64.b64encode(s).decode('utf-8')} for s in batch]
    }
    reqs_time = [test_serialize_string(json_data) for _ in range(500)]
    return reqs_time

def run_str_serialization_tests():
    df = pd.DataFrame(index=range(500))
    for batchsize in [4, 16, 64, 256, 1024]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(str_serialization_tests, b) for b in [batchsize]] # , 16, 64, 256, 1024
            res = [f.result() for f in concurrent.futures.as_completed(futures)]
        df[f'{batchsize}']=np.reshape(res,(500, 1)) #columns=[4, 16, 64, 256, 1024]
    print('saving results to csv file...')
    df.to_csv('str_batch.csv')
    return

if __name__ == '__main__':
    print(run_str_serialization_tests())
    print(run_arr_serialization_tests())
