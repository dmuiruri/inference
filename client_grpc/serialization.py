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

import mnist_input_data

######
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
import time
import base64
import requests
import pickle
import numpy as np
import random

model_name = 'mnist'
signature_name = 'predict_images'
input_name = 'images'
input_type = None
#batch_size = 1
work_dir = '/tmp'
#response_times = list()
test_data_set = mnist_input_data.read_data_sets(work_dir).test

def prepare_grpc_request(model_name, signature_name, data):
    start = time.time()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    request.inputs[input_name].CopyFrom(
        tf.make_tensor_proto(data, dtype=None))
    return float(time.time() - start)



def create_arr_batch(batch_size):
    img, label = test_data_set.next_batch(batch_size)
    batch = img.tolist()
    return batch

def serialize_str(batch_size):
    with open('0.png', 'rb') as payload:
        img_str = np.repeat(payload.read(), batch_size, axis=0).tolist()
    str_res = [prepare_grpc_request(model_name, signature_name, img_str) for _ in range(500)]
    return str_res

def serialize_arr(batch_size):
    batch = create_arr_batch(batch_size)
    arr_res = [prepare_grpc_request(model_name, signature_name, batch) for _ in range(500)]
    return arr_res

def run_str_serialization_tests():
    df = pd.DataFrame(index=range(500))
    for batchsize in [4, 16, 64, 256, 1024]:
        print(f'str: serializing batch size {batchsize}')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(serialize_str, b) for b in [batchsize]] # , 16, 64, 256, 1024
            res = [f.result() for f in concurrent.futures.as_completed(futures)]
        df[f'{batchsize}']=np.reshape(res,(500, 1)) #columns=[4, 16, 64, 256, 1024]
    print('saving results to csv file...')
    df.to_csv('str_batch_protobuf.csv')
    return

def run_arr_serialization_tests():
    df = pd.DataFrame(index=range(500))
    for batchsize in [4, 16, 64, 256, 1024]:
        print(f'arr: serializing batch size {batchsize}')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(serialize_arr, b) for b in [batchsize]] # , 16, 64, 256, 1024
            res = [f.result() for f in concurrent.futures.as_completed(futures)]
        df[f'{batchsize}']=np.reshape(res,(500, 1)) #columns=[4, 16, 64, 256, 1024]
    print('saving results to csv file...')
    df.to_csv('arr_batch_protobuf.csv')
    return

# def test_serialize_arr(json_data): 
#     start = time()
#     with requests.Session() as sess:
#         req = requests.Request("post", 'http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
#         prepared_req = sess.prepare_request(req)
#     return float(time() - start)

# def arr_serialization_tests(batch):
#     results = {}
#     print(f'arr batch size: {batch}') 
#     batch = create_arr_batch(batch)
#     json_data = {
#         "signature_name": 'predict_images',
#         "instances": batch
#     }
#     arr_res = [test_serialize_arr(json_data) for _ in range(500)]
#     return arr_res

# def run_arr_serialization_tests():
#     df = pd.DataFrame(index=range(500))
#     for batchsize in [4, 16, 64, 256, 1024]:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             futures = [executor.submit(arr_serialization_tests, b) for b in [batchsize]] # , 16, 64, 256, 1024
#             res = [f.result() for f in concurrent.futures.as_completed(futures)]
#         df[f'{batchsize}']=np.reshape(res,(500, 1)) #columns=[4, 16, 64, 256, 1024]
#     print('saving results to csv file...')
#     df.to_csv('arr_batch_protobuff.csv')
#     return

### String tests
# def test_serialize_string(json_data):
#     start = time()
#     with requests.Session() as sess:
#         req = requests.Request("post", 'http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
#         prepared_req = sess.prepare_request(req)
#     return float(time() - start)

# def str_serialization_tests(batch_size):
#     print(f'str batch size: {batch_size}')
#     batch = create_str_batch(batch_size)
#     json_data = {
#         "signature_name": 'predict_images',
#         "instances":[{'b64': base64.b64encode(s).decode('utf-8')} for s in batch]
#     }
#     reqs_time = [test_serialize_string(json_data) for _ in range(500)]
#     return reqs_time

# def run_str_serialization_tests():
#     df = pd.DataFrame(index=range(500))
#     for batchsize in [4, 16, 64, 256, 1024]:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             futures = [executor.submit(str_serialization_tests, b) for b in [batchsize]] # , 16, 64, 256, 1024
#             res = [f.result() for f in concurrent.futures.as_completed(futures)]
#         df[f'{batchsize}']=np.reshape(res,(500, 1)) #columns=[4, 16, 64, 256, 1024]
#     print('saving results to csv file...')
#     df.to_csv('str_batch.csv')
#     return

# def predict_string():
#     # payload = data.read()
#     # encoded_image = base64.b64encode(payload).decode('utf-8')
#     # instance = [{"b64": encoded_image}]
#     # return json.dumps({"instances": instance})

#     with open('0.png', 'rb') as payload:
#         img = payload.read()
#     img_encoded = base64.b64encode(img).decode('utf-8')
#     json_data = {
#         "signature_name": 'predict_images',
#         "instances": [{'b64': img_encoded}]
#     }
#     response = requests.post('http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
#     print(response.json())
#     return response.elapsed.total_seconds()

# def get_predictions():
#     """Inference querying
    
#     Predict returns the probabilities of the classes 0-9, so we need
#     to pick the highest probability

#     number = np.argmax(response_prediction.json()['predictions'][0])
#     """
#     json_data = prepare_data()
#     response = requests.post('http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
#     print(response.json())
#     return response.elapsed.total_seconds()

if __name__ == '__main__':
#    print(run_str_serialization_tests())
    run_str_serialization_tests()
    run_arr_serialization_tests()
