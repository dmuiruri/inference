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
import grpc
import time
import base64
import requests
import pickle
import numpy as np
import random
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from grpc.beta import implementations

model_name = 'mnist'
signature_name = 'predict_images'
hostport = '8500'
input_name = 'images'
input_type = None
batch_size = 16
number_of_tests = 20 #500
work_dir = './tmp'
#response_times = list()
test_data_set = mnist_input_data.read_data_sets(work_dir).test
img, label = test_data_set.next_batch(batch_size)
batch = img #np.repeat(img[0], batch_size, axis=0).tolist()
print(f'{img.shape}')

channel = grpc.insecure_channel('128.214.252.11:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
# channel = implementations.insecure_channel('localhost')
# stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

def prepare_grpc_full_request(model_name, signature_name, batchsize):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    img, label = test_data_set.next_batch(batchsize)
    request.inputs[input_name].CopyFrom(
        tf.make_tensor_proto(img[0], shape=[batchsize, img[0].size], dtype=None))
    return request

def prepare_grpc_request(model_name, signature_name, data):
    start = time.time()
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    request.inputs[input_name].CopyFrom(
        tf.make_tensor_proto(data, shape=[batch_size, img[0].size], dtype=None))
    return float(time.time() - start)

def create_arr_batch(batch_size):
    print(f'Batch size: {img.shape}')
    batch = np.repeat(img[0], batch_size, axis=0).tolist()
    return batch

def serialize_str(batch_size):
    with open('0.png', 'rb') as payload:
        img_str = np.repeat(payload.read(), batch_size, axis=0).tolist()
    str_res = [prepare_grpc_request(model_name, signature_name, img_str) for _ in range(number_of_tests)]
    return str_res

def serialize_arr(batch_size):
    batch = create_arr_batch(batch_size)
    arr_res = [prepare_grpc_request(model_name, signature_name, batch) for _ in range(number_of_tests)]
    return arr_res

def test_requests_arr(mn, sn, data):
    """Test grpc request and receive and response

    """
    req = prepare_grpc_full_request(mn, sn)
    resp = stub.Predict(req, timeout=600)
    #print(f'{resp}')
    sys.stdout.write('.')
    sys.stdout.flush()
    return resp

def perform_multiple_arr_requests():
    """
    Perform multiple GRPC requests
    """
#    data = create_arr_batch(b)
    arr_resp = [test_requests_arr(model_name, signature_name, batch) for _ in range(number_of_tests)]
    return arr_resp

def run_str_serialization_tests():
    df = pd.DataFrame(index=range(number_of_tests))
    for batchsize in [4, 16, 64, 256, 1024]:
        print(f'str: serializing batch size {batchsize}')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(serialize_str, b) for b in [batchsize]] # , 16, 64, 256, 1024
            res = [f.result() for f in concurrent.futures.as_completed(futures)]
        df[f'{batchsize}']=np.reshape(res,(number_of_tests, 1)) #columns=[4, 16, 64, 256, 1024]
    print('saving results to csv file...')
    df.to_csv('str_batch_protobuf.csv')
    return

def run_arr_serialization_tests():
    df = pd.DataFrame(index=range(number_of_tests))
    for batchsize in [4, 16, 64, 256, 1024]:
        print(f'arr: serializing batch size {batchsize}')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(serialize_arr, b) for b in [batchsize]] # , 16, 64, 256, 1024
            res = [f.result() for f in concurrent.futures.as_completed(futures)]
        df[f'{batchsize}']=np.reshape(res,(number_of_tests, 1)) #columns=[4, 16, 64, 256, 1024]
    print('saving results to csv file...')
    df.to_csv('arr_batch_protobuf.csv')
    return

if __name__ == '__main__':
#    print(run_str_serialization_tests())
    # run_str_serialization_tests()
    # run_arr_serialization_tests()

    # Testing full response on tensorboard
    perform_multiple_arr_requests()

    # ToDo
    # Implement str based requests
