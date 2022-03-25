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

from time import time

import mnist_input_data

work_dir = '/tmp'

test_data_set = mnist_input_data.read_data_sets(work_dir).test
img, label = test_data_set.next_batch(1)

batch_size = 1
batch = np.repeat(img, batch_size, axis=0).tolist()

json_data = {
    "signature_name": 'predict_images',
    "instances": batch
}

response_times = list()

def get_predictions():
    """
    Inference querying
    """
    sys.stdout.write('.')
    sys.stdout.flush()
    response_prediction = requests.post('http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
    # Predict returns the probabilities of the classes 0-9, so we need to pick the highest probability
    # number = np.argmax(response_prediction.json()['predictions'][0])
    # return response_prediction.elapsed.total_seconds()
    return


conn = aiohttp.TCPConnector(limit=0)

async def _range(num):
    """
    The asyncio library cannot use a regular range object we need to implement an acceptation version

    TypeError: 'async for' requires an object with __aiter__ method, got range
    """
    for i in range(num):
        yield i

async def async_inference(session):
    """
    Get prediction asynchronously
    """
    start = time()
    async with session.post('http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data) as resp:
        sys.stdout.write('.')
        # sys.stdout.write(resp.json())
        sys.stdout.flush()
        # resp_status = resp.status
        # resp_time = resp.elapsed.total_seconds()
        await resp.text()
    elapsed = time() - start
    response_times.append(elapsed)#resp_time

# async def async_collect_inferences(num=10):
#     """
#     Dispatch inferences

#     We need 10000 tests
#     """
#     url = 'http://128.214.252.11:8501/v1/models/mnist:predict'
#     async with aiohttp.ClientSession(connector=conn) as session:
#         post_tasks = []
#         async for _ in _range(num):
#             post_tasks.append(async_inference(session))
#         await asyncio.gather(*post_tasks) # send all at once

async def async_collect_inferences(num=10):
    """
    Dispatch inferences

    We need 10000 tests
    """
    url = 'http://128.214.252.11:8501/v1/models/mnist:predict'
    async with aiohttp.ClientSession(connector=conn) as session:
        post_tasks = []
        async for _ in _range(num):
            post_tasks.append(async_inference(session))
        await asyncio.gather(*post_tasks) # send all at once


if __name__ == '__main__':
    # Here we test using timeit but it is synchronous
    # iterations = 1000
    # result = timeit.timeit(get_predictions, number=iterations)
    # print(f"\n iterations: {iterations} time spent: {result}, {1000 * (result/iterations)}")

    # Here we use asyncio to create asynchronous requests
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_collect_inferences(num=1000))
    print(f'\n {len(response_times)}')
    np.save('./client_rest/data/response.npy', response_times)
