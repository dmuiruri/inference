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
#response_times = list()

test_data_set = mnist_input_data.read_data_sets(work_dir).test

def prepare_data():
    img, label = test_data_set.next_batch(1)
    batch_size = 1
    batch = np.repeat(img, batch_size, axis=0).tolist()
    test_data = {
        "signature_name": 'predict_images',
        "instances": batch
    }
    return test_data

json_data = prepare_data()

def get_predictions():
    """Inference querying
    
    Predict returns the probabilities of the classes 0-9, so we need
    to pick the highest probability

    number = np.argmax(response_prediction.json()['predictions'][0])
    """
    # sys.stdout.write('.')
    # sys.stdout.flush()
    # start = time()
    # response_prediction = requests.post('http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
    # elapsed = time() - start
    return requests.post('http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data).elapsed.total_seconds()

def run_performance_tests(iterations=10):
    return [get_predictions() for _ in range(iterations)]

if __name__ == '__main__':
    print(run_performance_tests())
