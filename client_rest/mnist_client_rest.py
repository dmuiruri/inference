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


def get_predictions():
    """
    Inference querying
    """
    sys.stdout.write('.')
    sys.stdout.flush()
    response_prediction = requests.post('http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
    # Predict returns the probabilities of the classes 0-9, so we need to pick the highest probability
    # number = np.argmax(response_prediction.json()['predictions'][0])
    return response_prediction.elapsed.total_seconds()
    

if __name__ == '__main__':
    for _ in range(100):
        resp_time_sec = get_predictions()

    print(f"\n {resp_time_sec}")
