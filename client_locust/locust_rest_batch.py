#! /usr/bin/env python

"""This script contains performance tests implimented using the
locust framework.

Supports more advanced user simulation features.

"""

import time
import numpy as np
import pandas as pd
import json
import sys
import subprocess
import os
import tensorflow as tf


from locust import HttpUser, FastHttpUser, task, tag, between, stats, run_single_user
import mnist_input_data

stats.CSV_STATS_INTERVAL_SEC = 1 # default is 1 second
stats.CSV_STATS_FLUSH_INTERVAL_SEC = 10 # Determines how often the data is flushed to disk, default is 10 seconds
stats.PERCENTILES_TO_REPORT = [25, 50, 75, 95]
work_dir = './tmp'

test_data_set = mnist_input_data.read_data_sets(work_dir).test
batch_size = int(os.environ['BATCHSIZE'])
img, labels = test_data_set.next_batch(batch_size)
batch = np.repeat(img, 1, axis=0).tolist()
# print(f'labels: {labels}, image size: {img.shape}, image type: {type(img)}, batch size: {len(batch)}')

json_data = {
    "signature_name": 'predict_images',
    "instances": batch
}

class restClientBatch(HttpUser):
    """
    A http user class to run batch reqeusts on a model endpoint
    """
    # host = 'http://128.214.252.11' # change if different host
    host = os.environ['SERVER']
    
    @tag('batchinference')
    @task
    def predict_batch(self):
        """
        Get prediction in a batch
        """
        response_prediction = self.client.post(':8501/v1/models/mnist:predict', json=json_data)
        # inference = response_prediction.json()['predictions']
        # print(f"batchsize: {len(inference)}, predictions: {[np.argmax(inf) for inf in inference]}")
        return

if __name__ == "__main__":
    run_single_user(restClientBatch)
    # cmd = 'locust -f locust_rest_batch.py --headless --csv=rest --csv-full-history -u 100 -r 10 --run-time 5m'
    # subprocess.run(cmd, shell=True)
