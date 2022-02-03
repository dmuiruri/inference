#! /usr/bin/env python

"""This script contains performance tests implimented using the
locust framework.

Supports more advansed user simulation features.

"""

import time
import numpy as np
import pandas as pd
import json
import sys
import subprocess

from locust import HttpUser, FastHttpUser, task, tag, between, stats, run_single_user
import mnist_input_data

stats.CSV_STATS_INTERVAL_SEC = 1 # default is 1 second
stats.CSV_STATS_FLUSH_INTERVAL_SEC = 10 # Determines how often the data is flushed to disk, default is 10 seconds

work_dir = '/tmp'

test_data_set = mnist_input_data.read_data_sets(work_dir).test
smallBatchSize = 1
img_s, label_s = test_data_set.next_batch(smallBatchSize)
smallBatch = np.repeat(img_s, smallBatchSize, axis=0).tolist()
json_data_s = {
    "signature_name": 'predict_images',
    "instances": smallBatch
}

largeBatchSize = 16
img_l, label_l = test_data_set.next_batch(largeBatchSize)
largeBatch = np.repeat(img_l, largeBatchSize, axis=0).tolist()
json_data_l = {
    "signature_name": 'predict_images',
    "instances": largeBatch
}

# url = ':8501/v1/models/mnist:predict'

class httpClientSingle(HttpUser):
    """A http user class to run HTTP requests to the model's REST endpoint

    """
    host = 'http://128.214.252.11'

    @tag('singleinference')
    @task
    def predict_single(self):
        """
        Get prediction for a single image
        """
        # sys.stdout.write('.')
        # sys.stdout.flush()
        response_prediction = self.client.post(':8501/v1/models/mnist:predict', json=json_data_s)
        return

class httpClientBatch(HttpUser):
    """
    A http user class to run batch reqeusts on a model endpoint
    """
    host = 'http://128.214.252.11'
    
    @tag('batchinference')
    @task
    def predict_batch(self):
        """
        Get prediction in a batch
        """
        response_prediction = self.client.post(':8501/v1/models/mnist:predict', json=json_data_l)
        return
        
# class fastHttpClient(FastHttpUser):
#     """A user based on geventhttpclient with support faster but increases
#     the number of requests

#     """
#     time_limit = 2
    
#     @tag('fasthttpuser')
#     @task
#     def predict_single(self):
#         response_prediction = self.client.post('http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
#         return

if __name__ == "__main__":
    run_single_user(httpClientSingle)
    run_single_user(httpClientBatch)
    # cmd = 'locust -f locustfile.py --headless --csv=rest --csv-full-history -u 100 -r 10 --run-time 5m'
    # subprocess.run(cmd, shell=True)
