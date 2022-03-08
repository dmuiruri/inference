#! /usr/bin/env python

"""This script contains performance tests implimented using the
locust framework.

Supports more advansed user simulation features.

"""
import time
import numpy as np
import json
import sys
import subprocess
import requests
import matplotlib.image as im

from locust import HttpUser, FastHttpUser, task, tag, between, stats, run_single_user
#import mnist_input_data

stats.CSV_STATS_INTERVAL_SEC = 1 # default is 1 second
stats.CSV_STATS_FLUSH_INTERVAL_SEC = 10 # Determines how often the data is flushed to disk, default is 10 seconds

img_fh = open('9.png', 'rb') # ToDo: load data dynamically
img = img_fh.read()

# batch_size = 1
# batch = np.repeat(img, batch_size, axis=0).tolist()
# json_data = {
#     "data": batch
# }

class httpClientSingle(HttpUser):
    """A http user class to run HTTP requests to the model's REST endpoint

    """
    host = 'http://128.214.252.11:8080'

    @tag('singleinference')
    @task
    def predict_single(self):
        """
        Get prediction for a single image
        """
        response_prediction = self.client.post('/predictions/mnist', data=img)
        return

if __name__ == "__main__":
    run_single_user(httpClientSingle)
    # cmd = 'locust -f locust_rest_single.py --headless --csv=rest --csv-full-history -u 100 -r 10 --run-time 5m'
    # subprocess.run(cmd, shell=True)
