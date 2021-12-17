#! /usr/bin/env python

"""This script contains performance tests implimented using the
locust framework.

Supports more advansed user simulation features.

"""

import time
import numpy as np
import pandas as pd
import json
from locust import HttpUser, task, between

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


class httpClient(HttpUser):
    """A http user class to run HTTP requests to the model's REST endpoint

    """
    @task
    def predict_single(self):
        """Get prediction for a single image from a re

        """
        sys.stdout.write('.')
        sys.stdout.flush()
        response_prediction = self.client.post('http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
        return


if __name__ == "__main__":
    pass
