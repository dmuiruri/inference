#! /usr/bin/env python

"""This script contains performance tests implimented using the
locust framework.

Supports more advansed user simulation features.
"""

from __future__ import print_function

import sys
import threading
import numpy as np
import grpc
import numpy
import tensorflow as tf
import pandas as pd
import time
import mnist_input_data
import grpc.experimental.gevent as grpc_gevent
import os

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from locust import HttpUser, User, task, tag, between, stats, run_single_user

grpc_gevent.init_gevent()

# stats.CSV_STATS_INTERVAL_SEC = 1 # default is 1 second
# stats.CSV_STATS_FLUSH_INTERVAL_SEC = 10 # frequency of data flushing to disk, default is 10 seconds

work_dir = './tmp'
test_data_set = mnist_input_data.read_data_sets(work_dir).test

batch_size = int(os.environ['BATCHSIZE'])
image, label = test_data_set.next_batch(batch_size)
batch = np.repeat(image[0], batch_size, axis=0).tolist()
print(label, image[0].size)

class GrpcClient:
    def __init__(self, environment, stub):
        self.env = environment
        self._stub_class = stub.__class__
        self._stub = stub

    def __getattr__(self, name):
        func = self._stub_class.__getattribute__(self._stub, name)

        def wrapper(*args, **kwargs):
            request_meta = {
                "request_type": "grpc",
                "name": name,
                "start_time": time.time(),
                "response_length": 0,
                "exception": None,
                "context": None,
                "response": None,
            }
            start_perf_counter = time.perf_counter()
            try:
                request_meta["response"] = func(*args, **kwargs)
                #request_meta["response_length"] = len(request_meta["response"].message)
            except grpc.RpcError as e:
                request_meta["exception"] = e
            request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000
            self.env.events.request.fire(**request_meta)
            return request_meta["response"]

        return wrapper

class GrpcUser(User):
    abstract = True
    stub_class = None

    def __init__(self, environment):
        super().__init__(environment)
        for attr_value, attr_name in ((self.host, "host"), (self.stub_class, "stub_class")):
            if attr_value is None:
                raise LocustError(f"You must specify the {attr_name}.")
        self._channel = grpc.insecure_channel(self.host)
        self._channel_closed = False
        stub = self.stub_class(self._channel)
        self.client = GrpcClient(environment, stub)

class SingleGrpcUser(GrpcUser):
    #host = "128.214.252.11:8500"
    host = os.environ['SERVER']
    stub_class = prediction_service_pb2_grpc.PredictionServiceStub
    request = predict_pb2.PredictRequest()
    def __init__(self):
        self.request = None

    def prepare_grpc_request(self, model_name, signature_name, data):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.model_spec.signature_name = signature_name
        request.inputs['images'].CopyFrom(
            tf.make_tensor_proto(data, shape=[batch_size, image[0].size], dtype=None))
        return request

    self.request = self.prepare_grpc_request('mnist', 'predict_images', batch)

    @task
    def predict_single(self):
        """
        Get prediction for a single image
        """
        if not self._channel_closed:

            # Returns a PredictResponse Object which contains the
            # probabilities of the classes 0-9, so we need to pick the
            # highest probability to determine the prediction.
            response = self.client.Predict(self.request, timeout=None)  #5 seconds
            print(response)
            time.sleep(20)
            return


if __name__ == "__main__":
    run_single_user(SingleGrpcUser)
    # cmd = 'locust -f locust_grpc_single.py --headless --csv=grpc --csv-full-history -u 100 -r 10 --run-time 5m --stop-timeout=60'
    # subprocess.run(cmd, shell=True)
