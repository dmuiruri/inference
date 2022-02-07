#! /usr/bin/env python

"""This script contains performance tests implimented using the
locust framework.

Supports more advansed user simulation features.

"""
from __future__ import print_function

import sys
import threading

# This is a placeholder for a Google-internal import.

import grpc
import numpy
import tensorflow as tf
import pandas as pd
import time

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import mnist_input_data
import grpc.experimental.gevent as grpc_gevent

grpc_gevent.init_gevent()
tf.compat.v1.app.flags.DEFINE_integer(
    'concurrency', 1, 'maximum number of concurrent inference requests')
#tf.compat.v1.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.compat.v1.app.flags.DEFINE_string('server', '',
                                     'PredictionService host:port')
tf.compat.v1.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.compat.v1.app.flags.FLAGS

from locust import HttpUser, User, task, tag, between, stats, run_single_user

stats.CSV_STATS_INTERVAL_SEC = 1 # default is 1 second
stats.CSV_STATS_FLUSH_INTERVAL_SEC = 10 # frequency of data flushing to disk, default is 10 seconds

work_dir = '/tmp'
# host = 'http://128.214.252.11'
# grpcport = '8500'
test_data_set = mnist_input_data.read_data_sets(work_dir).test

# channel = grpc.insecure_channel(host)
# stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
# request = predict_pb2.PredictRequest()
# request.model_spec.name = 'mnist'
# request.model_spec.signature_name = 'predict_images'
image, label = test_data_set.next_batch(1)

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
                # request_meta["response_length"] = len(request_meta["response"].message)
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
    host = "128.214.252.11:8500"
    # host = 'http://128.214.252.11:8500' # /v1/models/mnist:predict
    stub_class = prediction_service_pb2_grpc.PredictionServiceStub #(channel)
    request = predict_pb2.PredictRequest()

    @task
    def predict_single(self):
        """
        Get prediction for a single image
        """
        if not self._channel_closed:
            sys.stdout.write('.')
            sys.stdout.flush()
            self.request.model_spec.name: "mnist"
            self.request.model_spec.signature_name: "predict_images"
            self.request.inputs['images'].CopyFrom(
                tf.make_tensor_proto(image[0], shape=[1, image[0].size]))
            # result_future = stub.Predict.future(request, 5.0)  #5 seconds
            response = self.client.Predict(self.request, 5.0)  #5 seconds
        return

# class grpcClientSingle(HttpUser):
#     """A http user class to run HTTP requests to the model's REST endpoint

#     """
#     host = 'http://128.214.252.11'

#     @tag('single_inference_grpc')
#     @task
#     def predict_single(self):
#         """
#         Get prediction for a single image
#         """
#         # sys.stdout.write('.')
#         # sys.stdout.flush()
#         request.inputs['images'].CopyFrom(
#             tf.make_tensor_proto(image[0], shape=[1, image[0].size]))
#         result_future = stub.Predict.future(request, 5.0)  # 5 seconds
#         return

if __name__ == "__main__":
    run_single_user(SingleGrpcUser)
    # cmd = 'locust -f locust_grpc_single.py --headless --csv=rest --csv-full-history -u 100 -r 10 --run-time 5m'
    # subprocess.run(cmd, shell=True)
