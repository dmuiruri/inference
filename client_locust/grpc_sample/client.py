#!/usr/bin/env python

"""
gRPC client
"""

import grpc
import hello_pb2
import hello_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = hello_pb2_grpc.HelloStub(channel) # create a gRPC client
request = hello_pb2.HelloRequest(value="World") # create a request 
response = stub.Hello(request) # call Hello service, return HelloResponse

print(response.value)  # "Hello World"
