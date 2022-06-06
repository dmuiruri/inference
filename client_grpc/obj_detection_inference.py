import tensorflow as tf
import grpc
import time
import requests
import numpy as np
import sys
from PIL import Image
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from grpc.beta import implementations

model_name = 'centernet_hg_1024'
signature_name = 'serving_default'
hostport = '8500'
input_name = 'input_tensor'
input_type = None
number_of_tests = 200

channel = grpc.insecure_channel('128.214.252.11:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

def load_image_into_tensor(path):
    """Load an image from file into a tensor

    An image can be opened as a file object which will result in
    string (byte string) or an image can be openeded as an array. In
    this case we open the image into a numpy array and convert it into
    a tensor.

    """
    image_np = np.array(Image.open(path))
    # img_batch = np.repeat(image_np, batch_size, axis=0)#.tolist() How to perform batch inference in these models.
    print(f'image file opened')
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
    print('tensor created')
    #return input_tensor.numpy().tolist()
    return input_tensor

def prepare_grpc_full_request():
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = signature_name
    # img, label = test_data_set.next_batch(batchsize)
    img = load_image_into_tensor('./image3.jpg')
    request.inputs[input_name].CopyFrom(
        tf.make_tensor_proto(img, dtype=None)) #tf.make_tensor_proto(img[0], shape=[1, img[0].size], dtype=None))
    return request

def test_requests_arr(): #data
    """Test grpc request and receive and response

    """
    req = prepare_grpc_full_request()
    resp = stub.Predict(req, timeout=600)
    #print(f'{resp}')
    sys.stdout.write('.')
    sys.stdout.flush()
    return resp

if __name__ == '__main__':
    print(test_requests_arr())
