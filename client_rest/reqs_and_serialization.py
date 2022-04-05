#! /usr/bin/env python

import numpy as np
import pandas as pd
import requests
import base64
import concurrent.futures
from time import time

import mnist_input_data

work_dir = '/tmp'
test_data_set = mnist_input_data.read_data_sets(work_dir).test
number_of_tests = 500

def create_arr_batch(batch_size):
    """Create a batch of a given size

    """
    img, label = test_data_set.next_batch(batch_size)
    # print(f'Ground Truth: {label}')
    batch = img.tolist()
    return batch

def test_requests_arr(json_data):
    """
    Perform a single complete request to the server
    """
    start = time()
    with requests.Session() as sess:
        req = requests.Request("post", 'http://128.214.252.11:8501/v1/models/mnist:predict', json=json_data)
        prepared_req = sess.prepare_request(req)
        response = sess.send(prepared_req)
    # print(f'prediction: {[np.argmax(i) for i in response.json()["predictions"]]}')
    return float(time() - start)

def perform_multiple_arr_requests(batch):
    """Send multiple requests.

    For statistical stability of results, we perform multiple requests
    to get a general distribution of the performance.

    """
    batch = create_arr_batch(batch)
    json_data = {
        "signature_name": 'predict_images',
        "instances": batch
    }
    arr_res = [test_requests_arr(json_data) for _ in range(number_of_tests)]
    return arr_res

def run_arr_requests():
    """Test using multiple batch sizes

    Collect statistics from multiple batches sizes to observe the
    impact of increasing batch sizes.

    TODO: We can make use of threads to concurrently execute
    performance where each batch size gets its own thread, the
    challenge is that the threads may begin to impact the overall
    performance unless we can send each thread to its own core. So for
    now only one threading code is redundant.

    """
    df = pd.DataFrame(index=range(number_of_tests))
    for batchsize in [4, 16, 64, 256, 1024]: #
        print(f'arr batch size: {batchsize}')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(perform_multiple_arr_requests, b) for b in [batchsize]]
            #f = concurrent.futures.as_completed(futures)
            res = [f.result() for f in concurrent.futures.as_completed(futures)]
        df[f'{batchsize}'] = np.reshape(res, (number_of_tests, 1))
    print('Saving results to csv file')
    df.to_csv('arr_batch_req_json.csv')
    return
    
if __name__ == '__main__':
    run_arr_requests()
