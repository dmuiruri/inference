# Inference Performance
In this project we implement a tool to test inference performance in a
machine learning setting.  The test setup makes use of Tensorflow
models and the
[tensorflow serving](https://www.tensorflow.org/tfx/guide/serving)
framework for inference serving.

<Add architectural image>

## Server
We start two independent servers from Tensorflow Serving, each serving the model from independent endpoints/ports
grpc server cmd: 
```
docker run -p 8500:8500 --mount type=bind,source=/tmp/mnist,target=/models/mnist -e MODEL_NAME=mnist --name tf_serving_mnist_grpc -t tensorflow/serving
```
rest server cmd:
```
docker run -p 8501:8501 --mount type=bind,source=/tmp/mnist,target=/models/mnist -e MODEL_NAME=mnist  --name tf_serving_mnist_rest -t tensorflow/serving
```

## Clients

### Locust Clients

Build the clients based on the respective Dockefiles
```
sudo docker build . -f rest.Dockerfile -t dmuiruri/rest_batch_locust
sudo docker build . -f grpc.Dockerfile -t dmuiruri/grpc_batch_locust
```

We use locust performance testing framework to generate traffic and to
collect statistics. An example of launching a locust client where the
--host flag indicates the server IP address where TFServing is running.

The BATCHSIZE and SERVER environment variables allows the user to
configure these settings locally. Testing different batch sizes is
facilitated by changing the BATCHSIZE but the logs will be overwritten
after every run -(move/save previous logs when necessary).

The --mount flag ensures the user can access the logs from the
container by providing a local directory where the container can store
the logs.

```
docker run --mount type=bind,source=/home/stats,target=/usr/src/app/stats -e "SERVER=127.x.x.x" -e "BATCHSIZE=4" rest_batch_locust

```
The resulting statistics files are stored in a stats folder.

Currently there are four type of user clients:
* REST single (Multiple users each issuing a single rest request
* gRPC single (Multiple users each issuing a single grpc requst
* REST batch (Multiple users each issuing a batch reqeuests to the REST endpoint
* gRPC batch (Multiple usesrs each issuing batch requests to the gRPC endpoint

### Basic Clients

Image Building

There are two clients (gRPC and REST) which can be built to generate
independent traffic to the respective servers
```
sudo docker build . -t mnist_client_rest
sudo docker build . -t mnist_client_grpc
```
#### Running the client containers

The client collects some statistics which we store in the VM in a
location shared between the client container(volume) and the VM.
```
sudo docker run --mount type=bind,source=/home/ubuntu/infer/client_grpc/data,target=/usr/src/app/data mnist_client_rest
sudo docker run --mount type=bind,source=/home/ubuntu/infer/client_grpc/data,target=/usr/src/app/data mnist_client_grpc
```
## Data Analysis

To avoid transfer of data from the remote VM, we open a Jupyter
Notebook in the remote VM for in-situ data processing.

In the remote VM open a browserless Notebook session to be served at port 8889
```
jupyter notebook --no-browser --port=8889
```

Enable port forwarding to the remote VM port from the local machine
```
ssh -N -f -L localhost:8888:localhost:8889 username@<CSC_public_ip/your_remote_host_name> -i <path/to/public/key/file>
```

The notebook can now be accessed from the local machine through the browser by accessing the address below
```
localhost:8888
```
### Locust performance data
Performance statistics can be collected from locust after obtaining running locust based clients.
Running the locust client using the command shown in the clients section results in the files shown below.
- client_locust/grpc_batch_exceptions.csv
- client_locust/grpc_batch_failures.csv
- client_locust/grpc_batch_stats.csv
- client_locust/grpc_batch_stats_history.csv
The fourth file contains historical statistics which are used for  analysis.

## Traffic Analysis (Low level)

Installed wireshark, tshark (wireshark cli) and tcpdump(available by default in unix)

The goal is to capture the traffic remotely( VM and Container) and the
files can be analyzed locally.  When recording the pcap file `tshark
-i ens3 -w capture-output.pcap` may require the output file to be
created somewhere else other than the user directory and the file
contains the right permissions to facilitate `scp` of the file to a
local machine.

There is a good
[tutorial](https://opensource.com/article/20/1/wireshark-linux-tshark)
on how to use tshark.

## Monitoring

Prometheus is adopted for monitoring of the server containers.

Docker already [exposes
prometheus](https://prometheus.io/docs/instrumenting/exporters/) like
metrics that allows the prometheus server to pull metrics from docker
without other additional tooling. It is however required to run docker
in
[experimental](https://docs.docker.com/engine/reference/commandline/dockerd/)
mode and create a daemon.json file under /etc/docker/. The json
contains the key value items.

```
{
    "metrics-addr": "127.0.0.1:9323",
    "experimental": true
}
```

After this has been file is created, the docker daemon needs to be
restarted. If any containers were running, they need to be stopped
otherwise they will be zombied and not visible from the container
listings.

```
sudo systemctl restart docker
```
After successful restart, running the command ```docker version```
will indicated Experimental:true listed in the output.
