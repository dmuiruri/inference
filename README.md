# inference
Reviewing inference setup, effect of various variables to the inference rate in ML setups


## Client

### Image Building

There are two clients (gRPC and REST) which can be built to generate
independent traffic to the respective servers
```
sudo docker build . -t mnist_client_rest
sudo docker build . -t mnist_client_grpc
```
### Running the client containers

The client collects some statistics which we store in the VM in a
location shared between the client container(volume) and the VM.
```
sudo docker run --mount type=bind,source=/home/ubuntu/infer/client_grpc/data,target=/usr/src/app/data mnist_client_rest
sudo docker run --mount type=bind,source=/home/ubuntu/infer/client_grpc/data,target=/usr/src/app/data mnist_client_grpc
```

## Data Analysis

To avoid transfer of data from the remote VM, we open a Jupyter
Notebook in the remote VM for in-situ data processing.