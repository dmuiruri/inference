# inference
Reviewing inference setup, effect of various variables to the inference rate in ML setups

## TODO Items

* Conduct benchmark against dedicated [gRPC](https://ghz.sh/) test tool
* Conduct benchmark against a [REST](https://www.bswen.com/2019/08/others-Use-Apache-Bench(ab)-command-to-test-RESTful-apis-example.html) API testing tool
Results from these tools help to  establish the reliability of our own implementation.
* Test the feasibility of using [wireshark](https://www.wireshark.org/) to see low level traffic

Open Questions
* How much of that time is spent by the model inference itself not transport layer handshakes
* Is this a CPU or IO bound problem
* Is the server applying some form of caching
* Impact of schema and data type (JSON, Binary, String etc)


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

## Traffic Analysis (Low level)

Installed wireshark, tshark (wireshark cli) and tcpdump(available by default in unix)

The goal is to capture the traffic remotely( VM and Container) and the
files can be analyzed locally.  When recording the pcap file `tshark
-i wlan0 -w capture-output.pcap` may require the output file to be
created somewhere else other than the user directory.

There is a good
[tutorial](https://opensource.com/article/20/1/wireshark-linux-tshark)
on how to use tshark.