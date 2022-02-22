# Instructions to build a Docker image for a locust grpc client to
# test server performance

FROM python:3

WORKDIR /usr/src/app

# Environment variables
ENV SERVER=http://127.0.0.1
ENV BATCHSIZE=1

RUN mkdir ./tmp
RUN mkdir ./stats

RUN  apt update
RUN  pip install --upgrade pip

RUN  pip install numpy
RUN  pip install pandas
RUN  pip install locust
RUN  pip install grpcio
RUN  pip install tensorflow
RUN  pip install tensorflow-serving-api

COPY ./mnist_input_data.py .
COPY ./locust_grpc_batch.py .
# Copy locust config file, configurations can be overriden by env variables
#COPY

RUN ulimit -n 200000

CMD locust -f locust_grpc_batch.py --headless --csv=stats/rest --csv-full-history -u 500 -r 10 --run-time 5m