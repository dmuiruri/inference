FROM python:3.9

WORKDIR /usr/src/app

RUN mkdir ./data

RUN  apt update
RUN  pip install --upgrade pip

RUN  pip install numpy
RUN  pip install pandas
RUN  pip install requests
RUN  pip install tensorflow
RUN  pip install tensorflow-serving-api

# Install relevant packages for asynchronous REST requests
RUN  pip install aiohttp
RUN  pip install asyncio
RUN  pip install aiohttp
RUN  pip install cchardet
RUN  pip install aiodns

COPY . .

#CMD ["python", "./mnist_client.py", "--num_tests=10", "--server=128.214.252.11:8500", "--concurrency=1"]
CMD ["python", "./mnist_client.py", "--server=128.214.252.11:8500"]
