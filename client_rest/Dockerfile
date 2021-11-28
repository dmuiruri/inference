FROM python:3.9

WORKDIR /usr/src/app

RUN mkdir ./data

RUN  apt update
RUN  pip install --upgrade pip

RUN  pip install numpy
RUN  pip install pandas
RUN  pip install requests

# Install relevant packages for asynchronous REST requests
RUN  pip install aiohttp
RUN  pip install asyncio
RUN  pip install aiohttp
RUN  pip install cchardet
RUN  pip install aiodns

COPY . .

CMD ["python", "mnist_client.py"]
