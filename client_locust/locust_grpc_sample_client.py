# make sure you use grpc version 1.39.0 or later,
# because of https://github.com/grpc/grpc/issues/15880 that affected earlier versions
import grpc
import hello_pb2_grpc
import hello_pb2
from locust import events, User, task, run_single_user
from locust.exception import LocustError
from locust.user.task import LOCUST_STATE_STOPPING
# from hello_server import start_server # server has other name
import gevent
import time
import sys

# patch grpc so that it uses gevent instead of asyncio
import grpc.experimental.gevent as grpc_gevent

grpc_gevent.init_gevent()

# We've alreadt started the server manually
# @events.init.add_listener
# def run_grpc_server(environment, **_kwargs):
#     # Start the dummy server. This is not something you would do in a real test.
#     gevent.spawn(start_server)


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
                # request_meta["response_length"] = len(request_meta["response"])
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


class HelloGrpcUser(GrpcUser):
    host = "localhost:50051"
    stub_class = hello_pb2_grpc.HelloStub # HelloServicer

    @task
    def sayHello(self):
        if not self._channel_closed:
            # self.client.SayHello(hello_pb2.HelloRequest(name="Test"))
            response = self.client.Hello(hello_pb2.HelloRequest(value='Test'))
            # response = stub_class.Hello(request)
            # sys.stdout.write(response.value)
            # sys.stdout.flush()

if __name__ == '__main__':
    run_single_user(HelloGrpcUser)
