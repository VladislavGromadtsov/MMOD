import logging

import numpy as np
import pandas as pd

from channel import Channel
from request import Request


class Stats:
    def __init__(self, system):
        self.system = system

        self.queue_sizes = []
        self.working_channels = []
        self.total_requests = []
        self.requests = []
        self.request_queue_times = []
        self.request_times = []

        self.work_intervals = []
        self.process_intervals = []

        self.times_graphics = []
        self.finished_req_graphics = []
        self.cancelled_req_graphics = []
        self.running_req_graphics = []
        self.queue_req_graphics = []

        self.rejections = 0
        self.cancellations = 0

    def collect(self):
        cur_working_channels = 0
        for channel in self.system.channels:
            cur_working_channels += not channel.free

        cur_queue_size = len(self.system.queue)

        self.queue_sizes.append(cur_queue_size)
        self.working_channels.append(cur_working_channels)
        self.total_requests.append(cur_queue_size + cur_working_channels)

    def collect_for_graphics(self, cur_time, running_req, queue_size):
        self.times_graphics.append(cur_time)
        self.finished_req_graphics.append(len(self.requests))
        self.cancelled_req_graphics.append(self.cancellations)
        self.running_req_graphics.append(running_req)
        self.queue_req_graphics.append(queue_size)

    def cancel(self):
        self.cancellations += 1

    def reject(self):
        self.rejections += 1

    def add_work(self, interval):
        self.work_intervals.append(interval)

    def add_process(self):
        self.process_intervals.append(self)

    def out(self, request):
        self.requests.append(request)
        self.request_queue_times.append(request.time_in_queue)
        self.request_times.append(request.time_in_system)

    def build(self):
        d = {'Размер очереди': self.queue_sizes,
             'Занятые каналы': self.working_channels,
             'Заявки в системе': self.total_requests}

        d1 = {'Время запроса в очереди': self.request_queue_times,
              'Время запроса в системе': self.request_times}

        return pd.DataFrame(data=d), pd.DataFrame(data=d1)

    def get_cancel_prob(self):
        return self.cancellations / self.system.request_limit

    def get_states_probs(self):
        states = list(i for i in range(self.system.n + 1))
        states += list(i for i in range(self.system.n + 1, self.system.n + self.system.m + 1))

        state_counts = np.zeros(len(states))

        for req in self.total_requests:
            state_counts[req] += 1

        return states, state_counts


class System:
    def __init__(self, n, m, lambda_, mu, p, tick_size, request_limit, t):
        self.n = n
        self.m = m
        self.lambda_ = lambda_
        self.mu = mu
        self.p = p
        self.q = 1 - p
        self.tick_size = tick_size
        self.request_limit = request_limit
        self.t = t

        self.stats = Stats(self)

        self.request = 0
        self.channels = [
            Channel(self.mu, self.q, self.request_rejected) for _ in range(n)
        ]
        self.queue = []
        self.cur_time = 0.
        self.next_time = np.random.exponential(lambda_)

    def log(self):
        logging.info('Текущее время %.4f, следующий запрос поступит %.4f' %
                     (self.cur_time, self.next_time))

    def request_rejected(self, request=None):
        self.stats.reject()
        if request:
            self.push(request)

    def push(self, request=None):
        if not request:
            self.request += 1
            request = Request(self.cur_time)

        if len(self.queue) < self.m:
            request.enqueue(self.cur_time, 1. / self.t)
            self.queue.append(request)
        else:
            self.stats.cancel()
            logging.info('[Отменён] %s' % (request,))

    def free_channels(self):
        for channel in self.channels:
            request = channel.try_free(self.cur_time)
            if request:
                request.out(self.cur_time)
                self.stats.out(request)

    def dequeue_requests(self):
        for channel in self.channels:
            if not len(self.queue):
                return

            if channel.free:
                request = self.queue.pop(0)
                request.dequeue(self.cur_time)
                channel.run(self.cur_time, request)

    def free_all(self):
        while not self.is_all_free():
            self.tick()

    def is_all_free(self):
        for channel in self.channels:
            if not channel.free:
                return False
        return True

    def tick(self):
        self.free_channels()
        self.dequeue_requests()
        self.stats.collect()

        channels_in_work = 0
        for channel in self.channels:
            if not channel.free:
                channels_in_work += 1
        self.stats.collect_for_graphics(self.cur_time, channels_in_work, len(self.queue))

        self.cur_time += self.tick_size

    def check_queue(self):
        if not len(self.queue):
            return
        for request in self.queue:
            if self.cur_time >= request.time_waiting_in_queue:
                self.queue.remove(request)

    def run(self):
        while self.request < self.request_limit:
            if self.cur_time >= self.next_time:
                self.next_time = self.cur_time + \
                                 np.random.exponential(1 / self.lambda_)
                self.push()
                self.log()

            self.check_queue()
            self.tick()

        self.free_all()
