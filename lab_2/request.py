import logging
import numpy as np


class Request:
    COUNT = 0

    def __init__(self, cur_time):
        Request.COUNT += 1
        self.id = Request.COUNT
        self.time_in_queue = 0
        self.time_in_system = 0
        self.start_in_queue = None
        self.start_in_system = cur_time
        self.time_waiting_in_queue = None

    def enqueue(self, cur_time, v):
        self.start_in_queue = cur_time
        self.time_waiting_in_queue = self.start_in_queue + np.random.exponential(1 / v)
        logging.info('[Поступил] Запрос #%d в %.4f' % (self.id, cur_time))

    def dequeue(self, cur_time):
        self.time_in_queue += cur_time - self.start_in_queue
        logging.info('[Выполняется] Запрос #%d в %.4f' % (self.id, cur_time))

    def out(self, cur_time):
        self.time_in_system += cur_time - self.start_in_system

    def __str__(self):
        return 'Запрос #%d: в очереди %.4f, в системе %.4f' % \
               (self.id, self.time_in_queue, self.time_in_system)
