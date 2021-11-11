import random
import logging

import numpy as np


class Channel:
    COUNT = 0

    def __init__(self, mu, reject_probability, on_reject):
        Channel.COUNT += 1
        self.id = Channel.COUNT

        self.reject_probability = reject_probability
        self.on_reject = on_reject
        self.mu = mu

        self.free = True
        self.end_at = 0
        self.request = None

    def run(self, start_at, request):
        self.free = False
        self.end_at = start_at + np.random.exponential(1 / self.mu)
        self.request = request
        logging.info('[Стартовал] Канал #%d: с %.4f до %.4f' %
                     (self.id, start_at, self.end_at))

    def try_free(self, cur_time):
        if not self.free and self.end_at < cur_time:
            self.free = True

            rejected = random.random() <= self.reject_probability
            if rejected:
                self.on_reject(self.request)
                logging.info('[Отклонено] Канал #%d: освободился в %.4f' %
                             (self.id, cur_time))
            else:
                logging.info('[Выполнено] Канал #%d: осободился в %.4f' %
                             (self.id, cur_time))
                return self.request
