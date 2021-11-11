import os
import logging

import numpy as np
import pandas as pd
import markdown_generator as mg
import matplotlib.pyplot as plt
import math


from datetime import datetime


from system import System

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)


def get_free_probability(p, n, m, beta):
    p0 = 1

    for k in range(1, n + 1):
        p0 += p ** k / math.factorial(k)

    sum = 0
    i = 1
    while i <= m:
        p_up = p ** i
        l = 1
        compos = 1
        while l <= i:
            compos *= (n + l * beta)
            l += 1
        sum += p_up / compos
        i += 1

    p0 += ((p ** n) / math.factorial(n)) * sum
    return 1 / p0


def get_pn_probability(p, n, m, beta):
    p0 = get_free_probability(p, n, m, beta)
    return p0 * (p ** n) / math.factorial(n)


def get_state_probs(rho, n, m, beta):
    p0 = get_free_probability(rho, n, m, beta)
    probs = [p0]

    for k in range(1, n + 1):
        probs.append(rho ** k / math.factorial(k) * p0)

    pn = get_pn_probability(rho, n, m, beta)
    for i in range(1, m + 1):
        l = 1
        compos = 1
        while l <= i:
            compos *= (n + l * beta)
            l += 1
        probs.append(pn * (rho ** i) / compos)

    return probs


def get_cancel_prob(p, n, m, beta):
    pn = get_pn_probability(p, n, m, beta)
    l = 1
    denominator = 1
    while l <= m:
        denominator *= (n + l * beta)
        l += 1
    return pn * (p ** m) / denominator


def get_theor_interval_len(n, m, probs):
    sum = 0
    for i in range(0, m):
        sum += ((i + 1) * probs[n + 1 + i])
    return sum


def get_theor_channel_loaded(n, m, probs):
    sum = 0

    for k in range(0, n):
        sum += probs[k + 1] * (k + 1)

    for i in range(0, m):
        sum += n * probs[n + i + 1]

    return sum


def get_overall_request_time(theor_relative_bandwidth, mu, t):
    return theor_relative_bandwidth / mu + t


def generate_report(lambda_, mu, p, n, m, t, stats):
    path = 'res'
    hists_dir_name = 'hists'
    hists_path = os.path.join(path, hists_dir_name)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(hists_path):
        os.makedirs(hists_path)
    time_now = datetime.now().strftime('%d%m%Y_%H%M%S')
    filename = 'result_%s.md' % (time_now,)
    hist_name = time_now + '.png'
    hist_name_2 = time_now + '-2' + '.png'
    hist_name_3 = time_now + '-3' + '.png'

    with open(os.path.join(path, filename), 'w', encoding='utf-8') as f:
        doc = mg.Writer(f)
        doc.write_heading('Статистика')
        doc.writelines([
            'λ = %.2f' % lambda_, '',
            'μ = %.2f' % mu, '',
            'n = %d, m = %d' % (n, m), ''
        ])

        df_c, df_times = stats.build()
        doc.writelines([
            df_c.describe().T.to_markdown(), '',
        ])

        doc.writelines([
            'Всего отменено: %d' % stats.cancellations, '',
            'Всего выполнено: %d' % len(stats.requests), ''
        ])

        states_bins, states_counts = stats.get_states_probs()
        _rho = lambda_ / mu

        plt.xticks(states_bins)
        plt.hist(stats.total_requests, bins=np.array(states_bins) - 0.5, density=True)
        plt.savefig(os.path.join(hists_path, hist_name))

        beta = 1 / (t * mu)
        probs = get_state_probs(_rho, n, m, beta)

        doc.writelines([
            'Вероятности для состояний системы:',
            '![hist](%s)' % (os.path.join(hists_dir_name, hist_name),), '',
            pd.DataFrame(data={
                'Теоретическая вероятность': probs,
                'Практическая вероятность': states_counts / sum(states_counts)
            }).T.to_markdown(), ''
        ])

        plt.plot(stats.times_graphics, stats.finished_req_graphics, label="Finished")
        plt.plot(stats.times_graphics, stats.cancelled_req_graphics, label="Cancelled")
        plt.legend()
        plt.savefig(os.path.join(hists_path, hist_name_2))

        doc.writelines([
            'Рост числа выполненных и отменённых заявок со временем:',
            '![graph](%s)' % (os.path.join(hists_dir_name, hist_name_2),), ''
        ])

        plt.clf()
        plt.plot(stats.times_graphics, stats.running_req_graphics, label="Running")
        plt.plot(stats.times_graphics, stats.queue_req_graphics, label="In queue")
        plt.xlim(max(stats.times_graphics) - 20, max(stats.times_graphics))
        plt.legend()
        plt.savefig(os.path.join(hists_path, hist_name_3))

        doc.writelines([
            'Количество заявок в каналах и очереди в течение времени выполнения:',
            '![graph](%s)' % (os.path.join(hists_dir_name, hist_name_3),), ''
        ])

        cancel_prob = stats.get_cancel_prob()
        theor_cancel_prob = get_cancel_prob(_rho, n, m, beta)
        relative_bandwidth = 1 - cancel_prob
        theor_relative_bandwidth = 1 - theor_cancel_prob
        absolute_bandwidth = relative_bandwidth * lambda_
        theor_absolute_bandwidth = lambda_ * theor_relative_bandwidth
        theor_queue_size = get_theor_interval_len(n, m, probs)
        theor_channel_loaded = get_theor_channel_loaded(n, m, probs)
        theor_system_load = theor_queue_size + theor_channel_loaded

        doc.writelines([
            pd.DataFrame({
                'Вероятность отказа': [theor_cancel_prob, cancel_prob],
                'Относительная пропускная способность': [theor_relative_bandwidth, relative_bandwidth],
                'Абсолютная пропускная способность': [theor_absolute_bandwidth, absolute_bandwidth],
                'Длина очереди': [theor_queue_size, np.mean(stats.queue_sizes)],
                'Количество занятых каналов': [theor_channel_loaded, np.mean(stats.working_channels)],
                'Количество заявок в системе': [theor_system_load, np.mean(stats.total_requests)],
            }, index=['Теор.', 'Практ.']).T.to_markdown(), ''
        ])

        doc.writelines([
            df_times.describe().T.to_markdown(), ''
        ])

        theor_overall_request_time = get_overall_request_time(theor_relative_bandwidth, mu, t)

        doc.writelines([
            pd.DataFrame({
                'Теор. среднее время пребывания заявки в СМО': [theor_overall_request_time]
            }, index=['Значение']).T.to_markdown(), ''
        ])

        plt.clf()


def main():
    lambda_ = [10, 15]
    mu = [3, 3.5]
    p = 1
    n = [2, 4]
    m = [10, 6]
    t = 1. / 6.
    for i in range(0, len(lambda_)):
        system = System(n[i], m[i], lambda_[i], mu[i], p, 0.01, 10000, t)
        system.log()
        system.run()
        generate_report(lambda_[i], mu[i], p, n[i], m[i], t, system.stats)


if __name__ == '__main__':
    main()