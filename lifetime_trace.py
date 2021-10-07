import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from threading import Thread, Lock
from warnings import warn, filterwarnings

# import TimeTagger
from python.driver import TimeTagger
from time_tagger_utility import *


class LifetimeTrace:
    def __init__(self, click_channels, start_channels, binwidth, n_bins, int_time, tagger=None, serial='1910000OZQ', mode='mean', offset=88100, min_delay=0, max_delay=200e3):
        # Create a TimeTagger instance to control your hardware
        if tagger is None:
            self.tagger = TimeTagger.createTimeTagger(serial)
        else:
            self.tagger = tagger

        self.click_channels = click_channels
        self.start_channels = start_channels
        self.binwidth = binwidth
        self.n_bins = n_bins
        self.int_time = int_time
        self.__lifetime = np.zeros(0)
        self.__intensity = np.zeros(0)
        self.hist = TimeTagger.Histogram(self.tagger, self.click_channels, self.start_channels,
                                         self.binwidth, self.n_bins)
        self.hist.stop()
        self.__sub_thread: Thread
        self.__signal = 'none'
        self.__data_end_idx = 0
        self.__new_data_idx = 0
        self.__lock = Lock()
        self.mode = mode
        self.offset = offset
        self.max_delay = max_delay
        self.min_delay = min_delay

    def set(self, click_channels=None, start_channels=None, binwidth=None, n_bins=None, int_time=None, mode=None):
        if click_channels is not None:
            self.click_channels = click_channels
        if start_channels is not None:
            self.start_channels = start_channels
        if binwidth is not None:
            self.binwidth = binwidth
        if n_bins is not None:
            self.n_bins = n_bins
        if int_time is not None:
            self.int_time = int_time
        if mode is not None:
            self.mode = mode

        self.hist = TimeTagger.Histogram(self.tagger, self.click_channels, self.start_channels,
                                         self.binwidth, self.n_bins)
        self.hist.stop()

    def getDataPerBin(self):
        self.hist.startFor(int(self.int_time * 1e12))
        self.hist.waitUntilFinished()

        res = np.array(self.hist.getData())

        if self.mode == 'fit':
            lifetime = self.__fit_lifetime(np.arange(0, res.size) * self.binwidth * 1e-12, res)
        else:
            delay = np.array(self.hist.getIndex())
            delay = delay-self.offset
            res[delay < self.min_delay] = 0
            res[delay > self.max_delay] = 0
            intensity = np.sum(res)
            lifetime = np.sum(delay*res)/intensity*1e-3

        return lifetime, intensity

    def startFor(self, duration, clear=True, timeout=-1):
        if clear:
            self.clear(timeout)

        if self.isRunning() == 0:
            self.__sub_thread = Thread(target=self.__measurement_routine, args=(duration,))
            self.__sub_thread.daemon = True
            self.__signal = 'idle'
            self.__sub_thread.start()
        else:
            warn('The measurement is running.')

        return 0

    def clear(self, timeout=-1):
        if self.stop(timeout) == 1:
            self.__lifetime = np.zeros(200)
            self.__intensity = np.zeros(200)
            self.__data_end_idx = 0
            return 1
        else:
            return 0

    def start(self, clear=True):
        return self.startFor(-1, clear)

    def stop(self, timeout=-1):
        if self.__signal != 'none':
            self.__signal = 'stop'
            res = self.waitUntilFinished(timeout)
        else:
            res = True
        return res

    def waitUntilFinished(self, timeout):
        if self.__signal != 'none':
            if timeout >= 0:
                time.sleep(timeout)
                return not self.__sub_thread.is_alive()
            else:
                while self.__sub_thread.is_alive():
                    time.sleep(0.1)

                return True
        else:
            return True

    def isRunning(self):
        if self.__signal != 'none':
            return self.__sub_thread.is_alive()
        else:
            return False

    def getNewData(self):
        with self.__lock:
            res = (self.__lifetime.copy(), self.__intensity.copy())
            res = res[0][self.__new_data_idx:self.__data_end_idx], res[1][self.__new_data_idx:self.__data_end_idx]
            self.__new_data_idx = self.__data_end_idx

        if 'res' in locals():
            return res[0], res[1]
        else:
            return None, None

    def getData(self):
        with self.__lock:
            res = (self.__lifetime.copy(), self.__intensity.copy())
            self.__new_data_idx = self.__data_end_idx

        if 'res' in locals():
            return res[0][:self.__data_end_idx], res[1][:self.__data_end_idx]
        else:
            return None, None

    @staticmethod
    def __fit_lifetime(delay, count):
        def exp1(t, a1, t1):
            return a1 * np.exp(-1 / t1 * 1e9 * t)

        def mov_mean(x, w):
            return np.convolve(x, np.ones(w) / w, mode='valid')

        start_idx = np.argmax(count)

        if start_idx - 60 > 0:
            end_idx = np.nonzero(mov_mean(count[start_idx:], 9) > np.mean(count[:start_idx - 50])) + start_idx
        else:
            end_idx = np.nonzero(mov_mean(count[start_idx:], 9) > np.mean(count[-50:-1])) + start_idx
        end_idx = end_idx[0, -1]

        filterwarnings('ignore')
        wt = 1 / count[start_idx:end_idx]
        filterwarnings('default')

        p0 = [count[start_idx],
              (delay[np.nonzero(count > count[start_idx] * np.exp(-1))[0][-1]]-delay[start_idx]) * 1e9]

        popt, pcov = curve_fit(exp1, delay[start_idx:end_idx]-delay[start_idx], count[start_idx:end_idx], p0=p0,
                               sigma=wt, absolute_sigma=True)

        return popt[1]

    def __measurement_routine(self, duration):
        print('Sub thread starts!')
        if duration > 0:
            loop_num = int(duration // self.int_time)
        else:
            loop_num = -1

        while loop_num:
            if self.__signal == 'stop':
                break
            else:
                pass

            lifetime, intensity = self.getDataPerBin()
            with self.__lock:
                self.__lifetime[self.__data_end_idx] = lifetime
                self.__intensity[self.__data_end_idx] = intensity
                self.__data_end_idx += 1
                if self.__data_end_idx == self.__lifetime.size:
                    self.__intensity = np.hstack((self.__intensity, np.zeros(200)))
                    self.__lifetime = np.hstack((self.__lifetime, np.zeros(200)))

            loop_num -= 1

        self.__signal = 'none'
        print('Sub thread stops!')
        return 0

    # def __del__(self):
    #     TimeTagger.freeTimeTagger(self.tagger)
    #     if self.isRunning():
    #         if self.stop(1) is False:
    #             self.__sub_thread.terminate()
    #             print('Subprocess (pid:%s) stops!' % self.__pid)


if __name__ == '__main__':
    tagger = TimeTagger.createTimeTagger('1910000OZQ')
    tagger.setTestSignal(1, True)
    tagger.setTestSignal(2, True)

    meas = LifetimeTrace(2, 1, 10, 100000, int_time=0.100, tagger=tagger)
    meas.startFor(5)
    plot_lifetime_trace(meas)

    tagger.setTestSignal(1, False)
    tagger.setTestSignal(2, False)
