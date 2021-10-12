import time
import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy.optimize import curve_fit
from threading import Thread, Lock
from warnings import warn, filterwarnings

# import TimeTagger
from python.driver import TimeTagger
from time_tagger_utility import *


class LifetimeTrace(TimeTagger.CustomMeasurement):
    """
    Example for a single start - multiple stop measurement.
        The class shows how to access the raw time-tag stream.
    """

    def __init__(self, tagger, click_channel, start_channel, binwidth, n_bins, int_time,
                 offset=88100, min_delay=0, max_delay=200e3):
        TimeTagger.CustomMeasurement.__init__(self, tagger)
        self.click_channel = click_channel
        self.start_channel = start_channel
        self.binwidth = binwidth
        self.n_bins = n_bins
        self.int_time = int_time

        self.__lifetime = np.zeros(200)
        self.__intensity = np.zeros(200)
        self.__data_end_idx = 0
        self.__last_start_timestamp = 0
        self.__bin_start_timestamp = 0
        self.__hist_data = np.zeros((self.n_bins,), dtype=np.uint64)

        self.offset = offset
        self.max_delay = max_delay
        self.min_delay = min_delay

        self.inspect_var = 0

        # The method register_channel(channel) activates
        # that data from the respective channels is transferred
        # from the Time Tagger to the PC.
        self.register_channel(channel=click_channel)
        self.register_channel(channel=start_channel)

        self.clear_impl()

        # At the end of a CustomMeasurement construction,
        # we must indicate that we have finished.
        self.finalize_init()

    def __del__(self):
        # The measurement must be stopped before deconstruction to avoid
        # concurrent process() calls.
        self.stop()

    def startForSecond(self, capture_duration, clear=True):
        self.startFor(int(capture_duration*1e12), clear)

    def getData(self):
        # Acquire a lock this instance to guarantee that process() is not running in parallel
        # This ensures to return a consistent data.
        self._lock()
        lifetime = self.__lifetime[:self.__data_end_idx].copy()*1e-3
        intensity = self.__intensity[:self.__data_end_idx].copy()
        # We have gathered the data, unlock, so measuring can continue.
        self._unlock()
        return lifetime, intensity

    def getIndex(self):
        # This method does not depend on the internal state, so there is no
        # need for a lock.
        arr = np.arange(0, self.__data_end_idx)*self.int_time
        return arr

    def clear_impl(self):
        # The lock is already acquired within the backend.
        self.__lifetime = np.zeros(200)
        self.__intensity = np.zeros(200)
        self.__data_end_idx = 0
        self.__last_start_timestamp = 0
        self.__bin_start_timestamp = 0
        self.__hist_data = np.zeros((self.n_bins,), dtype=np.uint64)

    def on_start(self):
        # The lock is already acquired within the backend.
        pass

    def on_stop(self):
        # The lock is already acquired within the backend.
        pass

    # def set(self, **kwargs):
    #     for key, value in kwargs.items():
    #         self.__setattr__(key, value)

    @staticmethod
    @numba.jit(nopython=True, nogil=True)
    def fast_histogram_process(
            tags,
            hist_data,
            click_channel,
            start_channel,
            binwidth,
            last_start_timestamp,
            bin_start_timestamp,
            int_time,
            lifetime,
            intensity,
            data_end_idx,
            offset,
            min_delay,
            max_delay,
            n_bins):
        """
        A precompiled version of the histogram algorithm for better performance
        nopython=True: Only a subset of the python syntax is supported.
                       Avoid everything but primitives and numpy arrays.
                       All slow operation will yield an exception
        nogil=True:    This method will release the global interpreter lock. So
                       this method can run in parallel with other python code
        """

        for tag in tags:
            # tag.type can be: 0 - TimeTag, 1- Error, 2 - OverflowBegin, 3 -
            # OverflowEnd, 4 - MissedEvents
            if tag['type'] != 0:
                # tag is not a TimeTag, so we are in an error state, e.g. overflow
                last_start_timestamp = 0

            elif tag['channel'] == start_channel:
                last_start_timestamp = tag['time']

                if tag['time'] - bin_start_timestamp > int_time:
                    delay = np.arange(0, n_bins)*binwidth-offset
                    hist_data[delay < min_delay] = 0
                    hist_data[delay > max_delay] = 0

                    intensity[data_end_idx] = np.sum(hist_data)
                    if intensity[data_end_idx] == 0:
                        lifetime[data_end_idx] = -1
                    else:
                        lifetime[data_end_idx] = np.sum(delay*hist_data)/intensity[data_end_idx]

                    hist_data[:] = 0
                    bin_start_timestamp = tag['time']

                    # Expand the data array length
                    data_end_idx += 1
                    if data_end_idx == lifetime.size:
                        intensity = np.hstack((intensity, np.zeros(200)))
                        lifetime = np.hstack((lifetime, np.zeros(200)))

            elif tag['channel'] == click_channel and last_start_timestamp != 0:
                if tag['time'] - bin_start_timestamp > int_time:
                    delay = np.arange(0, n_bins)*binwidth-offset
                    hist_data[delay < min_delay] = 0
                    hist_data[delay > max_delay] = 0

                    intensity[data_end_idx] = np.sum(hist_data)
                    if intensity[data_end_idx] == 0:
                        lifetime[data_end_idx] = -1
                    else:
                        lifetime[data_end_idx] = np.sum(delay*hist_data)/intensity[data_end_idx]

                    hist_data[:] = 0
                    bin_start_timestamp = tag['time']

                    # Expand the data array length
                    data_end_idx += 1
                    if data_end_idx == lifetime.size:
                        intensity = np.hstack((intensity, np.zeros(200)))
                        lifetime = np.hstack((lifetime, np.zeros(200)))

                # valid event
                index = (tag['time'] - last_start_timestamp) // binwidth
                if index < hist_data.shape[0]:
                    hist_data[index] += 1

        return last_start_timestamp, bin_start_timestamp, data_end_idx, intensity, lifetime

    def process(self, incoming_tags, begin_time, end_time):
        """
        Main processing method for the incoming raw time-tags.

        The lock is already acquired within the backend.
        self.data is provided as reference, so it must not be accessed
        anywhere else without locking the mutex.

        Parameters
        ----------
        incoming_tags
            The incoming raw time tag stream provided as a read-only reference.
            The storage will be deallocated after this call, so you must not store a reference to
            this object. Make a copy instead.
            Please note that the time tag stream of all channels is passed to the process method,
            not only the onces from register_channel(...).
        begin_time
            Begin timestamp of the of the current data block.
        end_time
            End timestamp of the of the current data block.
        """
        self.inspect_var = self.int_time
        self.__last_start_timestamp, self.__bin_start_timestamp, self.__data_end_idx, self.__intensity, self.__lifetime =\
            LifetimeTrace.fast_histogram_process(
                incoming_tags,
                self.__hist_data,
                self.click_channel,
                self.start_channel,
                self.binwidth,
                self.__last_start_timestamp,
                self.__bin_start_timestamp,
                self.int_time,
                self.__lifetime,
                self.__intensity,
                self.__data_end_idx,
                self.offset,
                self.min_delay,
                self.max_delay,
                self.n_bins)


class LifetimeTraceWithFileWriter(TimeTagger.SynchronizedMeasurements):
    def __init__(self, tagger, click_channel, start_channel, binwidth, n_bins, int_time, filename,
                 offset=88100, min_delay=0, max_delay=200e3):
        TimeTagger.SynchronizedMeasurements.__init__(self, tagger)

        self.click_channel = click_channel
        self.start_channel = start_channel
        self.binwidth = binwidth
        self.n_bins = n_bins
        self.int_time = int_time
        self.offset = offset
        self.max_delay = max_delay
        self.min_delay = min_delay
        self.filename = filename

        self.lifetime_trace = LifetimeTrace(self.getTagger(), self.click_channel, self.start_channel, self.binwidth,
                                            self.n_bins, self.int_time, self.offset, self.min_delay, self.max_delay)
        self.file_writer = TimeTagger.FileWriter(self.getTagger(), self.filename,
                                                 [self.click_channel, self.start_channel])

    def getData(self):
        return self.lifetime_trace.getData()

    def getIndex(self):
        return self.lifetime_trace.getIndex()

    def startForSecond(self, capture_duration, clear=True):
        self.startFor(int(capture_duration*1e12), clear)

    def split(self, *args, **kwargs):
        self.file_writer.split(*args, **kwargs)

    def setMaxFileSize(self, max_file_size):
        self.file_writer.setMaxFileSize(max_file_size)

    def getMaxFileSize(self):
        return self.file_writer.getMaxFileSize()

    def getTotalEvents(self):
        return self.file_writer.getTotalEvents()

    def getTotalSize(self):
        return self.file_writer.getTotalSize()


class LifetimeTraceFromFile(LifetimeTrace):
    def __init__(self, click_channel, start_channel, binwidth, n_bins, int_time,
                 offset=88100, min_delay=0, max_delay=200e3):

        self.virtual_tagger = TimeTagger.createTimeTaggerVirtual()
        LifetimeTrace.__init__(self, self.virtual_tagger, click_channel, start_channel, binwidth,
                               n_bins, int_time, offset=offset, min_delay=min_delay, max_delay=max_delay)

    def replay(self, file, begin=0, duration=-1, queue=False):
        self.virtual_tagger.replay(file=file, begin=begin, duration=duration, queue=queue)

    def replayAndWait(self, file, begin=0, duration=-1, queue=False, timeout=-1):
        ID = self.virtual_tagger.replay(file=file, begin=begin, duration=duration, queue=queue)
        if self.waitForCompletion(ID=ID, timeout=timeout):
            self.stop()

    def stopReplay(self):
        self.virtual_tagger.stop()

    def waitForCompletion(self, ID=0, timeout=-1):
        return self.virtual_tagger.waitForCompletion(ID=ID, timeout=timeout)

    def setReplaySpeed(self, speed):
        self.virtual_tagger.setReplaySpeed(speed)

    def getReplaySpeed(self):
        return self.virtual_tagger.getReplaySpeed()

    def getConfiguration(self):
        return self.virtual_tagger.getConfiguration()

    # def __setattr__(self, key, value):
    #     if key == 'virtual_tagger':
    #         self.__dict__[key] = value
    #     else:
    #         self.__dict__[key] = value
    #         param = {key: value}
    #         self.set(**param)


class LifetimeTraceLegacy:
    def __init__(self, click_channel, start_channel, binwidth, n_bins, int_time, tagger=None, serial='1910000OZQ',
                 mode='mean', offset=88100, min_delay=0, max_delay=200e3):
        # Create a TimeTagger instance to control your hardware
        if tagger is None:
            self.tagger = TimeTagger.createTimeTagger(serial)
        else:
            self.tagger = tagger

        self.click_channel = click_channel
        self.start_channel = start_channel
        self.binwidth = binwidth
        self.n_bins = n_bins
        self.int_time = int_time
        self.__lifetime = np.zeros(0)
        self.__intensity = np.zeros(0)
        self.hist = TimeTagger.Histogram(self.tagger, self.click_channel, self.start_channel,
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

    def set(self, click_channel=None, start_channel=None, binwidth=None, n_bins=None, int_time=None, mode=None):
        if click_channel is not None:
            self.click_channel = click_channel
        if start_channel is not None:
            self.start_channel = start_channel
        if binwidth is not None:
            self.binwidth = binwidth
        if n_bins is not None:
            self.n_bins = n_bins
        if int_time is not None:
            self.int_time = int_time
        if mode is not None:
            self.mode = mode

        self.hist = TimeTagger.Histogram(self.tagger, self.click_channel, self.start_channel,
                                         self.binwidth, self.n_bins)
        self.hist.stop()

    def getDataPerBin(self):
        self.hist.startFor(int(self.int_time))
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

    # meas = LifetimeTraceLegacy(2, 1, 10, 100000, int_time=0.100, tagger=tagger)
    # meas = LifetimeTrace(tagger, 2, 1, 100, 10000, int(0.1*1e12), offset=0, max_delay=int(200e3))
    filename = r'E:\yyz\Quantum Dot Device\Lifetime trace\20210915\test.ttbin'
    meas = LifetimeTraceWithFileWriter(tagger, 2, 1, 100, 10000, int(0.1*1e12), filename, offset=0, max_delay=int(200e3))

    meas.startFor(int(5e12))
    plot_lifetime_trace(meas)

    tagger.setTestSignal(1, False)
    tagger.setTestSignal(2, False)
