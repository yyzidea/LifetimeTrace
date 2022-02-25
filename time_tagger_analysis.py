import matplotlib.pyplot as plt
import numpy as np
import numba
from python.driver import TimeTagger
from time_tagger_utility import *


class GatedHistogram(TimeTagger.CustomMeasurement):
    def __init__(self, tagger, click_channel, start_channel, binwidth, n_bins, gate_binwidth, gate_bins):
        TimeTagger.CustomMeasurement.__init__(self, tagger)
        self.click_channel = click_channel
        self.start_channel = start_channel
        self.binwidth = binwidth
        self.n_bins = n_bins
        self.gate_binwidth = gate_binwidth
        self.gate_bins = gate_bins

        self.__hist_data = np.zeros((self.n_bins,), dtype=np.uint64)
        self.__last_start_timestamp = 0

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
        self.startFor(int(capture_duration * 1e12), clear)

    def getData(self):
        # Acquire a lock this instance to guarantee that process() is not running in parallel
        # This ensures to return a consistent data.
        self._lock()
        data = self.__hist_data.copy()
        # We have gathered the data, unlock, so measuring can continue.
        self._unlock()
        return data

    def getIndex(self):
        # This method does not depend on the internal state, so there is no
        # need for a lock.
        arr = np.arange(0, self.n_bins) * self.binwidth
        return arr

    def clear_impl(self):
        # The lock is already acquired within the backend.
        self.__last_start_timestamp = 0
        self.__hist_data = np.zeros((self.n_bins,), dtype=np.uint64)

    def on_start(self):
        # The lock is already acquired within the backend.
        pass

    def on_stop(self):
        # The lock is already acquired within the backend.
        pass

    @staticmethod
    @numba.jit(nopython=True, nogil=True)
    def sub_process(tags,
                    data,
                    click_channel,
                    start_channel,
                    binwidth,
                    last_start_timestamp,
                    gate_binwidth,
                    gate_bins):

        for tag in tags:
            # tag.type can be: 0 - TimeTag, 1- Error, 2 - OverflowBegin, 3 -
            # OverflowEnd, 4 - MissedEvents
            if tag['type'] != 0:
                # tag is not a TimeTag, so we are in an error state, e.g. overflow
                last_start_timestamp = 0
            elif tag['channel'] == click_channel and last_start_timestamp != 0:
                bin_idx = tag['time'] // gate_binwidth
                if bin_idx < len(gate_bins) and gate_bins[bin_idx]:
                    # valid event
                    index = (tag['time'] - last_start_timestamp) // binwidth
                    if index < data.shape[0]:
                        data[index] += 1
            elif tag['channel'] == start_channel:
                last_start_timestamp = tag['time']

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

        self.last_start_timestamp = GatedHistogram.sub_process(
            incoming_tags,
            self.__hist_data,
            self.click_channel,
            self.start_channel,
            self.binwidth,
            self.__last_start_timestamp,
            self.gate_binwidth,
            self.gate_bins)


class GatedHistogramFromFile(GatedHistogram):
    def __init__(self, file, click_channel, start_channel, binwidth, n_bins, gate_binwidth, gate_bins):
        self.virtual_tagger = TimeTagger.createTimeTaggerVirtual()
        self.file = file
        GatedHistogram.__init__(self, self.virtual_tagger, click_channel, start_channel, binwidth, n_bins,
                                gate_binwidth, gate_bins)

    def replay(self, begin=0, duration=-1, queue=False):
        self.clear()
        self.virtual_tagger.replay(file=self.file, begin=begin, duration=duration, queue=queue)

    def replayAndWait(self, begin=0, duration=-1, queue=False, timeout=-1):
        ID = self.virtual_tagger.replay(file=self.file, begin=begin, duration=duration, queue=queue)
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

    def isRunning(self):
        return not self.waitForCompletion(timeout=0)

    def __del__(self):
        self.virtual_tagger.__del__()


if __name__ == '__main__':
    sample = 'lll_device_L_1'
    data_dir = r'E:\yyz\Quantum Dot Device\Lifetime trace\202202'

    file = data_dir + '\\' + 'lll_device_L_1_10_1_lifetimetrace_cw_450nm_300nw_100ms'

    click_channel = 2
    start_channel = 3

    binwidth = 50
    n_bins = int(1e6 / binwidth)
    int_time = int(0.1e12)

    def gate_func(a):
        return 1

    # virtual_tagger = TimeTagger.createTimeTaggerVirtual()
    # hist = TimeTagger.Histogram(virtual_tagger, click_channel, start_channel, binwidth, n_bins)
    # virtual_tagger.replay(file)
    #
    # virtual_tagger.waitForCompletion()
    # plot_data(hist)
    #
    # duration = hist.getCaptureDuration()

    gate_bins = np.zeros(int(np.ceil(500000000000000/int_time)))
    meas = GatedHistogramFromFile(file,
                                  click_channel=2,
                                  start_channel=3,
                                  binwidth=binwidth,
                                  n_bins=n_bins,
                                  gate_binwidth=int_time,
                                  gate_bins=gate_bins)

    meas.replay()
    plot_data(meas)

