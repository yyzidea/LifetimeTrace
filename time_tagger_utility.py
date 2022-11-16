import matplotlib.pyplot as plt
import numpy as np
from device.wave_generator import waveform_generate

# if 'TimeTagger' not in locals():
#     from python.driver import TimeTagger


def __plot_lifetime_trace_sub(t, lifetime, intensity, plot_format_func):
    plt.clf()

    plt.subplot(2, 5, (1, 4))
    plt.plot(t, lifetime)
    plt.ylabel('Lifetime (ns)')
    ylim1 = plt.gca().get_ylim()

    plt.subplot(2, 5, 5)
    plt.hist(lifetime, 100, orientation='horizontal')
    plt.gca().set_xticks([])
    plt.gca().set_ylim(ylim1)
    plt.gca().set_yticks([])

    plt.subplot(2, 5, (6, 9))
    plt.plot(t, intensity)
    plt.xlabel('Time (s)')
    plt.ylabel('Count (pcs)')
    ylim2 = plt.gca().get_ylim()

    plt.subplot(2, 5, 10)
    plt.hist(intensity, 100, orientation='horizontal')
    plt.gca().set_xticks([])
    plt.gca().set_ylim(ylim2)
    plt.gca().set_yticks([])

    if plot_format_func is not None:
        plot_format_func(plt)

    plt.pause(0.1)


def plot_lifetime_trace(meas, plot_format_func=None):
    plt.figure()
    fig_num = plt.gcf().number

    flag = 0
    while plt.fignum_exists(fig_num):
        lifetime, intensity = meas.getData()
        t = np.arange(0, lifetime.size)*meas.int_time*1e-12
        plt.figure(fig_num)

        __plot_lifetime_trace_sub(t, lifetime, intensity/meas.int_time/1e-12, plot_format_func)
        if flag:
            break

        if not meas.isRunning():
            flag = 1


def __plot_normal_data(t, data, plot_format_func):
    plt.clf()

    plt.plot(t, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Count (cps)')

    if plot_format_func is not None:
        plot_format_func(plt)

    plt.pause(0.1)


def plot_data(meas, plot_length=None, plot_format_func=None):
    if 'LifetimeTrace' in meas.__class__.__name__:
        plot_lifetime_trace(meas, plot_format_func)
    else:
        plt.figure()
        fig_num = plt.gcf().number

        temp = meas.getIndex()
        binwidth = (temp[1]-temp[0])*1e-12

        flag = 0
        while plt.fignum_exists(fig_num):
            data = np.transpose(meas.getData())
            t = np.arange(0, data.shape[0])*binwidth

            if plot_length is not None and data.size > plot_length:
                data = data[-1-plot_length:]
                t = t[-1-plot_length:]

            plt.figure(fig_num)
            if meas.__class__.__name__ == 'Counter':
                __plot_normal_data(t, data/binwidth, plot_format_func)
            elif meas.__class__.__name__ == 'Histogram' or meas.__class__.__name__ == 'GatedHistogramFromFile':
                def f(plt):
                    plt.yscale('log')
                    plt.xlabel('Delay (ns)')
                    plt.xlim(0, 200)
                    if plot_format_func is not None:
                        plot_format_func(plt)
                offset = t[np.argmax(data)]

                __plot_normal_data((t-offset)*1e9, data, f)

            else:
                t = meas.getIndex()*1e-3
                __plot_normal_data(t, data, plot_format_func)

            if flag:
                break

            if not meas.isRunning():
                flag = 1


def save_data(fname, meas):
    if 'LifetimeTrace' in meas.__class__.__name__:
        lifetime, intensity = meas.getData()
        t = np.arange(0, lifetime.size)*meas.int_time*1e-12
        np.savetxt(fname, np.vstack((t, lifetime, intensity/meas.int_time/1e-12)).transpose(), delimiter=',')
    else:
        temp = meas.getIndex()
        binwidth = (temp[1]-temp[0])*1e-12
        data = np.transpose(meas.getData())
        t = np.arange(0, data.shape[0])*binwidth
        if meas.__class__.__name__ == 'Counter':
            np.savetxt(fname, np.vstack((t, data/binwidth)).transpose(), delimiter=',')
        else:
            np.savetxt(fname, np.vstack((t, data)).transpose(), delimiter=',')


def plot_file(fname, fig_num=None, meas_type='LifetimeTrace', plot_format_func=None):
    plt.figure(fig_num)
    data = np.loadtxt(fname, delimiter=',')

    if meas_type == 'LifetimeTrace':
        __plot_lifetime_trace_sub(data[:,0], data[:,1], data[:,2], plot_format_func)
    else:
        __plot_normal_data(data[:,0], data[:,1:], plot_format_func)


# def plot_lifetime_trace_with_voltage(meas, V_l, V_h, scan_rate, mode, plot_format_func=None):
#     def plot_func(t, lifetime, intensity, plot_format_func):
#         plt.clf()
#
#         amplitude = abs(V_h-V_l)
#         offset = (V_h+V_l)/2
#         frequency = scan_rate/amplitude/2
#         phase = 0
#
#         if mode == 'sweep':
#             if V_l <= 0 <= V_h:
#                 phase = V_l/amplitude*180
#         elif mode == 'binary':
#             if V_l < V_h:
#                 phase = 180
#             else:
#                 phase = 0
#
#         mode_waveform_map = {'sweep': 'triangle', 'binary': 'square'}
#
#         plt.subplot(3, 5, (1, 4))
#         plt.plot(t, waveform_generate(mode_waveform_map[mode], t, amplitude, frequency, offset, phase))
#         plt.ylabel('Lifetime (ns)')
#
#         plt.subplot(3, 5, (6, 9))
#         plt.plot(t, lifetime)
#         plt.ylabel('Lifetime (ns)')
#         ylim1 = plt.gca().get_ylim()
#
#         plt.subplot(3, 5, 10)
#         plt.hist(lifetime, 100, orientation='horizontal')
#         plt.gca().set_xticks([])
#         plt.gca().set_ylim(ylim1)
#         plt.gca().set_yticks([])
#
#         plt.subplot(3, 5, (11, 14))
#         plt.plot(t, intensity)
#         plt.xlabel('Time (s)')
#         plt.ylabel('Count (pcs)')
#         ylim2 = plt.gca().get_ylim()
#
#         plt.subplot(3, 5, 15)
#         plt.hist(intensity, 100, orientation='horizontal')
#         plt.gca().set_xticks([])
#         plt.gca().set_ylim(ylim2)
#         plt.gca().set_yticks([])
#
#         if plot_format_func is not None:
#             plot_format_func(plt)
#
#         plt.pause(0.1)
#
#     plt.figure()
#     fig_num = plt.gcf().number
#
#     while plt.fignum_exists(fig_num):
#         lifetime, intensity = meas.getData()
#         t = np.arange(0, lifetime.size)*meas.int_time*1e-12
#         plt.figure(fig_num)
#
#         plot_func(t, lifetime, intensity/meas.int_time/1e-12, plot_format_func)
#         if not meas.isRunning():
#             break


def plot_lifetime_trace_with_voltage(meas, V_l, V_h, scan_rate, mode, plot_format_func=None, delay=0):
    def plot_func(t, lifetime, intensity, plot_format_func):
        plt.clf()

        amplitude = abs(V_h-V_l)
        offset = (V_h+V_l)/2
        frequency = scan_rate/amplitude/2
        phase = 0

        if mode == 'sweep':
            if V_l <= 0 <= V_h:
                phase = V_l/amplitude*180
        elif mode == 'binary':
            if V_l < V_h:
                phase = 180
            else:
                phase = 0

        if t.size < 2:
            return
        else:
            mode_waveform_map = {'sweep': 'triangle', 'binary': 'square'}

            V = waveform_generate(mode_waveform_map[mode], t, amplitude, frequency, offset, phase)
            int_time = t[1]-t[0]
            point_delay = int(np.round(delay/int_time))
            V = np.roll(V, point_delay)[point_delay:]

            lifetime = lifetime[point_delay:]
            intensity = intensity[point_delay:]
            t = t[point_delay:]

        if mode == 'binary':
            plt.subplot(3, 5, (1, 4))
            plt.plot(t, V)
            plt.ylabel('Voltage (V)')

            plt.subplot(3, 5, (6, 9))
            plt.plot(t, lifetime)
            plt.ylabel('Lifetime (ns)')
            ylim1 = plt.gca().get_ylim()

            plt.subplot(3, 5, 10)
            plt.hist(lifetime, 100, orientation='horizontal')
            plt.gca().set_xticks([])
            plt.gca().set_ylim(ylim1)
            plt.gca().set_yticks([])

            plt.subplot(3, 5, (11, 14))
            plt.plot(t, intensity)
            plt.xlabel('Time (s)')
            plt.ylabel('Count (pcs)')
            ylim2 = plt.gca().get_ylim()

            plt.subplot(3, 5, 15)
            plt.hist(intensity, 100, orientation='horizontal')
            plt.gca().set_xticks([])
            plt.gca().set_ylim(ylim2)
            plt.gca().set_yticks([])

        elif mode == 'sweep':
            if not hasattr(plot_func, 'old_data_length'):
                plot_func.old_data_length = 0
                plot_func.point_length_per_cycle = int(np.round(1/frequency/(int_time)))
                plot_func.sum_lifetime = np.zeros(plot_func.point_length_per_cycle)
                plot_func.lifetime_divider = np.zeros(plot_func.point_length_per_cycle)

            while plot_func.old_data_length < t.size:
                plot_func.sum_lifetime[plot_func.old_data_length % plot_func.point_length_per_cycle] +=\
                    lifetime[plot_func.old_data_length]
                plot_func.lifetime_divider[plot_func.old_data_length % plot_func.point_length_per_cycle] += 1
                plot_func.old_data_length += 1

            plt.subplot(1, 7, (1, 2))
            plt.plot(V, lifetime)

            if plot_func.point_length_per_cycle > V.size:
                V_unique = np.roll(V, -(plot_func.old_data_length % plot_func.point_length_per_cycle))
                mean_lifetime = np.roll(plot_func.sum_lifetime[:V_unique.size]/plot_func.lifetime_divider[:V_unique.size],
                                        -(plot_func.old_data_length % plot_func.point_length_per_cycle))
                plt.plot(V_unique, mean_lifetime, 'k')
            else:
                mean_lifetime = np.roll(plot_func.sum_lifetime/plot_func.lifetime_divider,
                                        -(plot_func.old_data_length % plot_func.point_length_per_cycle))
                V_unique =V[:plot_func.point_length_per_cycle]
                V_unique = np.roll(V_unique, -(plot_func.old_data_length % plot_func.point_length_per_cycle))
                plt.plot(V_unique, mean_lifetime, 'k')


            plt.ylabel('Lifetime (ns)')
            plt.xlabel('Voltage (V)')

            plt.subplot(2, 7, (3, 6))
            plt.plot(t, lifetime)
            plt.ylabel('Lifetime (ns)')
            ylim1 = plt.gca().get_ylim()

            plt.subplot(2, 7, 7)
            plt.hist(lifetime, 100, orientation='horizontal')
            plt.gca().set_xticks([])
            plt.gca().set_ylim(ylim1)
            plt.gca().set_yticks([])

            plt.subplot(2, 7, (10, 13))
            plt.plot(t, intensity)
            plt.xlabel('Time (s)')
            plt.ylabel('Count (pcs)')
            ylim2 = plt.gca().get_ylim()

            plt.subplot(2, 7, 14)
            plt.hist(intensity, 100, orientation='horizontal')
            plt.gca().set_xticks([])
            plt.gca().set_ylim(ylim2)
            plt.gca().set_yticks([])

            plt.tight_layout()
            plt.gcf().set_figwidth(10)

        if plot_format_func is not None:
            plot_format_func(plt)

        plt.pause(0.1)

    plt.figure()
    fig_num = plt.gcf().number

    flag = 0
    while plt.fignum_exists(fig_num):
        lifetime, intensity = meas.getData()
        t = np.arange(0, lifetime.size)*meas.int_time*1e-12
        plt.figure(fig_num)

        plot_func(t, lifetime, intensity/meas.int_time/1e-12, plot_format_func)
        if flag:
            break

        if not meas.isRunning():
            flag = 1

