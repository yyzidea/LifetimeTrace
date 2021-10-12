import matplotlib.pyplot as plt
import numpy as np

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

    while plt.fignum_exists(fig_num):
        lifetime, intensity = meas.getData()
        t = np.arange(0, lifetime.size)*meas.int_time*1e-12
        plt.figure(fig_num)

        __plot_lifetime_trace_sub(t, lifetime, intensity/meas.int_time/1e-12, plot_format_func)
        if not meas.isRunning():
            break


def __plot_normal_data(t, data, plot_format_func):
    plt.clf()

    plt.plot(t, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Count (pcs)')

    if plot_format_func is not None:
        plot_format_func(plt)

    plt.pause(0.1)


def plot_data(meas, plot_format_func=None):
    if 'LifetimeTrace' in meas.__class__.__name__:
        plot_lifetime_trace(meas, plot_format_func)
    else:
        plt.figure()
        fig_num = plt.gcf().number

        temp = meas.getIndex()
        binwidth = (temp[1]-temp[0])*1e-12

        while plt.fignum_exists(fig_num):
            data = np.transpose(meas.getData())
            t = np.arange(0, data.shape[0])*binwidth
            plt.figure(fig_num)
            if meas.__class__.__name__ is 'Counter':
                __plot_normal_data(t, data/binwidth, plot_format_func)
            elif meas.__class__.__name__ is 'Histogram':
                def f(plt):
                    plt.yscale('log')
                    if plot_format_func is not None:
                        plot_format_func(plt)
                __plot_normal_data(t, data, f)
            else:
                __plot_normal_data(t, data, plot_format_func)

            if not meas.isRunning():
                break


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
        if meas.__class__.__name__ is 'Counter':
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

