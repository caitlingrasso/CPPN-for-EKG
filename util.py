import numpy as np
import pickle
import matplotlib.pyplot as plt


def plot_ekg(data, save=False, filename='test.png'):
    """Plot EKG"""
    fig, axs = plt.subplots(3,4, figsize=[15,10])

    axs[0,0].plot(data[:,0])
    axs[0,0].set_title('I')

    axs[1, 0].plot(data[:, 1])
    axs[1, 0].set_title('II')

    axs[2, 0].plot(data[:, 2])
    axs[2, 0].set_title('III')

    axs[0, 1].plot(data[:, 3])
    axs[0, 1].set_title('aVR')

    axs[1, 1].plot(data[:, 4])
    axs[1, 1].set_title('aVL')

    axs[2, 1].plot(data[:, 5])
    axs[2, 1].set_title('aVF')

    axs[0, 2].plot(data[:, 6])
    axs[0, 2].set_title('V1')

    axs[1, 2].plot(data[:, 7])
    axs[1, 2].set_title('V2')

    axs[2, 2].plot(data[:, 8])
    axs[2, 2].set_title('V3')

    axs[0, 3].plot(data[:, 9])
    axs[0, 3].set_title('V4')

    axs[1, 3].plot(data[:, 10])
    axs[1, 3].set_title('V5')

    axs[2, 3].plot(data[:, 11])
    axs[2, 3].set_title('V6')

    if not save:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)


def write_to_txt(calculated12Lead, filename):
    """Converting from pickle file to comma delimited text file"""
    f = open(filename, "w+")

    for i in range(calculated12Lead.shape[0]):
        line = ''
        for j in range(calculated12Lead.shape[1]-1):
            line += str(calculated12Lead[i,j]) + ','
        line += str(calculated12Lead[i,-1]) + '\n'
        f.write(line)

    f.close()


def load_txt_file(filename):
    """loads in a text file and save it as a numpy array in a pickle file"""
    f = open(filename, 'r')
    data_list = f.readlines()
    f.close()

    rows = len(data_list[0].split(','))

    data = np.zeros((len(data_list), rows), dtype=float)

    for i, line in enumerate(data_list):
        line_as_list = line.split(",")
        for j in range(len(line_as_list)):
            data[i, j] = line_as_list[j]

    save_fn = filename[0:len(filename)-4] + '.p'

    f = open(save_fn, 'wb')
    pickle.dump(data, f)
    f.close()

def pickle_file(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def load_data(filename):
    """loads data from a pickle file and returns it"""
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()

    return data

def normalize(x):
    """feature scaling - between [0,1]"""
    xmin = np.min(x)
    xmax = np.max(x)
    x = (x-xmin)/(xmax-xmin)
    return x

def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y




