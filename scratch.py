import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt

from util import write_to_txt, plot_ekg, load_data, load_txt_file
from evaluate_EKG import evaluate_EKG

rmse = evaluate_EKG(0)
print(rmse)
exit()

f = open('data/calculated12Lead_csg.p', 'rb')
calculated12Lead = pickle.load(f)
f.close()

f = open('data/answerEKG.p', 'rb')
answerEKG = pickle.load(f)
f.close()

print(calculated12Lead.shape)
print(answerEKG.shape)

max_amplitude_calculated = np.max(calculated12Lead)
max_amplitude_answer = np.max(answerEKG)
print('max_amplitude_calculated: ' + str(np.max(calculated12Lead)))
print('max_amplitude_answer: ' + str(np.max(answerEKG)))

min_amplitude_calculated = np.min(calculated12Lead)
min_amplitude_answer = np.min(answerEKG)
print('min_amplitude_calculated: ' + str(np.min(calculated12Lead)))
print('min_amplitude_answer: ' + str(np.min(answerEKG)))

max_time_calculated = calculated12Lead.shape[0]
max_time_actual = answerEKG.shape[0]
print('max_time_calculated: ' + str(calculated12Lead.shape[0]))
print('max_time_actual: ' + str(answerEKG.shape[0]))

# feature scaling - between [0,1]
def normalize(x):
    xmin = np.min(x)
    xmax = np.max(x)
    x = (x-xmin)/(xmax-xmin)
    return x

calculated12Lead_normalized = normalize(calculated12Lead)
answerEKG_normalized = normalize(answerEKG)

factor = round(len(calculated12Lead)/len(answerEKG_normalized))

# Interpolate the shorter time series
answerEKG_interp = np.zeros(calculated12Lead_normalized.shape)
for i in range(answerEKG_normalized.shape[1]): #number of leads
    y = answerEKG_normalized[:,i]
    x = np.arange(len(answerEKG_normalized))
    x = x*factor
    xvals = np.arange(len(calculated12Lead_normalized))
    answerEKG_interp[:,i] = np.interp(xvals, x, y)

rmse = mean_squared_error(answerEKG_interp, calculated12Lead_normalized, squared=False)
print("Root mean square error:", rmse)


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

print(calculated12Lead_normalized.shape)
print(answerEKG_interp.shape)

calculated12Lead_smoothed = np.zeros(calculated12Lead_normalized.shape)
for i in range(calculated12Lead_normalized.shape[1]): #number of leads
    y = smooth(calculated12Lead_normalized[:,i])
    calculated12Lead_smoothed[:,i] = y[0:len(calculated12Lead_normalized)]

rmse = mean_squared_error(answerEKG_interp, calculated12Lead_smoothed, squared=False)
print("Root mean square error:", rmse)

plot_ekg(answerEKG_interp)

# write_to_txt(calculated12Lead_smoothed, 'data/calculated12Lead_smoothed.txt')
# write_to_txt(answerEKG_interp, 'data/answerEKG_interp.txt')
# write_to_txt(calculated12Lead_normalized, 'data/calculated12Lead_normalized.txt')



