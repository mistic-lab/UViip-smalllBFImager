import tkinter as tk
from tkinter.filedialog import askopenfilename
import numpy as np


def filename_from_GUI():
    """
    Get filename from tk GUI window.
    """

    filedialog = tk.Tk()
    filedialog.withdraw()
    filedialog.update()
    filename = askopenfilename()
    filedialog.update()
    return filename


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
                     length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent
                                  complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def running_mean(x, N):
    """
    Returns array. Calculates rolling average of length N on array x.
    """

    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)
