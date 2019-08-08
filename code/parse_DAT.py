import numpy as np
import os
import DRAO_utils as du
from coordDiffFromCSV import coordDiffFromCSV
import filterFuncs as ff
import utils


global DATfile


def dat_to_arrDict(filename):
    """
    Return dict from beamformer dat file. Each dict entry is an
    array of either width 1 (CW) or sweep-width with a buffer.

    Parameters
    ----------
    filename : string
    """
    global DATfile
    DATfile = filename

    dump = 8229
    # dataLength = 8192  # not needed, just interesting

    frequency = du.get_frequency()
    data_width, padding = du.get_dataWidth_padding()

    chan = 4096 - frequency  # channels start @ 0 @ 4096

    arrDict = {}

    antDict = du.get_antennaDict()

    with open(filename, 'rb') as f:
        current = 0

        while (current < os.path.getsize(filename)):
            str = f.read(dump)
            utils.printProgressBar(current+dump, os.path.getsize(filename), prefix='  Building arrDict:', length=50)

            topic, sts, data = str.split(b' ', 2)
            dummy, row, col = topic.split(b':')
            row = int(row)
            col = int(col)

            if row in antDict.keys() and col in antDict.keys():

                data = np.frombuffer(data, dtype=np.complex)
                chanData = np.zeros((1,
                                    data_width+2*padding)).astype(np.complex)
                chanData[0, :] = data[chan-padding:chan+data_width+padding]

                antStr = 'ant{}{}'.format(antDict[row], antDict[col])

                try:
                    arrDict[antStr]
                except KeyError:
                    arrDict[antStr] = np.zeros(
                        (0, data_width+2*padding)).astype(np.complex)

                arrDict[antStr] = np.append(arrDict[antStr], chanData, 0)

            current = current + dump

    return arrDict


def dat_to_acmDict(filename, key='all'):
    """
    Return ACM in Dict format.

    Parameters
    ----------
    filename : string
    """
    global DATfile
    DATfile = filename

    dump = 8229
    # dataLength = 8192  # not needed, just interesting

    numChans = 512

    acmDict = {}
    antDict = du.get_antennaDict()

    with open(filename, 'rb') as f:
        current = 0

        while (current < os.path.getsize(filename)):
            str = f.read(dump)
            utils.printProgressBar(current+dump, os.path.getsize(filename), prefix='  Building acmDict:', length=50)

            topic, sts, data = str.split(b' ', 2)
            dummy, row, col = topic.split(b':')
            row = int(row)
            col = int(col)

            if row in antDict.keys() and col in antDict.keys():
                antStr = 'ant{}{}'.format(antDict[row], antDict[col])

                if key == 'all' or key == antStr:

                    data = np.frombuffer(data, dtype=np.complex)
                    chanData = np.zeros((1, numChans)).astype(np.complex)
                    chanData[0, :] = data

                    try:
                        acmDict[antStr]
                    except KeyError:
                        acmDict[antStr] = np.zeros(
                            (0, numChans)).astype(np.complex)

                    acmDict[antStr] = np.append(acmDict[antStr], chanData, 0)

                if row is not col and key == 'all':
                    antStr = 'ant{}{}'.format(antDict[col], antDict[row])

                    try:
                        acmDict[antStr]
                    except KeyError:
                        acmDict[antStr] = np.zeros(
                            (0, numChans)).astype(np.complex)

                    acmDict[antStr] = np.append(acmDict[antStr], np.conjugate(chanData), 0)

            current = current + dump

    return acmDict

    