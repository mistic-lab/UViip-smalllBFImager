import numpy as np
from coordDiffFromCSV import coordDiffFromCSV
import DRAO_utils as du
import filterFuncs as ff
import utils
import os


def separate_cross_auto_dicts(arrDict):
    """
    Return crossDict and autoDict in that order. Input is main arrDict.

    Parameters
    ----------
    arrDict : dictionary
    """

    keys = arrDict.keys()
    crossDict = {}
    autoDict = {}
    for i in keys:
        row = i[-2]
        col = i[-1]
        if row != col:
            crossDict[i] = arrDict[i]
        else:
            autoDict[i] = arrDict[i]

    return crossDict, autoDict


def addDistanceToDict(crossDict):
    """
    Adds distances to input dict. If key is 'ant12', then the distance is
    stored in key 'dant12'. Also adds distance from each antenna to TX. Keys
    are 'dTXx' where x is the antenna number.

    Parameters
    ----------
    crossDict : dictionary
            No point in autodict for distances.
    """
    distDict = {}
    tx_done = []
    for key in crossDict:
        dKey = 'd{}'.format(key)
        distDict[dKey] = coordDiffFromCSV(du.get_config(), key[-2], key[-1])[0]
        if key[-2] not in tx_done:
            dTXkey = 'dTX{}'.format(key[-2])
            distDict[dTXkey] = coordDiffFromCSV(du.get_config(), 0, key[-2])[0]
            tx_done.append(key[-2])
        if key[-1] not in tx_done:
            dTXkey = 'dTX{}'.format(key[-1])
            distDict[dTXkey] = coordDiffFromCSV(du.get_config(), 0, key[-1])[0]
            tx_done.append(key[-1])

    crossDict.update(distDict)
    return crossDict


def removeDistanceFromDict(crossDict):
    """
    Removes distances from input dict. If key starts with a d then it's
    removed.

    Parameters
    ----------
    crossDict : dictionary
            No point in autodict for distances.
    """
    filtered_dict = {k: v for (k, v) in crossDict.items() if 'd' not in k}
    return filtered_dict


def acmDict_to_ACM(acmDict):
    """
    Return square matrix from acmDict.

    Parameters
    ----------
    filename : string
    """

    # this ain't good. It's going to end up with row and col > 4
    # numAnts = int(math.sqrt(len(acmDict.keys())))
    numAnts = 10
    numChans = 512  # make this cooler
    numIndexes = max(len(x) for x in acmDict.values())

    ACM = np.zeros((numAnts, numAnts, numIndexes, numChans), dtype=np.complex)

    for key in acmDict:
        row = int(key[3])
        col = int(key[4])

        for index in range(numIndexes-1):
            ACM[row, col, index, :] = acmDict[key][index]

    return ACM


def crossDict_to_phaseDict(crossDict):
    """
    Return dict of phase elements from a dict of complex numbers

    Parameters
    ----------
    crossDict : dict
    """
    phaseDict = {}
    for key in crossDict:
        phaseDict[key] = np.angle(crossDict[key])

    return phaseDict


def quickInitDicts(arrDict):
    if du.get_config() == 2:
        arrDict = ff.conjugate_array(arrDict, 'ant76')
        del arrDict['ant76']
    for key in arrDict:
        arrDict[key] = ff.lin_interp_zeros(arrDict[key])
    crossDict, autoDict = separate_cross_auto_dicts(arrDict)
    phaseDict = crossDict_to_phaseDict(crossDict)
    crossDict = addDistanceToDict(crossDict)

    return arrDict, crossDict, autoDict, phaseDict

def radDictTodegDict(radDict):
    degDict = {}
    for key in radDict:
        degDict[key] = np.degrees(radDict[key])
    return degDict