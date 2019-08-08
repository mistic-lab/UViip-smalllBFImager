import utils
import parse_DAT as pd


@utils.run_once
def globalize_notJunk():
    """
    Check that filename exists and make notJunk modularly global.
    """

    try:
        filename = pd.DATfile
    except AttributeError:
        filename = utils.filename_from_GUI()

    global notJunk
    junk, notJunk = filename.split('/Config')


def get_config():
    """
    Returns antenna configuration (1 or 2) as int.
    """

    globalize_notJunk()
    config = int(notJunk[0])
    return config


def get_frequency():
    """
    Returns TX/RX frequency in use as int in kHz.
    """

    globalize_notJunk()
    local_notJunk = notJunk[2:-4]  # get rid of .dat and configX/
    frequency = int(local_notJunk.split('_')[0])
    return frequency


def get_dataWidth_padding():
    """
    Returns data_width and padding for building data arrays.
    """

    globalize_notJunk()
    frequency = get_frequency()
    if 'CW' in notJunk:
        data_width = 1
        padding = 0
    elif 'LFM' in notJunk:
        data_width = int(notJunk.split('_')[1]) - frequency
        padding = 2

    return data_width, padding


def get_antennaDict():
    """
    Returns dictionary of beamformer indexes to antenna numbers (depends
    on configuration).
    """

    config = get_config()
    BFIndexes = [0, 8, 1, 9]
    antDict = {}
    if config == 1:
        antennas = ['1', '4', '5', '6']
    elif config == 2:
        antennas = ['2', '6', '7', '8']

    antDict = dict(zip(BFIndexes, antennas))

    return antDict


def get_ACMDict():
    """
    Returns dictionary of beamformer indexes to linear numbers.
    """

    BFIndexes = [0, 8, 1, 9]
    antDict = {}
    antennas = [0, 1, 2, 3]

    antDict = dict(zip(BFIndexes, antennas))

    return antDict
