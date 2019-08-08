import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import utils

global fignum
fignum = 0

freqs = np.arange(3.584, 4.096, .001)
chan = lambda frequency: 512 - (4096 - frequency)
label_freqs = np.array([3600, 3700, 3800, 3900, 4000])
label_locs = np.zeros(len(label_freqs))
for i in range(len(label_freqs)):
    label_locs[i] = chan(label_freqs[i])


def __make_AVG_ACM__(ACM):
    acmSide = ACM.shape[0]
    numIndexes = ACM.shape[2]

    avgACM = np.zeros((acmSide, acmSide, numIndexes), dtype=np.complex)

    for i in range(numIndexes-1):
        utils.printProgressBar(i+1, numIndexes-1, prefix='  Averaging ACM:', length=50)
        for row in range(acmSide-1):
            for col in range(acmSide-1):
                avgACM[row, col, i] = np.mean(ACM[row, col, i, :])

    return avgACM


def stream_grids(ACM, speed_multiplier=1, avgACM=0):

    global fignum

    if avgACM==0:
        avgACM = __make_AVG_ACM__(ACM)
    
    numIndexes = avgACM.shape[2]
    magACM = np.zeros(numIndexes, dtype=np.ndarray)
    phaseACM = np.zeros(numIndexes, dtype=np.ndarray)

    magHigh = np.ceil(np.max(np.log10(np.abs(avgACM))))

    for i in range(numIndexes-1):
        magACM[i] = np.log10(np.abs(avgACM[:, :, i]))
        phaseACM[i] = np.angle(avgACM[:, :, i])*180/np.pi

    fig = plt.figure(fignum, figsize=(15, 8))
    plt.suptitle('Array Covariance Matrix (index=ant#)')

    plt.rcParams.update({'font.size': 22})

    mag = fig.add_subplot(121)
    mag.set_title('Magnitude')
    cmmag = mag.matshow(magACM[0], cmap=cm.Reds)
    cmmag.set_clim(vmin=0, vmax=magHigh)
    fig.colorbar(cmmag, fraction=0.046, pad=0.04, label='(dB)')

    phase = fig.add_subplot(122)
    phase.set_title('Phase')
    cmphase = phase.matshow(phaseACM[0], cmap=cm.seismic)
    cmphase.set_clim(vmin=-180, vmax=180)
    fig.colorbar(cmphase, fraction=0.046, pad=0.04, label='(deg)')

    def update(j):
        utils.printProgressBar(j, numIndexes, prefix='  Streamed:', length=50)
        cmmag = mag.matshow(magACM[j], cmap=cm.Reds)
        cmphase = phase.matshow(phaseACM[j], cmap=cm.seismic)
        return cmmag, cmphase

    anim = manimation.FuncAnimation(fig, update, frames=numIndexes,
                                    interval=400*1/speed_multiplier, blit=True)

    plt.show()

    print('Fignum: {}'.format(fignum))
    fignum += 1


def freeze_grids(ACM, index=1, avgACM='needed'):

    global fignum
    
    if avgACM=='needed':
        avgACM = __make_AVG_ACM__(ACM)

    numIndexes = 1
    magACM = np.zeros(numIndexes-1, dtype=np.ndarray)
    phaseACM = np.zeros(numIndexes-1, dtype=np.ndarray)

    magHigh = np.ceil(np.max(np.log10(np.abs(avgACM))))

    magACM = np.log10(np.abs(avgACM[:, :, index]))
    phaseACM = np.angle(avgACM[:, :, index])*180/np.pi

    plt.rcParams.update({'font.size': 22})

    fig = plt.figure(fignum, figsize=(12, 5))
    # plt.suptitle('Array Covariance Matrix (index=ant#)')

    mag = fig.add_subplot(121)
    mag.set_title('Magnitude', pad=20)
    cmmag = mag.matshow(magACM, cmap=cm.Reds)
    cmmag.set_clim(vmin=0, vmax=magHigh)
    fig.colorbar(cmmag, fraction=0.046, pad=0.04, label='(dB)')

    phase = fig.add_subplot(122)
    phase.set_title('Phase', pad=20)
    cmphase = phase.matshow(phaseACM, cmap=cm.seismic)
    cmphase.set_clim(vmin=-180, vmax=180)
    fig.colorbar(cmphase, fraction=0.046, pad=0.04, label='(deg)')

    # plt.subplots_adjust(bottom=0.0) #! not working, manually adjusting when saving
    plt.tight_layout()

    plt.show()

    print('Fignum: {}'.format(fignum))
    fignum += 1


def stream_spectrum(acmDict, key, speed_multiplier=1):
    global fignum

    numIndexes = len(acmDict[key])
    magVal = np.zeros(numIndexes, dtype=np.ndarray)
    phaseVal = np.zeros(numIndexes, dtype=np.ndarray)

    magMat = np.zeros(numIndexes, dtype=np.ndarray)
    phaseMat = np.zeros(numIndexes, dtype=np.ndarray)

    matWidth = acmDict[key].shape[1]
    matHeight = int(acmDict[key].shape[1]*5/4)

    for i in range(len(magMat)):
        magMat[i] = np.zeros((matHeight, matWidth))
        phaseMat[i] = np.zeros((matHeight, matWidth))

    magHigh = np.ceil(np.max(np.log10(np.abs(acmDict[key]))))

    for i in range(numIndexes-1):
        magVal[i] = np.log10(np.abs(acmDict[key][i]))
        phaseVal[i] = np.angle(acmDict[key][i])*180/np.pi

    for j in range(len(magVal)+1):
        if j < matHeight:
            temp = np.arange(j, 0, -1)
        else:
            temp = np.arange(j, j-matHeight, -1)
        for i in range(len(temp)):
            magMat[j-1][i, :] = magVal[temp[i]-1]
            phaseMat[j-1][i, :] = phaseVal[temp[i]-1]
            
    plt.rcParams.update({'font.size': 18})

    fig = plt.figure(fignum, figsize=(10, 8))
    fig.suptitle('{} Spectra'.format(key))

    gs = GridSpec(4, 5)
    gs.update(left=0.1, right=1.2, wspace=1, hspace=1)

    magSpec = fig.add_subplot(gs[0:2])
    magSpec.set_title('Magnitude (dB)')
    magLine, = magSpec.plot(freqs, magVal[0])
    magSpec.set_ylim(0, magHigh)

    phaseSpec = fig.add_subplot(gs[2:4])
    phaseSpec.set_title('Phase (deg)')
    phaseLine, = phaseSpec.plot(freqs, phaseVal[0])
    phaseSpec.set_ylim(-180, 180)

    magWater = fig.add_subplot(gs[1:, 0:2])
    cmmag = magWater.matshow(magMat[0], cmap=cm.seismic)
    # magWater.set_xticks(label_locs)
    # magWater.set_xticklabels(label_freqs)
    cmmag.set_clim(0, vmax=magHigh)

    phaseWater = fig.add_subplot(gs[1:, 2:4])
    cmphase = phaseWater.matshow(phaseMat[0], cmap=cm.seismic)
    cmphase.set_clim(vmin=-180, vmax=180)

    def update(j):
        utils.printProgressBar(j, numIndexes, prefix='  Streamed:', length=50)

        plt.cla()

        magLine.set_ydata(magVal[j])
        phaseLine.set_ydata(phaseVal[j])
        magWater.matshow(magMat[j], cmap=cm.seismic)
        phaseWater.matshow(phaseMat[j], cmap=cm.seismic)

        return [magLine, phaseLine, magWater, phaseWater,]

    anim = manimation.FuncAnimation(fig, update, frames=numIndexes,
                                    interval=400*1/speed_multiplier, blit=True)
    # http://devosoft.org/making-efficient-animations-in-matplotlib-with-blitting/
    plt.show()

    print('Fignum: {}'.format(fignum))
    fignum += 1


def freeze_spectrum(acmDict, key, plotT='firstFull'):
    global fignum

    numIndexes = len(acmDict[key])
    magVal = np.zeros(numIndexes, dtype=np.ndarray)
    phaseVal = np.zeros(numIndexes, dtype=np.ndarray)

    magMat = np.zeros(numIndexes, dtype=np.ndarray)
    phaseMat = np.zeros(numIndexes, dtype=np.ndarray)

    matWidth = acmDict[key].shape[1]
    matHeight = int(acmDict[key].shape[1]*5/4)
    print('matWidth={}'.format(matWidth))
    print('matHeight={}'.format(matHeight))

    if plotT == 'end':
        plotIndex = -2
    elif plotT =='firstFull':
        plotIndex = matHeight
    else:
        raise ValueError("{} not an option for 'plotT' parameter. Try 'end' or 'firstFull'".format(plotT))

    for i in range(len(magMat)):
        magMat[i] = np.zeros((matHeight, matWidth))
        phaseMat[i] = np.zeros((matHeight, matWidth))

    magHigh = np.ceil(np.max(np.log10(np.abs(acmDict[key]))))

    for i in range(numIndexes-1):
        magVal[i] = np.log10(np.abs(acmDict[key][i]))
        phaseVal[i] = np.angle(acmDict[key][i])*180/np.pi

    for j in range(len(magVal)+1):
        utils.printProgressBar(j+1, len(magVal)+1, prefix='  Making mats:', length=50)
        if j < matHeight:
            temp = np.arange(j, 0, -1)
        else:
            temp = np.arange(j, j-matHeight, -1)
        for i in range(len(temp)):
            magMat[j-1][i, :] = magVal[temp[i]-1]
            phaseMat[j-1][i, :] = phaseVal[temp[i]-1]


    plt.rcParams.update({'font.size': 18})

    fig = plt.figure(fignum, figsize=(10, 8))
    fig.suptitle('{} Spectra'.format(key))

    gs = GridSpec(4, 4)
    gs.update(wspace=0.1, hspace=0)

    magSpec = fig.add_subplot(gs[0:2])
    magSpec.set_title('Magnitude')
    magLine = magSpec.plot(freqs, magVal[plotIndex])
    magSpec.set_xticklabels(" ")
    magSpec.set_ylabel('(dB)')
    magSpec.set_xlim(freqs[0], freqs[-1])
    magSpec.set_ylim(0, magHigh)

    phaseSpec = fig.add_subplot(gs[2:4])
    phaseSpec.set_title('Phase')
    phaseLine = phaseSpec.plot(freqs, phaseVal[plotIndex])
    phaseSpec.set_xticklabels(" ")
    phaseSpec.yaxis.tick_right()
    phaseSpec.yaxis.set_label_position("right")
    phaseSpec.set_ylabel('(deg)')
    phaseSpec.set_xlim(freqs[0], freqs[-1])
    phaseSpec.set_ylim(-180, 180)

    magWater = fig.add_subplot(gs[1:, 0:2])
    cmmag = magWater.matshow(magMat[plotIndex], cmap=cm.seismic)
    magWater.xaxis.set_label_position("bottom")
    magWater.xaxis.set_ticks_position('bottom')
    magWater.set_xticks(label_locs)
    magWater.set_xticklabels(label_freqs/1000)
    magWater.set_xlabel('Frequency (MHz)')

    magWater.set_ylabel('Time (s)')
    magWater.set_yticklabels(np.arange(-40, 280, 40))
    cmmag.set_clim(0, vmax=magHigh)

    phaseWater = fig.add_subplot(gs[1:, 2:4])
    cmphase = phaseWater.matshow(phaseMat[plotIndex], cmap=cm.seismic)
    phaseWater.yaxis.tick_right()
    phaseWater.yaxis.set_label_position("right")
    phaseWater.xaxis.set_label_position("bottom")
    phaseWater.xaxis.set_ticks_position('bottom')
    phaseWater.set_xticks(label_locs)
    phaseWater.set_xticklabels(label_freqs/1000)
    phaseWater.set_ylabel('Time (s)')
    phaseWater.set_xlabel('Frequency (MHz)')
    phaseWater.set_yticklabels(np.arange(-40, 280, 40))
    cmphase.set_clim(vmin=-180, vmax=180)

    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1


def make_mats(acmDict, key):
    """
    Makes variables to be used for plotting for full experiment

    Parameters
    ----------
    acmDict : dict
    key : acmDict key to use
    """
    numIndexes = len(acmDict[key])
    magVal = np.zeros(numIndexes, dtype=np.ndarray)
    phaseVal = np.zeros(numIndexes, dtype=np.ndarray)

    magMat = np.zeros(numIndexes, dtype=np.ndarray)
    phaseMat = np.zeros(numIndexes, dtype=np.ndarray)

    matWidth = acmDict[key].shape[1]
    matHeight = int(acmDict[key].shape[1]*5/4)

    for i in range(len(magMat)):
        magMat[i] = np.zeros((matHeight, matWidth))
        phaseMat[i] = np.zeros((matHeight, matWidth))

    magHigh = np.ceil(np.max(np.log10(np.abs(acmDict[key]))))

    for i in range(numIndexes-1):
        magVal[i] = np.log10(np.abs(acmDict[key][i]))
        phaseVal[i] = np.angle(acmDict[key][i])*180/np.pi

    for j in range(len(magVal)+1):
        utils.printProgressBar(j+1, len(magVal)+1, prefix='  Making mats:', length=50)
        if j < matHeight:
            temp = np.arange(j, 0, -1)
        else:
            temp = np.arange(j, j-matHeight, -1)
        for i in range(len(temp)):
            magMat[j-1][i, :] = magVal[temp[i]-1]
            phaseMat[j-1][i, :] = phaseVal[temp[i]-1]

    return magMat, phaseMat, magVal, phaseVal, magHigh


def make_mat(acmDict, key, index):
    """
    Makes variables to be used for plotting (single frame)

    Parameters
    ----------
    acmDict : dict
    key : acmDict key to use
    index : int
        whichever END index you want to plot up to
    """
    numIndexes = 1
    magVal = np.zeros(index, dtype=np.ndarray)
    phaseVal = np.zeros(index, dtype=np.ndarray)

    magMat = np.zeros(numIndexes, dtype=np.ndarray)
    phaseMat = np.zeros(numIndexes, dtype=np.ndarray)

    matWidth = acmDict[key].shape[1]
    matHeight = int(acmDict[key].shape[1]*5/4)

    magMat[0] = np.zeros((matHeight, matWidth))
    phaseMat[0] = np.zeros((matHeight, matWidth))

    for i in range(index-1):
        magVal[i] = np.log10(np.abs(acmDict[key][i]))
        phaseVal[i] = np.angle(acmDict[key][i])*180/np.pi

    if index < matHeight:
        temp = np.arange(index, 0, -1)
    else:
        temp = np.arange(index, index-matHeight, -1)
    for i in range(len(temp)):
        magMat[0][i, :] = magVal[temp[i]-1]
        phaseMat[0][i, :] = phaseVal[temp[i]-1]

    magHigh = np.ceil(np.max(np.log10(np.abs(acmDict[key]))))

    return magMat, phaseMat, magVal, phaseVal, magHigh


def plot_mat(magMat, phaseMat, magVal, phaseVal, key, magHigh, plotT='firstFull', index=1):
    """
    Plots single spectra/waterfall frame.

    Parameters
    ----------
    all the outputs from make_mat() : arrays and stuff
    plotT : string
        'firstFull' = first filled waterfall from t=0
        'end' = last filled waterfall up to t=end
        'index' = plots some specific frame. If you use this also pass in the frame number to index as an int.
    index : int
        see above. Only need if plotT=='index'
    """

    global fignum

    if plotT == 'end':
        plotIndex = -2
        index = len(magVal)
    elif plotT == 'firstFull':
        plotIndex = magMat[0].shape[0]-1
    elif plotT == 'index':
        plotIndex = 0
    else:
        raise ValueError("{} not an option for 'plotT' parameter. Try 'end' or 'firstFull'".format(plotT))

    fig = plt.figure(fignum, figsize=(10, 8))
    fig.suptitle('{} Spectra'.format(key))

    gs = GridSpec(4, 4)
    gs.update(wspace=0.1, hspace=0)

    magSpec = fig.add_subplot(gs[0:2])
    magSpec.set_title('Magnitude')
    magLine = magSpec.plot(freqs, magVal[plotIndex])
    magSpec.set_xticklabels(" ")
    magSpec.set_ylabel('(dB)')
    magSpec.set_xlim(freqs[0], freqs[-1])
    magSpec.set_ylim(0, magHigh)

    phaseSpec = fig.add_subplot(gs[2:4])
    phaseSpec.set_title('Phase')
    phaseLine = phaseSpec.plot(freqs, phaseVal[plotIndex])
    phaseSpec.set_xticklabels(" ")
    phaseSpec.yaxis.tick_right()
    phaseSpec.yaxis.set_label_position("right")
    phaseSpec.set_ylabel('(deg)')
    phaseSpec.set_xlim(freqs[0], freqs[-1])
    phaseSpec.set_ylim(-180, 180)

    magWater = fig.add_subplot(gs[1:, 0:2])
    cmmag = magWater.matshow(magMat[plotIndex], cmap=cm.seismic)
    magWater.xaxis.set_label_position("bottom")
    magWater.xaxis.set_ticks_position('bottom')
    magWater.set_xticks(label_locs)
    magWater.set_xticklabels(label_freqs/1000)
    magWater.set_xlabel('Frequency (MHz)')

    magWater.set_ylabel('Time (s)')
    ylabels = np.arange(-40, 280, 40)+(index+magMat[0].shape[0])*0.4
    ylabels = np.arange((((index-magMat[0].shape[0])*0.4)-40), (index*0.4)+24, 40)
    magWater.set_yticklabels(ylabels)
    cmmag.set_clim(0, vmax=magHigh)

    phaseWater = fig.add_subplot(gs[1:, 2:4])
    cmphase = phaseWater.matshow(phaseMat[plotIndex], cmap=cm.seismic)
    phaseWater.yaxis.tick_right()
    phaseWater.yaxis.set_label_position("right")
    phaseWater.xaxis.set_label_position("bottom")
    phaseWater.xaxis.set_ticks_position('bottom')
    phaseWater.set_xticks(label_locs)
    phaseWater.set_xticklabels(label_freqs/1000)
    phaseWater.set_ylabel('Time (s)')
    phaseWater.set_xlabel('Frequency (MHz)')
    phaseWater.set_yticklabels(ylabels)
    cmphase.set_clim(vmin=-180, vmax=180)

    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1
