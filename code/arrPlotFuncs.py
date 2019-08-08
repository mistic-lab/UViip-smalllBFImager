import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import imageio
import DRAO_utils as du
import parse_DAT as pd
import utils
import calcVirtualHeight as VH
import math
import dictUtils


global fignum
fignum = 0


def allPhaseMagCW(crossDict):
    """
    Plots full length magnitude and phase of all dict items.

    Parameters
    ----------
    crossDict : dictionary (it plots phase, so auto will be boring)
    """

    global fignum
    fig = plt.figure(fignum)
    fig.suptitle('{}kHz CW'.format(du.get_frequency()))
    phase = plt.subplot(211)
    phase.set_title('Phase')
    phase.set_ylabel('Degrees')
    for key in crossDict:
        phase.plot(np.angle(crossDict[key])*180/np.pi, label=key)
        # arr18 was conj
    mag = plt.subplot(212)
    mag.set_title('Magnitude')
    mag.set_ylabel('dB (Voltage)')
    mag.set_xlabel('Samples')
    for key in crossDict:
        mag.plot(10.*np.log10(np.abs(crossDict[key])), label=key)
    plt.legend()

    plt.show(block=False)

    print('Current fignum: {}'.format(fignum))
    fignum += 1


def singlePhaseMagCW(crossDict, key):
    """
    Plots full length magnitude and phase of single dict item.

    Parameters
    ----------
    crossDict : dictionary (it plots phase, so auto will be boring)

    key : which dict item to plot
    """

    global fignum
    fig = plt.figure(fignum)
    fig.suptitle('{}kHz CW - {}'.format(du.get_frequency(), key))
    phase = plt.subplot(211)
    phase.set_title('Phase')
    phase.set_ylabel('Degrees')
    phase.plot(np.angle(crossDict[key])*180/np.pi, label=key)
    mag = plt.subplot(212)
    mag.set_title('Magnitude')
    mag.set_ylabel('dB (Voltage)')
    mag.set_xlabel('Samples')
    mag.plot(10.*np.log10(np.abs(crossDict[key])), label=key)
    # plt.legend()

    plt.show(block=False)

    print('Current fignum: {}'.format(fignum))
    fignum += 1


def avgComplexCW(crossDict, length=0):
    """
    Plots rolling average of length "length" of magnitude and phase of all dict items.

    Parameters
    ----------
    crossDict : dictionary
                (it plots phase, so auto will be boring)
    length :    int (default: len(crossDict[key]))
    """

    global fignum

    if length == 0:  # Only run if no useful length is supplied
        for key in crossDict:
            if key[0] != 'd':
                    length = max(length, len(crossDict[key]))
    avgMagDict = {}
    avgPhaseDict = {}

    for key in crossDict:
        if key[0] != 'd':
            avgMagDict[key] = utils.running_mean(
                10.*np.log10(np.abs(crossDict[key])), length)
            avgPhaseDict[key] = utils.running_mean(
                np.angle(crossDict[key])*180/np.pi, length)

    fig = plt.figure(fignum)
    fig.suptitle('{}kHz CW - rolling averages of length {}'.format(
        du.get_frequency(), length))
    phase = plt.subplot(211)
    phase.set_title('Phase')
    phase.set_ylabel('Degrees')
    for key in crossDict:
        phase.plot(avgPhaseDict[key], label=key)
    mag = plt.subplot(212)
    mag.set_title('Magnitude')
    mag.set_ylabel('dB (Voltage)')
    for key in crossDict:
        mag.plot(avgMagDict[key], label=key)
    plt.legend()
    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1


def avgFloatCW(floatDict, unitLabel='', length=0):
    """
    Plots rolling average of length "length" of all dict items without changing their unit. If input is rads, output is rads.

    Parameters
    ----------
    floatDict : dictionary
    length :    int (default: len(floatDict[key]))
    unitLabel : label for y-axis
    """

    global fignum

    fullAvg = False

    if length == 0:  # Only run if no useful length is supplied
        fullAvg = True
        for key in floatDict:
                length = max(length, len(floatDict[key]))
    avgfloatDict = {}
    for key in floatDict:
        avgfloatDict[key] = utils.running_mean(floatDict[key], length)

    plt.rcParams.update({'font.size': 18})

    plt.figure(fignum)
    plt.ylabel(unitLabel)

    if fullAvg is True:
        for key in avgfloatDict:
            plt.plot(np.repeat(avgfloatDict[key], 2), label=key)
        plt.xlabel('Samples')
        plt.title('{}kHz CW - experiment average'.format(
            du.get_frequency()))
        plt.tick_params(
            axis='x',           # changes apply to the x-axis
            which='both',       # both major and minor ticks are affected
            bottom=False,       # ticks along the bottom edge are off
            top=False,          # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
    else:
        for key in avgfloatDict:
            plt.plot(avgfloatDict[key], label=key)
        plt.xlabel('Samples')
        plt.title('{}kHz CW - rolling averages of length {}'.format(
            du.get_frequency(), length))

    plt.legend()
    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1


def animation3Points(crossDict, key1, key2, key3):
    """
    Plots single point of phase of three dict items with equal space
    between them. It's sort of like a moving histogram.

    Parameters
    ----------
    crossDict : dictionary
        It plots phase, so auto will be boring.
    key1 : string
        Must be a key from fullDict
    key2 : string
        Must be a key from fullDict
    key3 : string
        Must be a key from fullDict
    """

    global fignum

    plt.figure(fignum)
    plt.title('{}kHz 3-point'.format(du.get_frequency()))
    plt.ylabel('Degrees')
    for pt in range(50):
        plt.plot(1, np.angle(crossDict[key1])*180/np.pi,
                 label=key1, marker='o')
        plt.plot(2, np.angle(crossDict[key2])*180/np.pi,
                 label=key2, marker='o')
        plt.plot(3, np.angle(crossDict[key3])*180/np.pi,
                 label=key3, marker='o')
        plt.savefig(pt)
        plt.clf()
    plt.legend()

    print('Fignum: {}'.format(fignum))
    fignum += 1


def stackedArrays(crossDict):
    """
    Plots full signal phase of dict items with equal space between them.
    It's like allMagPhase but with space between signals.

    Parameters
    ----------
    crossDict : dictionary
        It plots phase, so auto will be boring.
    """

    global fignum

    fig = plt.figure(fignum)
    ax = fig.add_subplot(111, projection='3d')
    i = 1
    length = max(len(v) for v in crossDict.values())
    for key in crossDict:
        zline = np.repeat(i, length)
        xline = np.angle(crossDict[key])*180/np.pi
        yline = np.linspace(0, length, num=length)
        ax.plot3D(xline, yline, zline, label=key)

        i += 1

    plt.legend()
    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1


def fullMesh3Arrays(crossDict, refAnt, timeStep=0.4):
    """
    Plots mesh between full phase of three dict items with real distance
    between them.

    Parameters
    ----------
    crossDict : dictionary
        It plots phase, so auto will be boring.
    refAnt : int
        Which antenna is being used as a reference.
    timeStep: float (default: 0.4)
        Seconds between data points.
    """

    global fignum

    length = 0
    plotDict = {}
    for key in crossDict:
        if str(refAnt) in key:
            plotDict[key] = crossDict[key]
            if key[0] != 'd':
                length = max(length, len(plotDict[key]))
    plotDictSortedKeys = sorted(plotDict)

    xValues = np.repeat(0, length)
    zValues = np.repeat(0, length)

    for key in plotDictSortedKeys:
        if key[0] is 'd' and key[1] is not 'T':
            xValues = np.append(xValues, np.repeat(plotDict[key], length))
            print('Adding {} to xValues.'.format(key))
        else:
            zValues = np.append(zValues, np.angle(plotDict[key])*180/np.pi)
            print('Adding {} to zValues.'.format(key))

    t = np.linspace(0, length, length)*timeStep
    yValues = np.hstack((t, t, t, t))

    fig = plt.figure(fignum)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(xValues, yValues, zValues, cmap=cm.magma,
                           linewidth=0)
    fig.colorbar(surf)
    ax.set_xlabel('Distance from A{}[m]'.format(refAnt))
    ax.set_zlabel('Phase from A{}[deg]'.format(refAnt))
    ax.set_ylabel('Time[s]')
    maxPhase = max(zValues)
    minPhase = min(zValues)
    ax.set_zlim(minPhase-5, maxPhase+5)
    fig.tight_layout()
    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1


def varMesh3Arrays(crossDict, refAnt, timeStep=0.4, pltLen=10,
                                 numFrames=200, pltStart=1):
    """
    Plots mesh between phase of three dict items with real distance
    between them.

    Parameters
    ----------
    crossDict : dictionary
        Must include distance items.
    refAnt : int
        Which antenna is being used as a reference.
    timeStep: float (default: 0.4)
        Seconds between data points.
    plotLen : int (default: 10)
        Length of plot (number of samples/frame).
    numFrames : int (default: 200)
        Length of the animation in frames.
    pltStart : int (default 1)
        Which index in the array to start plotting at.
    """

    global fignum

    ans = 'n' # default save to file

    if numFrames < 10:
        ans = input("Save to file [n] or show live plot [y]: ")

    if ans == 'n':
        filepath = './for-animation/'
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print('Will save files to {}'.format(filepath))
    elif ans != 'n' and ans != 'y':
        print('Please enter a valid option (y/n) next time.')
        print('Terminating.')
        return

    plotDict = {}

    for key in crossDict:
        if str(refAnt) in key:
            plotDict[key] = crossDict[key]

    plotDictSortedKeys = sorted(plotDict)

    xValues = np.repeat(0, pltLen)  # Reference distance
    for key in plotDictSortedKeys:
        if key[0] is 'd' and key[1] is not 'T':
            xValues = np.append(xValues, np.repeat(plotDict[key], pltLen))

    for i in range(numFrames):
        zValues = np.repeat(0, pltLen)  # Reference phase
        for key in plotDictSortedKeys:
            if key[0] != 'd':
                zValues = np.append(zValues, np.angle(
                    plotDict[key][pltStart:pltStart+pltLen])*180/np.pi)

        t = np.linspace(pltStart, pltStart+pltLen, pltLen)*timeStep
        yValues = np.hstack((t, t, t, t))

        plt.rcParams.update({'font.size': 18})

        fig = plt.figure(fignum, figsize=(21,5))
        ax = fig.add_subplot(111, projection='3d')
        #  In np.diag, 4 args are x_aspect, y_aspect, z_aspect
        #  All are normalized from 0-1
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax),
                                     np.diag([0.4, 1, 0.95, 1]))
        surf = ax.plot_trisurf(xValues, yValues, zValues, cmap=cm.inferno,
                               linewidth=0)
        fig.colorbar(surf,
                     fraction=0.046,
                     pad=0.04,
                     label='(deg)',
                     orientation='horizontal')
        surf.set_clim(vmin=-180, vmax=180)
        ax.set_xlabel('\nDistance from A{}[m]'.format(refAnt), linespacing=3.2)
        ax.set_zlabel('\nPhase from A{}[deg]'.format(refAnt), linespacing=3.2)
        ax.set_ylabel('\nTime[s]', linespacing=3.2)
        ax.set_zlim(-180, 180)
        ax.set_zticks([-100, 0, 100])
        ax.set_xticks([0, 20, 40, 60])

        ax.view_init(azim=-35, elev=35)
        fig.tight_layout()
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0.12)

        if ans == 'n':
            plt.savefig('./for-animation/animationframe{}.png'.format(i),bbox_inches='tight', pad_inches=0)
            print('Fignum: {}'.format(fignum))
            fignum += 1
            plt.clf()
        elif ans == 'y':
            plt.show(block=False)
            print('Fignum: {}'.format(fignum))
            fignum += 1

        pltStart += 1




def save_pngs_as_gif(filespath, duration=0.1):
    """
    Save .GIF made from folder of .PNG images. Saves as movie.gif
    in current directory.

    Parameters
    ----------
    filespath : string
        Path to the folder which contains the .PNGs
    """

    filenames = os.listdir(filespath)
    i = 0
    images = []
    while i < len(filenames)-1:
        filename = '{}{}.png'.format(filespath, i)
        print('Added {}'.format(filename))
        images.append(imageio.imread(filename))
        i += 1
    imageio.mimsave('./movie.gif', images, duration=duration)


def compare2ReferencesForAnimation(crossDict, timeStep=0.4,
                                   pltLen=10, numFrames=200, pltStart=2000):
    """
    Plots mesh between phase of three dict items with real distance
    between them.

    Parameters
    ----------
    crossDict : dictionary
        It plots phase, so auto will be boring.
    timeStep: float (default: 0.4)
        Seconds between data points.
    plotLen : int (default: 10)
        Length of plot (number of samples/frame).
    numFrames : int (default: 200)
        Length of the animation in frames.
    pltStart : int (default 2000)
        Which index in the array to start plotting at.
    """

    global fignum

    filepath = './for-animation/'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    plotDict2 = {}
    plotDict8 = {}

    plotDict2['ant26'] = crossDict['ant26']
    plotDict2['ant27'] = crossDict['ant27']
    plotDict2['ant28'] = crossDict['ant28']
    plotDict8['ant78'] = crossDict['ant78']
    plotDict8['ant68'] = crossDict['ant68']
    plotDict8['ant28'] = crossDict['ant28']

    plotDict2 = pd.addDistanceToDict(plotDict2)
    plotDict8 = pd.addDistanceToDict(plotDict8)

    # for key in crossDict:
    #     if str(2) in key:
    #         plotDict2[key] = crossDict[key]
    #     if str(8) in key:
    #         plotDict8[key] = crossDict[key]

    plotDict2SortedKeys = sorted(plotDict2)
    plotDict8SortedKeys = sorted(plotDict8)

    xValues2 = []
    xValues8 = xValues2

    for key in plotDict2SortedKeys:
        if key[0] is 'd' and key[1] is not 'T':
            xValues2 = np.append(xValues2, np.repeat(plotDict2[key], pltLen))
    for key in plotDict8SortedKeys:
        if key[0] is 'd' and key[1] is not 'T':
            xValues8 = np.append(xValues8, np.repeat(plotDict8[key], pltLen))

    for i in range(numFrames):
        zValues2 = []
        zValues8 = []
        for key in plotDict2SortedKeys:
            if key[0] != 'd':
                zValues2 = np.append(zValues2, np.angle(
                    plotDict2[key][pltStart:pltStart+pltLen])*180/np.pi)
        for key in plotDict8SortedKeys:
            if key[0] != 'd':
                zValues8 = np.append(zValues8, np.angle(
                    plotDict8[key][pltStart:pltStart+pltLen])*180/np.pi)

        t = np.linspace(pltStart, pltStart+pltLen, pltLen)*timeStep
        yValues = np.hstack((t, t, t))

        fig = plt.figure(fignum, figsize=(10, 5))
        ax8 = fig.add_subplot(122, projection='3d')
        ax8.plot_trisurf(xValues2, yValues, zValues8, cmap=cm.inferno,
                         linewidth=0)
        ax2 = fig.add_subplot(121, projection='3d', sharex=ax8)
        surf2 = ax2.plot_trisurf(xValues2, yValues, zValues2, cmap=cm.inferno,
                                 linewidth=0)

        # fig.colorbar(surf2)
        ax2.set_xlabel('Distance from A2[m]')
        ax2.set_zlabel('Phase from A2[deg]')
        ax2.set_ylabel('Time[s]')
        ax2.set_zlim(-180, 180)
        ax2.set_title('A2 as reference')
        ax8.set_xlabel('Distance from A2[m]')
        ax8.set_zlabel('Phase from A8[deg]')
        ax8.set_ylabel('Time[s]')
        ax8.set_title('A8 as reference')
        ax8.set_zlim(-180, 180)

        ratio = 1

        for ax in [ax2, ax8]:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.set_aspect(abs((xmax-xmin)/(ymax-ymin))*ratio, adjustable='box-forced')

        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.05])
        fig.colorbar(surf2, cax=cbar_ax, orientation='horizontal')
        surf2.set_clim(vmin=-180, vmax=180)

        plt.savefig('./for-animation/{}.png'.format(i))
        plt.clf()

        pltStart += 1

    print('Fignum: {}'.format(fignum))
    fignum += 1


def compare4ReferencesForAnimation(crossDict, timeStep=0.4, pltLen=10,
                                   numFrames=200, pltStart=2000):
    """
    Plots mesh between phase of three dict items with real distance
    between them.

    Parameters
    ----------
    crossDict : dictionary
        It plots phase, so auto will be boring.
    timeStep: float (default: 0.4)
        Seconds between data points.
    plotLen : int (default: 10)
        Length of plot (number of samples/frame).
    numFrames : int (default: 200)
        Length of the animation in frames.
    pltStart : int (default 2000)
        Which index in the array to start plotting at.
    """

    global fignum

    filepath = './for-animation/'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    plotDict2 = {}
    plotDict8 = {}
    plotDict6 = {}
    plotDict7 = {}

    plotDict2['ant26'] = crossDict['ant26']
    plotDict2['ant27'] = crossDict['ant27']
    plotDict2['ant28'] = crossDict['ant28']
    plotDict2 = pd.addDistanceToDict(plotDict2)
    plotDict2['ant22'] = np.zeros(crossDict['ant26'].shape, dtype=np.complex)

    plotDict8['ant88'] = plotDict2['ant22']
    plotDict8['ant78'] = crossDict['ant78']
    plotDict8['ant68'] = crossDict['ant68']
    plotDict8['ant28'] = crossDict['ant28']

    plotDict6['ant26'] = crossDict['ant26']
    plotDict6['ant66'] = plotDict2['ant22']
    plotDict6['ant67'] = np.conjugate(crossDict['ant76'])
    plotDict6['ant68'] = crossDict['ant68']

    plotDict7['ant27'] = crossDict['ant26']
    plotDict7['ant67'] = np.conjugate(crossDict['ant76'])
    plotDict7['ant77'] = plotDict2['ant22']
    plotDict7['ant78'] = crossDict['ant78']

    plotDict2SortedKeys = sorted(plotDict2)
    plotDict8SortedKeys = sorted(plotDict8)
    plotDict6SortedKeys = sorted(plotDict6)
    plotDict7SortedKeys = sorted(plotDict7)

    xValues2 = [np.repeat(0, pltLen)]

    for key in plotDict2SortedKeys:
        if key[0] is 'd' and key[1] is not 'T':
            xValues2 = np.append(xValues2, np.repeat(plotDict2[key], pltLen))

    for i in range(numFrames):
        zValues2 = []
        zValues8 = []
        zValues6 = []
        zValues7 = []
        for key in plotDict2SortedKeys:
            if key[0] != 'd':
                zValues2 = np.append(zValues2, np.angle(
                    plotDict2[key][pltStart:pltStart+pltLen])*180/np.pi)
        for key in plotDict8SortedKeys:
            if key[0] != 'd':
                zValues8 = np.append(zValues8, np.angle(
                    plotDict8[key][pltStart:pltStart+pltLen])*180/np.pi)
        for key in plotDict6SortedKeys:
            if key[0] != 'd':
                zValues6 = np.append(zValues6, np.angle(
                    plotDict6[key][pltStart:pltStart+pltLen])*180/np.pi)
        for key in plotDict7SortedKeys:
            if key[0] != 'd':
                zValues7 = np.append(zValues7, np.angle(
                    plotDict7[key][pltStart:pltStart+pltLen])*180/np.pi)

        t = np.linspace(pltStart, pltStart+pltLen, pltLen)*timeStep
        yValues = np.hstack((t, t, t, t))

        fig = plt.figure(fignum, figsize=(12, 12))
        ax2 = fig.add_subplot(221, projection='3d')
        surf2 = ax2.plot_trisurf(xValues2, yValues, zValues2, cmap=cm.inferno,
                                 linewidth=0)
        ax8 = fig.add_subplot(222, projection='3d', sharex=ax2)
        ax8.plot_trisurf(xValues2, yValues, zValues8, cmap=cm.inferno,
                         linewidth=0)
        ax6 = fig.add_subplot(223, projection='3d', sharex=ax2)
        ax6.plot_trisurf(xValues2, yValues, zValues6, cmap=cm.inferno,
                         linewidth=0)
        ax7 = fig.add_subplot(224, projection='3d', sharex=ax2)
        ax7.plot_trisurf(xValues2, yValues, zValues7, cmap=cm.inferno,
                         linewidth=0)

        # fig.colorbar(surf2)
        ax2.set_xlabel('Distance from A2[m]')
        ax2.set_zlabel('Phase from A2[deg]')
        ax2.set_ylabel('Time[s]')
        ax2.set_zlim(-180, 180)
        ax2.set_title('A2 as reference')
        ax8.set_xlabel('Distance from A2[m]')
        ax8.set_zlabel('Phase from A8[deg]')
        ax8.set_ylabel('Time[s]')
        ax8.set_title('A8 as reference')
        ax8.set_zlim(-180, 180)
        ax6.set_xlabel('Distance from A2[m]')
        ax6.set_zlabel('Phase from A6[deg]')
        ax6.set_ylabel('Time[s]')
        ax6.set_zlim(-180, 180)
        ax6.set_title('A6 as reference')
        ax7.set_xlabel('Distance from A2[m]')
        ax7.set_zlabel('Phase from A7[deg]')
        ax7.set_ylabel('Time[s]')
        ax7.set_title('A7 as reference')
        ax7.set_zlim(-180, 180)

        ratio = 1

        for ax in [ax2, ax8, ax7, ax6]:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.set_aspect(abs((xmax-xmin)/(ymax-ymin))*ratio, adjustable='box-forced')

        fig.subplots_adjust(bottom=0.2)
        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.05])
        fig.colorbar(surf2, cax=cbar_ax, orientation='horizontal')
        surf2.set_clim(vmin=-180, vmax=180)

        plt.savefig('./for-animation/{}.png'.format(i))
        plt.clf()

        pltStart += 1

    print('Fignum: {}'.format(fignum))
    fignum += 1


def FFTAll(crossDict, timeStep=0.4, N=-1):

    global fignum

    crossDict = dictUtils.removeDistanceFromDict(crossDict)

    if N == -1:
        N = len(crossDict[sorted(crossDict.keys())[0]])

    sortedKeys = sorted(crossDict.keys())
    dictSize = len(crossDict)  # Number of subplots to plot
    numCol = 1  # Num of columns to plot
    subInd = 1  # Current index for subplots

    fftCollection = np.empty((10.*np.log10(abs(np.fft.fftshift(np.fft.fft(
        crossDict[sortedKeys[0]], N, axis=0))/N))).shape)

    fig = plt.figure(fignum, figsize=(5, 9))
    axes = fig.subplots(dictSize, numCol, sharex=True)

    for key in sortedKeys:
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=timeStep))
        fftData = 10.*np.log10(abs(np.fft.fftshift(np.fft.fft(
            crossDict[key], N, axis=0))/N))
        axes[subInd-1].plot(freqs, fftData)
        axes[subInd-1].set_title(key, fontdict={'fontsize': 10})
        axes[subInd-1].set_ylabel('dB')
        axes[subInd-1].set_xlim(-1/2/timeStep, 1/2/timeStep)

        fftCollection = np.add(fftCollection, fftData)

        subInd += 1

    axes[subInd-2].set_xlabel('Hz')
    plt.tight_layout()
    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1

    fftCollection = np.true_divide(fftCollection, 6)

    plt.figure(fignum)
    plt.plot(freqs, fftCollection)
    plt.set_xlabel('Hz')
    plt.set_ylabel('dB')
    plt.set_title('Element-wise average of all FFTs')
    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1


def spectrogramAll(phaseDict, timeStep=0.4, NFFT=32, noverlap=16):

    global fignum


    sortedKeys = sorted(phaseDict.keys())
    dictSize = len(phaseDict)  # Number of subplots to plot
    numCol = 1  # Num of columns to plot
    subInd = 1  # Current index for subplots

    fig = plt.figure(fignum)#, figsize=(5, 9))
    axes = fig.subplots(dictSize, numCol, sharex=True)

    for key in sortedKeys:
        axes[subInd-1].specgram(phaseDict[key][:,0], Fs=1/timeStep, NFFT=NFFT, noverlap=16, scale='linear', mode='magnitude')
        axes[subInd-1].set_title(key, fontdict={'fontsize': 10})
        axes[subInd-1].set_ylabel('Frequency [Hz]')

        subInd += 1

    axes[subInd-2].set_xlabel('Time [s]')
    # plt.tight_layout()
    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1


def stream_mesh(crossDict, refAnt, timeStep=0.4, pltLen=10, pltStart=0):
    """
    Animates mesh between phase of three dict items with real distance
    between them.

    Parameters
    ----------
    crossDict : dictionary
        Must include distance items.
    refAnt : int
        Which antenna is being used as a reference.
    timeStep: float (default: 0.4)
        Seconds between data points.
    plotLen : int (default: 10)
        Length of plot (number of samples/frame).
    pltStart : int (default: 0)
        Which index in the array to start plotting at.
    """
    global fignum

    plotDict = {}

    for key in crossDict:
        if str(refAnt) in key:
            plotDict[key] = crossDict[key]

    plotDictSortedKeys = sorted(plotDict)

    numIndexes = len(plotDict[plotDictSortedKeys[0]]) - pltStart
    # x = dist
    # y = time
    # z = phase

    xValues = np.repeat(0, pltLen)  # Reference distance
    for key in plotDictSortedKeys:  # Array of distances
        if key[0] is 'd' and key[1] is not 'T':
            xValues = np.append(xValues, np.repeat(plotDict[key], pltLen))

    zValues = np.repeat(0, pltLen)  # Reference phase
    for key in plotDictSortedKeys:  # Array of phases
        if key[0] != 'd':
            zValues = np.append(zValues, np.angle(
                plotDict[key][pltStart:pltStart+pltLen])*180/np.pi)

    t = np.linspace(pltStart, pltStart+pltLen, pltLen)*timeStep
    yValues = np.hstack((t, t, t, t))  # Array of times

    fig = plt.figure(fignum)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(xValues, yValues, zValues, cmap=cm.inferno,
                           linewidth=0)
    fig.colorbar(surf)
    surf.set_clim(vmin=-180, vmax=180)
    ax.set_xlabel('Distance from A{}[m]'.format(refAnt))
    ax.set_zlabel('Phase from A{}[deg]'.format(refAnt))
    ax.set_ylabel('Time[s]')
    ax.set_zlim(-180, 180)

    fig.tight_layout()

    def update(pltStart):
        utils.printProgressBar(pltStart, numIndexes, prefix='  Streamed:', length=50)
        plt.cla()
        zValues = np.repeat(0, pltLen)  # Reference phase
        for key in plotDictSortedKeys:  # Array of phases
            if key[0] != 'd':
                zValues = np.append(zValues, np.angle(
                    plotDict[key][pltStart:pltStart+pltLen])*180/np.pi)
        t = np.linspace(pltStart, pltStart+pltLen, pltLen)*timeStep
        yValues = np.hstack((t, t, t, t))  # Array of times
        ax.set_zlim(-180, 180)
        ax.plot_trisurf(xValues, yValues, zValues, cmap=cm.inferno,
                        linewidth=0)
        return ax

    anim = manimation.FuncAnimation(fig, update, frames=numIndexes,
                                    interval=400)

    plt.show()

    print('Fignum: {}'.format(fignum))
    fignum += 1


def all_ionospheric_heights(phaseDict, mean=200, planar=True):
    """
    Plots the calculated ionospheric height (assuming planar wave arrival) between each set of phaseDict elements. A rolling mean is used.

    Parameters
    ----------
    phaseDict : dictionary
    mean : int
        rolling average window length (default=200)
    planar : boolean
        if False, calculated using triangle method

    """

    global fignum

    plotDict = {}
    freq = du.get_frequency()
    if du.get_config() != 2:
        print('Continuing, but config 2 is all this makes sense to me for. Not sure how to calculate height for config 1.')

    for key in phaseDict:
        plotDict[key] = utils.running_mean(phaseDict[key], mean)

    try:
        plotDict.pop('ant67')
        plotDict.pop('ant68')
        plotDict.pop('ant78')
    except:
        print('nothing to pop')

    length = len(plotDict[sorted(plotDict.keys())[0]])-mean+1

    for key in plotDict:
        if all(item in key for item in ['2', '6']):
            bl = 20
            td = 0
        if all(item in key for item in ['2', '7']):
            bl = 40
            td = 0
        if all(item in key for item in ['2', '8']):
            bl = 60
            td = 0
        if all(item in key for item in ['6', '7']):
            bl = 20
            td = 20
        if all(item in key for item in ['6', '8']):
            bl = 40
            td = 20
        if all(item in key for item in ['7', '8']):
            bl = 20
            td = 40
        print('bl={}, key={}'.format(bl, key))

        for i in range(0, length-1):
            phase = VH.get_phase_diff_as_dist(plotDict[key][i], freq*1000)
            if planar == True:
                plotDict[key][i] = VH.find_virtual_height_plane_wave(phase, 19841+td, bl)
            elif planar == False:
                plotDict[key][i] = VH.find_virtual_height_triangles(phase,19841+td,bl)

    time=np.linspace(0,len(plotDict['ant28']), num=len(plotDict['ant28']), endpoint=False)*0.4
    # print('pd: {}, time: {}'.format(len(phaseDict['ant28']), len(time)))
    
    plt.rcParams.update({'font.size': 18})

    plt.figure(fignum)
    for key in plotDict:
        plt.plot(time, plotDict[key]/1000, label=key)

    plt.xlabel('Time (s)')
    plt.ylabel('Height (km)')
    plt.title('Ionospheric virtual reflection height at {}MHz assuming planar wave={}'.format(freq/1000, planar))
    plt.xlim(time[0], time[-1])
    # plt.ylim(0, 400)
    plt.legend()
    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1


def ionospheric_height(phaseDict, key, mean=200, planar=True):
    """
    Plots the calculated ionospheric height (assuming planar wave arrival) between a single phaseDict element. A rolling mean is used.

    Parameters
    ----------
    phaseDict : dictionary
    key : string
        which element to plot
    mean : int
        rolling average window length (default=200)
    planar : boolean
        if False, calculated using triangle method
    """

    global fignum

    plotDict = {}
    freq = du.get_frequency()
    if du.get_config() != 2:
        print('Continuing, but config 2 is all this makes sense to me for. Not sure how to calculate height for config 1.')


    plotDict[key] = utils.running_mean(phaseDict[key], mean)

    length = len(plotDict[sorted(plotDict.keys())[0]])-mean+1

    if all(item in key for item in ['2', '6']):
        bl = 20
        td = 0
    if all(item in key for item in ['2', '7']):
        bl = 40
        td = 0
    if all(item in key for item in ['2', '8']):
        bl = 60
        td = 0
    if all(item in key for item in ['6', '7']):
        bl = 20
        td = 20
    if all(item in key for item in ['6', '8']):
        bl = 40
        td = 20
    if all(item in key for item in ['7', '8']):
        bl = 20
        td = 40
    print('bl={}, key={}'.format(bl, key))

    for i in range(0, length-1):
        phase = VH.get_phase_diff_as_dist(plotDict[key][i], freq*1000)
        if planar == True:
            plotDict[key][i] = VH.find_virtual_height_plane_wave(phase, 19841+td, bl)
        elif planar == False:
            plotDict[key][i] = VH.find_virtual_height_triangles(phase,19841+td,bl)

    time=np.linspace(0,length)*0.4
    plt.figure(fignum)
    for key in plotDict:
        plt.plot(time, plotDict[key]/1000, label=key)

    plt.xlabel('Time (s)')
    plt.ylabel('Height (km)')
    plt.title('Ionospheric virtual reflection height at {}MHz assuming planar wave={}'.format(freq/1000, planar))
    plt.xlim(time[0], time[-1])
    # plt.ylim(0, 400)
    plt.legend()
    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1


def phase_ramp(heights):
    """
    Plots the expected phase ramp for a set ionospheric height. A2 is assumed to be the reference

    Parameters
    ----------
    heights : array of ints (km)
    """

    global fignum

    l1 = 19841 #distance from tx to a2

    baselines = [0, 20, 40, 60]

    phase_diff = lambda bl, h: math.sqrt(h**2+(l1+bl)**2) - math.sqrt(h**2+l1**2)

    plt.figure(fignum)
    
    for i in heights:
        phase_differences = []
        for x in baselines:
            z = phase_diff(x, i*1000)
            phase_differences.append(z)

        plt.plot(baselines, phase_differences, label='{} km'.format(i))

    plt.xlabel('Distance from A2 (m)')
    plt.ylabel('Default phase difference (m)')
    plt.legend()
    plt.show(block=False)

    print('Fignum: {}'.format(fignum))
    fignum += 1
