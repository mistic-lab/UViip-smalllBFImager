import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
CHANS = [0,1,8,9]
chan = 133
string = 'dummy'

array01 = np.zeros((0,11)).astype(np.complex)
array08 = np.zeros((0,11)).astype(np.complex)
array09 = np.zeros((0,11)).astype(np.complex)
array18 = np.zeros((0,11)).astype(np.complex)
array19 = np.zeros((0,11)).astype(np.complex)
array89 = np.zeros((0,11)).astype(np.complex)

with open(filename, 'rb') as f:

    string = f.read(8229)

    while (len(string) == 8229):

        topic, sts, messagedata = string.split(' ', 2)

        dummy, row, col = topic.split(':')
        row = int(row)
        col = int(col)

       # print row, col, sts

        data = np.fromstring(messagedata, dtype=np.complex)[::-1]

        tmp = np.zeros((1,11)).astype(np.complex)
        tmp[0, :] = data[chan:chan+11]

        if (row == 0):
            if (col == 1):
                array01 = np.append(array01, tmp, 0)
            if (col == 8):
                array08 = np.append(array08, tmp, 0)
            if (col == 9):
                array09 = np.append(array09, tmp, 0)
        if (row == 1):
            if (col == 8):
                array18 = np.append(array18, tmp, 0)
            if (col == 9):
                array19 = np.append(array19, tmp, 0)
        if (row == 8):
            if (col == 9):
                array89 = np.append(array89, tmp, 0)



        string = f.read(8229)

plt.figure()
plt.subplot(161)
plt.imshow(np.angle(array01[:1000])*180/np.pi, aspect='auto', cmap='gray', clim=[-50,50])
plt.subplot(162)
plt.imshow(np.angle(array08[:1000])*180/np.pi, aspect='auto', cmap='gray', clim=[-50,50])
plt.subplot(163)
plt.imshow(np.angle(array09[:1000])*180/np.pi, aspect='auto', cmap='gray', clim=[-50,50])
plt.subplot(164)
plt.imshow(np.angle(array18[:1000])*180/np.pi, aspect='auto', cmap='gray', clim=[-50,50])
plt.subplot(165)
plt.imshow(np.angle(array19[:1000])*180/np.pi, aspect='auto', cmap='gray', clim=[-50,50])
plt.subplot(166)
plt.imshow(np.angle(array89[:1000])*180/np.pi, aspect='auto', cmap='gray', clim=[-50,50])
plt.show()

#img = np.zeros((len(array01), ))
#plt.figure()
#plt.hist(np.angle(array09))
#plt.show()
#
#plt.figure()
#plt.subplot(211)
#plt.plot(np.angle(array01)*180/np.pi, label='2/7')
#plt.plot(np.angle(array08)*180/np.pi, label='2/6')
#plt.plot(np.angle(array09)*180/np.pi, label='2/8')
#plt.plot(np.angle(array18)*180/np.pi, label='7/6')
#plt.plot(np.angle(array19)*180/np.pi, label='7/8')
#plt.plot(np.angle(array89)*180/np.pi, label='6/8')
#plt.legend()
#plt.subplot(212)
#plt.plot(10.*np.log10(np.abs(array01)), label='2/7')
#plt.plot(10.*np.log10(np.abs(array08)), label='2/6')
#plt.plot(10.*np.log10(np.abs(array09)), label='2/8')
#plt.plot(10.*np.log10(np.abs(array18)), label='7/6')
#plt.plot(10.*np.log10(np.abs(array19)), label='7/8')
#plt.plot(10.*np.log10(np.abs(array89)), label='6/8')
#plt.show()

#plt.figure()
#plt.hist(np.abs(array01), bins=500, normed=True)
#plt.show()