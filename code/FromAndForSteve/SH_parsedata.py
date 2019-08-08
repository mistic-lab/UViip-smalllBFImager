import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
CHANS = [0, 1, 8, 9]
chan = 113
string = 'dummy'

array01 = np.zeros((0, 11)).astype(np.complex)
array08 = np.zeros((0, 11)).astype(np.complex)
array09 = np.zeros((0, 11)).astype(np.complex)
array18 = np.zeros((0, 11)).astype(np.complex)
array19 = np.zeros((0, 11)).astype(np.complex)
array89 = np.zeros((0, 11)).astype(np.complex)

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
                array01 = np.append(array01, np.conjugate(tmp), 0)
            if (col == 8):
                array08 = np.append(array08, np.conjugate(tmp), 0)
            if (col == 9):
                array09 = np.append(array09, np.conjugate(tmp), 0)
        if (row == 1):
            if (col == 8):
                array18 = np.append(array18, tmp, 0)
            if (col == 9):
                array19 = np.append(array19, tmp, 0)
        if (row == 8):
            if (col == 9):
                array89 = np.append(array89, tmp, 0)



        string = f.read(8229)

#img = np.zeros((len(array01), ))

print np.shape(array01)
plt.figure()

plt.subplot(161)
plt.imshow(np.angle(array01[:400])*180/np.pi, aspect='auto', cmap='seismic')
plt.clim([-180, 180])

plt.subplot(162)
plt.imshow(np.angle(array08[:400])*180/np.pi, aspect='auto', cmap='seismic')
plt.clim([-180, 180])

plt.subplot(163)
plt.imshow(np.angle(array09[:400])*180/np.pi, aspect='auto', cmap='seismic')
plt.clim([-180, 180])

plt.subplot(164)
plt.imshow(np.angle(array18[:400])*180/np.pi, aspect='auto', cmap='seismic')
plt.clim([-180, 180])

plt.subplot(165)
plt.imshow(np.angle(array19[:400])*180/np.pi, aspect='auto', cmap='seismic')
plt.clim([-180, 180])

plt.subplot(166)
plt.imshow(np.angle(array89[:400])*180/np.pi, aspect='auto', cmap='seismic')
plt.clim([-180, 180])

plt.tight_layout()
plt.show()

#plt.figure()
#plt.subplot(211)
#plt.plot(np.angle(array01)*180/np.pi, label='1/5')
#plt.plot(np.angle(array08)*180/np.pi, label='1/4')
#plt.plot(np.angle(array09)*180/np.pi, label='1/6')
#plt.plot(np.angle(array18)*180/np.pi, label='5/4')
#plt.plot(np.angle(array19)*180/np.pi, label='5/6')
#plt.plot(np.angle(array89)*180/np.pi, label='4/6')
#plt.legend()
#plt.subplot(212)
#plt.plot(10.*np.log10(np.abs(array01)), label='1/5')
#plt.plot(10.*np.log10(np.abs(array08)), label='1/4')
#plt.plot(10.*np.log10(np.abs(array09)), label='1/6')
#plt.plot(10.*np.log10(np.abs(array18)), label='5/4')
#plt.plot(10.*np.log10(np.abs(array19)), label='5/6')
#plt.plot(10.*np.log10(np.abs(array89)), label='4/6')
#plt.show()
