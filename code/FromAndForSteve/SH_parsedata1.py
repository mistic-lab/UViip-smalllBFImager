import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
CHANS = [0,1,8,9]
chan = 115
string = 'dummy'

array01 = np.zeros((0,7)).astype(np.complex)
array08 = np.zeros((0,7)).astype(np.complex)
array09 = np.zeros((0,7)).astype(np.complex)
array18 = np.zeros((0,7)).astype(np.complex)
array19 = np.zeros((0,7)).astype(np.complex)
array89 = np.zeros((0,7)).astype(np.complex)

with open(filename, 'rb') as f:

    string = f.read(8229)

    while (len(string) == 8229):

        topic, sts, messagedata = string.split(' ', 2)

        dummy, row, col = topic.split(':')
        row = int(row)
        col = int(col)

       # print row, col, sts

        data = np.fromstring(messagedata, dtype=np.complex)[::-1]

        tmp = np.zeros((1,7)).astype(np.complex)
        tmp[0, :] = data[chan:chan+7]

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

#img = np.zeros((len(array01), ))

print np.shape(array01)
plt.figure()

plt.subplot(161)
plt.imshow(np.angle(array01[500:600])*180/np.pi, aspect='auto', cmap='seismic', vmin=-50, vmax=50)
plt.title('1/5')

plt.subplot(162)
plt.imshow(np.angle(array08[500:600])*180/np.pi, aspect='auto', cmap='seismic', vmin=-50, vmax=50)
plt.title('1/4')

plt.subplot(163)
plt.imshow(np.angle(array09[500:600])*180/np.pi, aspect='auto', cmap='seismic', vmin=-50, vmax=50)
plt.title('1/6')

plt.subplot(164)
plt.imshow(np.angle(array18[500:600])*180/np.pi, aspect='auto', cmap='seismic', vmin=-50, vmax=50)
plt.title('5/4')

plt.subplot(165)
plt.imshow(np.angle(array19[500:600])*180/np.pi, aspect='auto', cmap='seismic', vmin=-50, vmax=50)
plt.title('5/6')

plt.subplot(166)
plt.imshow(np.angle(array89[500:600])*180/np.pi, aspect='auto', cmap='seismic', vmin=-50, vmax=50)
plt.title('4/6')

plt.tight_layout()
plt.show()

#plt.figure()
#plt.subplot(211)
#plt.plot(np.angle(np.mean(array01, 1))*180/np.pi, label='1/5')
#plt.plot(np.angle(np.mean(array08, 1))*180/np.pi, label='1/4')
#plt.plot(np.angle(np.mean(array09, 1))*180/np.pi, label='1/6')
#plt.plot(np.angle(np.mean(array18, 1))*180/np.pi, label='5/4')
#plt.plot(np.angle(np.mean(array19, 1))*180/np.pi, label='5/6')
#plt.plot(np.angle(np.mean(array89, 1))*180/np.pi, label='4/6')
#plt.legend()
#plt.subplot(212)
#plt.plot(10.*np.log10(np.abs(np.mean(array01, 1))), label='1/5')
#plt.plot(10.*np.log10(np.abs(np.mean(array08, 1))), label='1/4')
#plt.plot(10.*np.log10(np.abs(np.mean(array09, 1))), label='1/6')
#plt.plot(10.*np.log10(np.abs(np.mean(array18, 1))), label='5/4')
#plt.plot(10.*np.log10(np.abs(np.mean(array19, 1))), label='5/6')
#plt.plot(10.*np.log10(np.abs(np.mean(array89, 1))), label='4/6')
#plt.show()