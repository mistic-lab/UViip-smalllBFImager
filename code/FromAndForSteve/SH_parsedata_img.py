import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
CHANS = [0,1,8,9]
chan = 115
string = 'dummy'

array01 = np.zeros(0).astype(np.complex)
array08 = np.zeros(0).astype(np.complex)
array09 = np.zeros(0).astype(np.complex)
array18 = np.zeros(0).astype(np.complex)
array19 = np.zeros(0).astype(np.complex)
array89 = np.zeros(0).astype(np.complex)

with open(filename, 'rb') as f:

    string = f.read(8229)

    while (len(string) == 8229):

        topic, sts, messagedata = string.split(' ', 2)

        dummy, row, col = topic.split(':')
        row = int(row)
        col = int(col)

        data = np.fromstring(messagedata, dtype=np.complex)[::-1]

        if (row == 0):
            if (col == 1):
                array01 = np.append(array01, data[chan])
            if (col == 8):
                array08 = np.append(array08, data[chan])
            if (col == 9):
                array09 = np.append(array09, data[chan])
        if (row == 1):
            if (col == 8):
                array18 = np.append(array18, data[chan])
            if (col == 9):
                array19 = np.append(array19, data[chan])
        if (row == 8):
            if (col == 9):
                array89 = np.append(array89, data[chan])



        string = f.read(8229)

for i in np.arange(65,80):
    img = np.zeros((9,3)).astype(np.complex)

    img[3, 1] = array18[i]
    img[5, 1] = np.conjugate(array18[i])

    img[1, 1] = np.conjugate(array08[i])
    img[7, 1] = array08[i]

    img[0, 1] = np.conjugate(array01[i])
    img[8, 1] = array01[i]

    img[5, 0] = array09[i]
    img[3, 2] = np.conjugate(array09[i]) 

    img[2, 0] = array89[i]
    img[6, 2] = np.conjugate(array89[i])

    img[1, 0] = array19[i]
    img[7, 2] = np.conjugate(array19[i])

    imginv = np.fft.ifft2(img)

    plt.figure()
    plt.imshow(np.abs(imginv), cmap='gray')
    plt.title(str(i))
    plt.show()

