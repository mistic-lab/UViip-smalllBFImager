from scipy.io import savemat


mdict = {'Config': 'parallel',
         'timeStep': 0.4,
         'BFIndexes': [0, 8, 1, 9],
         'Antennas': [2, 6, 7, 8],
         'arr00' : arr00,
         'arr01' : arr01,
         'arr08' : arr08,
         'arr09' : arr09,
         'arr11' : arr11,
         'arr18' : arr18,
         'arr19' : arr19,
         'arr88' : arr88,
         'arr89' : arr89,
         'arr99' : arr99}
filename = 'DRAO_CW_3700KHZ'
savemat(filename, mdict, do_compression=True)
