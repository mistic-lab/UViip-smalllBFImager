import os

size = 8229
file = '/Users/nsbruce-school/Downloads/3700KHZ_CW.dat'
collectables = ['00', '01', '08', '09']

f = open(file, 'rb')
g = open('3700KHZ_CW', 'wb')

current = 0
while (current < os.path.getsize(file)):
    str = f.read(size)
    if any(ant in str[4:6] for ant in collectables):
        if any(ant in str[7:9] for ant in collectables):
            print str[4:9]
            g.write(str)
    current = current+size

f.close()
g.close()
