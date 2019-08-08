#!/usr/bin/env python
'''
###############################################################################
#                                     _--__
#                                --- (     )
#                              /     / @    \
#                             /      &__/    |
#                            |           |   |
#                            |       /   |   |
#                           K  e  r  m  o  d  e
#
###############################################################################
#
#  Project     : Kermode Small Beamformer
#  File        : ACMCapture.py
#  Description : Capture small beamformer + HCTF data 
#  Author(s)   : Stephen Harrison
#
#  Copyright (c) National Research Council of Canada 2018
#
###############################################################################
'''

import sys
import zmq
import numpy as np
from astropy.time import Time
from scipy.io import savemat
import os
from datetime import datetime
import logging
from threading import Thread, Lock, Event
import time
from copy import copy

class ACMCapture(Thread):

    def __init__(self, host, size=16, nch=512):
        
        super(ACMCapture, self).__init__()

        self._host = host
        self._topic = 'ACM:'
        self._size = size
        self._nch = nch

        # HCTF State variables.
        self._hctfUtc = None
        self._state = None
        self._t1c = None
        self._t2c = None

        # Collection variables.
        self._N = None
        self._filename = None
        self._folder = None
        self._out = None
        self._tempOut = None
        self._collectables = None

        # State variable lock.
        self._stateLock = Lock()

        # Run flag. Initially set fals.
        self._run = Event()

        # Dumps complete flag.
        self._capture = Event()

        # Start ourselves.
        # Is this a good thing? I dont know.
        self.start() 

    def putHCTFstate(self, roofStateStr):
        """Put the HCTF state in an atomic manner."""

        utc, state, t1c, t2c = roofStateStr.split(',')
        
        if state not in ['open', 'closing', 'closed', 'opening']:
            raise ValueError('Invalid roof state: %s.' %(state))
        
        t1c = float(t1c) # Will raise ValueError if not convertible
        t2c = float(t2c) # Will raise ValueError if not convertible

        self._stateLock.acquire() # Block until lock is acquired.
        self._hctfUtc   = utc
        self._state     = state
        self._t1c       = t1c
        self._t2c       = t2c
        self._stateLock.release()

    def getHCTFstate(self):
        """Get the latest HCTF state in an atomic manner."""        

        self._stateLock.acquire() # Block until lock is acquired.
        hctfUtc = copy(self._hctfUtc)
        state   = copy(self._state)
        t1c     = copy(self._t1c)
        t2c     = copy(self._t2c)
        self._stateLock.release()

        return [hctfUtc, state, t1c, t2c]

    def chkDone(self):

        return not self._capture.isSet()

    def stop(self, timeout=10):
        """Returns false if the thread doesn't join in the timeout window."""
        self._run.clear()
        self.join(timeout)
        return not self.isAlive()

    def collectNdumps(self, n, filename, folder='.'):

        # Throw an error if you try to start a new dump.
        # Alternatively could just block until the current dump is done.
        if not self.chkDone():
            raise RuntimeError('Previous collection was not complete.')

        self._N = n

        self._filename = filename
        self._folder = folder

        self._capture.set()

    def getNdumps(self):

        if np.shape(self._out)[2] == 0 and not self._capture.isSet():
            return self._N

        return np.shape(self._out)[2]

    def run(self):

        self._run.set()

        # ZMQ stuff.
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, self._topic)
        socket.connect("tcp://%s" %(self._host))

        # ACM holding area
        acm = np.zeros((self._size, self._size, 1, self._nch)).astype(np.complex)

        # Output stuff.
        self._out = np.zeros((self._size, self._size, 0, self._nch)).astype(np.complex)
        tsList        = []
        hctfTsList    = []
        roofStateList = []
        t1cList       = []
        t2cList       = []
        firstTs       = None
        ts            = None

        while self._run.isSet():

            string = ''

            while string == '' and self._run.isSet():

                try:
                    string = socket.recv(flags=zmq.NOBLOCK)

                    topic, sts, messagedata = string.split(' ', 2)
                    dummy, row, col = topic.split(':')
                    row = int(row)
                    col = int(col)
                    ts = Time(sts)

                    if ts is None:
                        ts  = Time(sts)

                    if firstTs is None:
                        firstTs = ts

                except zmq.ZMQError:
                    time.sleep(0.5)

            # With this method the first dump we captured *could be* 
            # incomplete so maybe dont start dumping right away? 
            if ts > firstTs:

                if self._capture.isSet():
                    hctfTs, roofState, t1c, t2c = self.getHCTFstate()
                    tsList.append(sts)
                    self._out = np.append(self._out, acm, 2)
                    hctfTsList.append(hctfTs)
                    roofStateList.append(roofState)
                    t1cList.append(t1c)
                    t2cList.append(t2c)

#                    print('%s -- %d' %(ts, np.shape(out)[2]))

                firstTs = ts
                acm = np.zeros((self._size,self._size,1,self._nch)).astype(np.complex)

            if self._run.isSet():
                ydata = np.fromstring(messagedata, dtype=np.complex)[::-1]
                acm[row,col,0,:] = ydata
                acm[col,row,0,:] = np.conjugate(ydata) # NICK - doesn't this overwrite?

            if np.shape(self._out)[2] == self._N:

                # Cut out empty channels.
                # PLAN:
                # - Find channels where the absolutel value of the max is not 0 (write to collectables)
                # - Write all of the data where both of the indices are in collectables
                # to a tempFile
                # - Overwrite self._out with the temp file

                # Identify empty channels
                for (i=1:self._size):
                    for (dump=1:len(self._out[1,1,:,1]):
                        if (np.absolute(self._out[i,i,dump,:]).amax()!=0):
                            print('Data at [%s, %s, %s, :]', row, col, dump)
                            self._collectables = self._collectables.append(i)

                # Initialize the temp file
                self._tempOut = np.zeros((len(collectables),len(collectables),len(self._out[1,1,:,1]),self._nch)).astype(np.complex)

                # I guess I'll try and be verbose
                tempIndex = 0;

                # Write data where both indices are in collectabes to temp
                for (i in collectabes):
                    for (j in collectables):
                        self._tempOut = self._out[i,j,:,:]
                    print('BF index %s is pafACMs index %s',i,tempIndex)
                    tempIndex++

                # Hi my name is %s and I'm verbose
                print('Number of channels where data was saved: %s',len(self._tempOut[:,1,1,1]))

                # Write to self._out
                self._out = self._tempOut

                # Write the matlab file
                mdict = {'pafACMs': self._out,
                         'utc': tsList,
                         'hctfUtc' : hctfTsList,
                         'roofState' : roofStateList,
                         't1c' : t1cList,
                         't2c' : t2cList}
                filename = os.path.join(self._folder, self._filename)
                #print('Saving %s.mat' %(filename))
                savemat(filename, mdict, do_compression=True)

                # Reset the accumulation variables.
                self._out = np.zeros((self._size, self._size, 0, self._nch)).astype(np.complex)
                tsList        = []
                hctfTsList    = []
                roofStateList = []
                t1cList       = []
                t2cList       = []
                firstTs       = None
                ts            = None

                # Drop the flag.
                self._capture.clear()

        # If asked to shutdown before completion.
        # If we have data (shape[2] != 0) then write the partial file out.
        if np.shape(self._out)[2] > 0:
            
            # Write the matlab file
            mdict = {'pafACMs': self._out,
                     'utc': tsList,
                     'hctfUtc' : hctfTsList,
                     'roofState' : roofStateList,
                     't1c' : t1cList,
                     't2c' : t2cList}
            filename = os.path.join(self._folder, self._filename)
            #print('Saving %s.mat' %(filename))
            savemat(filename, mdict, do_compression=True)

        # Drop the flag.
        self._capture.clear()

        # Close the ZMQ socket.
        socket.close()


if __name__ == '__main__':

    print('Create instance.')
    c = ACMCapture('192.168.100.190:5556')

    print('Thread is alive: %s.' %(c.isAlive()))

    print('Set initial HCTF state.')
    c.putHCTFstate('testUTC,closed,31,32.3')

    print('Thread is done/ready: %s.' %(c.chkDone()))

    print('Collect N dumps.')
    c.collectNdumps(5, 'test')
    while not c.chkDone():
        time.sleep(.5)
        c.putHCTFstate('testUTC,closed,%d,32.2' %(c.getNdumps()))
        print('Thread is done/ready: %s (%d).' %(c.chkDone(), c.getNdumps()))

    for i in range(5):
        time.sleep(.5)
        c.putHCTFstate('testUTC,closed,%d,32.2' %(c.getNdumps()))
        print('Thread is done/ready: %s (%d).' %(c.chkDone(), c.getNdumps()))

    print('Collect N dumps.')
    c.collectNdumps(5, 'test')
    while not c.chkDone():
        time.sleep(.5)
        c.putHCTFstate('testUTC,closed,%d,32.2' %(c.getNdumps()))
        print('Thread is done/ready: %s (%d).' %(c.chkDone(), c.getNdumps()))

    for i in range(5):
        time.sleep(.5)
        c.putHCTFstate('testUTC,closed,%d,32.2' %(c.getNdumps()))
        print('Thread is done/ready: %s (%d).' %(c.chkDone(), c.getNdumps()))

    print('Stop the thread.')
    if not c.stop():
        print('Thread did not stop.')

    print('Thread is alive: %s.' %(c.isAlive()))


##
## END OF CODE
##
