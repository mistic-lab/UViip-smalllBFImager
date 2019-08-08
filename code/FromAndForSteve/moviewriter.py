import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D

npts = 500
nant = 4
d    = 20 # meters
dpts = 25 # display points
t    = .4 # seconds

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=1/t*10, metadata=metadata) 

fig = plt.figure()
ax = fig.gca(projection='3d')


x = np.arange(npts)*t
y = np.arange(nant)*d

# Dummy up some z data with PHASE
z = np.zeros((npts, nant))
z[:,1] = 30*np.cos(2.0*np.pi*np.arange(npts)*.1)
z[:,2] = 60*np.cos(2.0*np.pi*np.arange(npts)*.1)
z[:,3] = 90*np.cos(2.0*np.pi*np.arange(npts)*.1)



with writer.saving(fig, "phase.mp4", 300):
    for i in range(npts-dpts):
        print i
        ax.clear()
        ax.grid(False)
        ax.set_zlim(-180,180)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (m)')
        ax.set_zlabel('PHASE!!!!!!!!!!')
        ax.plot_trisurf(np.repeat(x[i:i+dpts], 4), np.tile(np.arange(4), dpts), np.reshape(z[i:i+dpts,:], dpts*nant))
        writer.grab_frame()