import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal_3d(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def label_axes(ax):
    '''Label x,y,z axes with default labels'''
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def vis_scene(camL,camR,pts3,looklength=20):
  '''Visualize a stereo scene reconstruction

  Parameters
  ----------
    camL,camR : Camera
    pts3 : 2D numpy.array (dtype=float) of shape (3,N)

  '''

  # generate coordinates of a line segment running from the center
  # of the camera to 3 units in front of the camera
  lookL = np.hstack((camL.t,camL.t+camL.R @ np.array([[0,0,looklength]]).T))
  lookR = np.hstack((camR.t,camR.t+camR.R @ np.array([[0,0,looklength]]).T))

  # 3D plot showing cameras and points
  fig = plt.figure()
  ax = fig.add_subplot(2,2,1,projection='3d')
  ax.plot(pts3[0,:],pts3[1,:],pts3[2,:],'.')
  ax.plot(camR.t[0],camR.t[1],camR.t[2],'ro')
  ax.plot(camL.t[0],camL.t[1],camL.t[2],'bo')
  ax.plot(lookL[0,:],lookL[1,:],lookL[2,:],'b')
  ax.plot(lookR[0,:],lookR[1,:],lookR[2,:],'r')
  set_axes_equal_3d(ax)
  label_axes(ax)
  plt.title('scene 3D view')

  # 2D view of the XZ axis
  ax = fig.add_subplot(2,2,2)
  ax.plot(pts3[0,:],pts3[2,:],'.')
  ax.plot(camR.t[0],camR.t[2],'ro')
  ax.plot(camL.t[0],camL.t[2],'bo')
  ax.plot(lookL[0,:],lookL[2,:],'b')
  ax.plot(lookR[0,:],lookR[2,:],'r')
  plt.title('XZ-view')
  plt.grid()
  plt.xlabel('x')
  plt.ylabel('z')

  # 2D view of the YZ axis
  ax = fig.add_subplot(2,2,3)
  ax.plot(pts3[1,:],pts3[2,:],'.')
  ax.plot(camR.t[1],camR.t[2],'ro')
  ax.plot(camL.t[1],camL.t[2],'bo')
  ax.plot(lookL[1,:],lookL[2,:],'b')
  ax.plot(lookR[1,:],lookR[2,:],'r')
  plt.title('YZ-view')
  plt.grid()
  plt.xlabel('y')
  plt.ylabel('z')

  # 2D view of the XY axis
  ax = fig.add_subplot(2,2,4)
  ax.plot(pts3[0,:],pts3[1,:],'.')
  ax.plot(camR.t[0],camR.t[1],'ro')
  ax.plot(camL.t[0],camL.t[1],'bo')
  ax.plot(lookL[0,:],lookL[1,:],'b')
  ax.plot(lookR[0,:],lookR[1,:],'r')
  ax.invert_yaxis()
  plt.title('XY-view')
  plt.grid()
  plt.xlabel('x')
  plt.ylabel('y')
