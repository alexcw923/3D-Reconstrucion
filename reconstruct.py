import numpy as np
import matplotlib.pyplot as plt
from recontools.camutils import Camera,triangulate,calibratePose,makerotation,reconstruct,mesh
import pickle, cv2
import recontools.visutils as visutils
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from recontools.meshutils import writeply


def writeFile(bls, camL, camR):
    """
    Write the pts and other information to .ply files with specified box limits.

    Parameters
    ----------
    
    bls : list[numpy.array(6)]
      list of boxlimits

    camL,camR : Camera
      camera parameters

    Returns
    -------
    None

    """
    for i in range(7):
        print(f'Writing teapot {i}')
        path=f'teapot/grab_{i}_u/'
        trithresh=0.3
        imprefixC0 = path+'frame_C0_'
        imprefixColorL = path+'color_C0_'
        imprefixC1 = path+'frame_C1_'
        imprefixColorR = path+'color_C1_'

        threshold,color_thresh = 0.02,0.13

        pts2L,pts2R,pts3,rgb = reconstruct(imprefixC0,imprefixColorL,imprefixC1,imprefixColorR,threshold,color_thresh,camL,camR)

        boxlimits = bls[i]
        pts3, tri, rgb = mesh(pts2L,pts2R,pts3,boxlimits,trithresh,rgb)
        writeply(pts3,rgb,tri,f'obj/teapot{i}.ply')

def execute():

    # load in the intrinsic camera parameters from 'calibration.pickle'
    file = open('calibration.pickle','rb')
    d=pickle.load(file)
    f=(d['fx']+d['fy'])/2
    c=np.array([[d['cx']],[d['cy']]])
    R=makerotation(0,0,0)
    t=np.array([[0],[0],[0]])

    # create Camera objects representing the left and right cameras
    # use the known intrinsic parameters you loaded in.
    camL = Camera(f,c,R,t)
    camR = Camera(f,c,R,t)

    # load in the left and right images and find the coordinates of
    # the chessboard corners using OpenCV
    imgL = plt.imread('calib_jpg_u/frame_C0_01.jpg')
    ret, cornersL = cv2.findChessboardCorners(imgL, (8,6), None)
    pts2L = cornersL.squeeze().T

    imgR = plt.imread('calib_jpg_u/frame_C1_01.jpg')
    ret, cornersR = cv2.findChessboardCorners(imgR, (8,6), None)
    pts2R = cornersR.squeeze().T

    # generate the known 3D point coordinates of points on the checkerboard in cm
    pts3 = np.zeros((3,6*8))
    yy,xx = np.meshgrid(np.arange(8),np.arange(6))
    pts3[0,:] = 2.8*xx.reshape(1,-1)
    pts3[1,:] = 2.8*yy.reshape(1,-1)


    # Now use your calibratePose function to get the extrinsic parameters
    # for the two images. You may need to experiment with the initialization
    # in order to get a good result

    paramsL = np.array([0,-1,1,0,0,-1])
    paramsR = np.array([0,1,1,0,0,-1])

    camL = calibratePose(pts3,pts2L,camL,paramsL)
    camR = calibratePose(pts3,pts2R,camR,paramsR)

    # As a final test, triangulate the corners of the checkerboard to get back there 3D locations
    pts3r = triangulate(pts2L,camL,pts2R,camR)

    bls = [np.array([-1,19.25,3,22,16,26.5]),np.array([-1,19,5,18,18,28]),np.array([-1,19.25,2,21,19,25.5]),
        np.array([-1,19.25,6,18,15,25]),np.array([-1,19,4,22,15,25]),np.array([7,19.25,5,23,10,25.5]),
        np.array([6,17.75,7.5,22,17.5,25])]
    #Meshing the images by looping through each box limit and file path
    for i in range(7):
        path=f'teapot/grab_{i}_u/'
        
        imprefixC0 = path+'frame_C0_'
        imprefixColorL = path+'color_C0_'
        imprefixC1 = path+'frame_C1_'
        imprefixColorR = path+'color_C1_'

        trithresh,threshold,color_thresh = 0.25,0.02,0.13

        pts2L,pts2R,pts3,rgb = reconstruct(imprefixC0,imprefixColorL,imprefixC1,imprefixColorR,threshold,color_thresh,camL,camR)

        boxlimits = bls[i]
        pts3, tri, rgb = mesh(pts2L,pts2R,pts3,boxlimits,trithresh,rgb)
        visutils.vis_scene(camL,camR,pts3,looklength=10)


    #write to ply files
    writeFile(bls, camL, camR)



if __name__ == '__main__':
    execute()