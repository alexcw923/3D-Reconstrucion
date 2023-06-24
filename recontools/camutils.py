import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def makerotation(rx,ry,rz):
    """
    Generate a rotation matrix    

    Parameters
    ----------
    rx,ry,rz : floats
        Amount to rotate around x, y and z axes in degrees

    Returns
    -------
    R : 2D numpy.array (dtype=float)
        Rotation matrix of shape (3,3)
    """
    rx = np.pi*rx/180.0
    ry = np.pi*ry/180.0
    rz = np.pi*rz/180.0

    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,-np.sin(ry)],[0,1,0],[np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    R = (Rz @ Ry @ Rx)
    
    return R 

class Camera:
    """
    A simple data structure describing camera parameters 
    
    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation 

    
    """    
    def __init__(self,f,c,R,t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'
    
    def project(self,pts3):
        """
        Project the given 3D points in world coordinates into the specified camera    

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        Returns
        -------
        pts2 : 2D numpy.array (dtype=float)
            Image coordinates of N points stored in an array of shape (2,N)

        """
        assert(pts3.shape[0]==3)

        # get point location relative to camera
        pcam = self.R.transpose() @ (pts3 - self.t)
         
        # project
        p = self.f * (pcam / pcam[2,:])
        
        # offset principal point
        pts2 = p[0:2,:] + self.c
        
        assert(pts2.shape[1]==pts3.shape[1])
        assert(pts2.shape[0]==2)
    
        return pts2
 
    def update_extrinsics(self,params):
        """
        Given a vector of extrinsic parameters, update the camera
        to use the provided parameters.
  
        Parameters
        ----------
        params : 1D numpy.array (dtype=float)
            Camera parameters we are optimizing over stored in a vector
            params[0:2] are the rotation angles, params[2:5] are the translation

        """
        self.R = makerotation(params[0],params[1],params[2])
        self.t = np.array([[params[3]],[params[4]],[params[5]]])


def triangulate(pts2L,camL,pts2R,camR):
    """
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coordinates relative
    to the global coordinate system


    Parameters
    ----------
    pts2L : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camL camera

    pts2R : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camR camera

    camL : Camera
        The first "left" camera view

    camR : Camera
        The second "right" camera view

    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
        (3,N) array containing 3D coordinates of the points in global coordinates

    """

    npts = pts2L.shape[1]

    qL = (pts2L - camL.c) / camL.f
    qL = np.vstack((qL,np.ones((1,npts))))

    qR = (pts2R - camR.c) / camR.f
    qR = np.vstack((qR,np.ones((1,npts))))
    
    R = camL.R.T @ camR.R
    t = camL.R.T @ (camR.t-camL.t)

    xL = np.zeros((3,npts))
    xR = np.zeros((3,npts))

    for i in range(npts):
        A = np.vstack((qL[:,i],-R @ qR[:,i])).T
        z,_,_,_ = np.linalg.lstsq(A,t,rcond=None)
        xL[:,i] = z[0]*qL[:,i]
        xR[:,i] = z[1]*qR[:,i]
 
    pts3L = camL.R @ xL + camL.t
    pts3R = camR.R @ xR + camR.t
    pts3 = 0.5*(pts3L+pts3R)

    return pts3


def residuals(pts3,pts2,cam,params):
    """
    Compute the difference between the projection of 3D points by the camera
    with the given parameters and the observed 2D locations

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    params : 1D numpy.array (dtype=float)
        Camera parameters we are optimizing over stored in a vector

    Returns
    -------
    residual : 1D numpy.array (dtype=float)
        Vector of residual 2D projection errors of size 2*N
        
    """

    cam.update_extrinsics(params)
    residual = pts2 - cam.project(pts3)
    
    return residual.flatten()

def calibratePose(pts3,pts2,cam_init,params_init):
    """
    Calibrate the provided camera by updating R,t so that pts3 projects
    as close as possible to pts2

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    cam : Camera
        Initial estimate of camera

    Returns
    -------
    cam_opt : Camera
        Refined estimate of camera with updated R,t parameters
        
    """

    # define our error function
    efun = lambda params: residuals(pts3,pts2,cam_init,params)        
    popt,_ = scipy.optimize.leastsq(efun,params_init)
    cam_init.update_extrinsics(popt)

    return cam_init



def decode(imprefix,start,threshold):
    """
    Decode 10bit gray code pattern with the given difference
    threshold.  We assume the images come in consective pairs
    with filenames of the form <prefix><start>.png - <prefix><start+20>.png
    (e.g. a start offset of 20 would yield image20.png, image01.png... image39.png)

    Parameters
    ----------
    imprefix : str
      prefix of where to find the images (assumed to be .png)

    start : int
      image offset.  

    threshold : float

    Returns
    -------
    code : 2D numpy.array (dtype=float)
        
    mask : 2D numpy.array (dtype=float)
    
    
    """
    nbits = 10
    
    imgs = list()
    imgs_inv = list()
    print('loading',end='')
    for i in range(start,start+2*nbits,2):
        fname0 = '%s%2.2d.png' % (imprefix,i)
        fname1 = '%s%2.2d.png' % (imprefix,i+1)
        print('(',i,i+1,')',end='')
        img = plt.imread(fname0)
        img_inv = plt.imread(fname1)
        if (img.dtype == np.uint8):
            img = img.astype(float) / 256
            img_inv = img_inv.astype(float) / 256
        if (len(img.shape)>2):
            img = np.mean(img,axis=2)
            img_inv = np.mean(img_inv,axis=2)
        imgs.append(img)
        imgs_inv.append(img_inv)
        
    (h,w) = imgs[0].shape
    print('\n')
    
    gcd = np.zeros((h,w,nbits))
    mask = np.ones((h,w))
    for i in range(nbits):
        gcd[:,:,i] = imgs[i]>imgs_inv[i]
        mask = mask * (np.abs(imgs[i]-imgs_inv[i])>threshold)
        
    bcd = np.zeros((h,w,nbits))
    bcd[:,:,0] = gcd[:,:,0]
    for i in range(1,nbits):
        bcd[:,:,i] = np.logical_xor(bcd[:,:,i-1],gcd[:,:,i])
        
    code = np.zeros((h,w))
    for i in range(nbits):
        code = code + np.power(2,(nbits-i-1))*bcd[:,:,i]
        
    return code,mask


def colorMask(background_path,obj_path,thresh):
    """
    Get color mask based on given paths and threshold.

    Parameters
    ----------
    background_path : str
      path for where the ground truth images are stored
      
    obj_path : str
      path for where the object images are stored

    thresh : float
      color threshold

    Returns
    -------

    mask : 2D numpy.array (dtype=float)

    """
    obj_img,back_img=plt.imread(obj_path),plt.imread(background_path)
    (h,w)= obj_img.shape[:2]
    mask = np.ones((h,w))
    mask = mask*((np.sum(np.abs(obj_img-back_img), axis=-1))>thresh)
    return mask
    

def reconstruct(imprefixL,imprefixColorL,imprefixR,imprefixColorR,threshold,colorthresh,camL,camR):
    """
    Simple reconstruction based on triangulating matched pairs of points
    between to view which have been encoded with a 20bit gray code.

    Parameters
    ----------
    imprefix : str
      prefix for where the images are stored

    threshold : float
      decodability threshold

    camL,camR : Camera
      camera parameters

    Returns
    -------
    pts2L,pts2R : 2D numpy.array (dtype=float)

    pts3 : 2D numpy.array (dtype=float)
    
    rgb_values : (3,n) numpy.array (dtype=float)

    """
    color_maskL=colorMask('%s%2.2d.png'%(imprefixColorL,0),'%s%2.2d.png'%(imprefixColorL,1),colorthresh)
    color_maskR=colorMask('%s%2.2d.png'%(imprefixColorR,0),'%s%2.2d.png'%(imprefixColorR,1),colorthresh)
    CLh,maskLh = decode(imprefixL,0,threshold)
    CLv,maskLv = decode(imprefixL,20,threshold)
    CRh,maskRh = decode(imprefixR,0,threshold)
    CRv,maskRv = decode(imprefixR,20,threshold)

    CL = CLh + 1024*CLv
    maskL = maskLh*maskLv*color_maskL
    CR = CRh + 1024*CRv
    maskR = maskRh*maskRv*color_maskR

    h = CR.shape[0]
    w = CR.shape[1]

    subR = np.nonzero(maskR.flatten())
    subL = np.nonzero(maskL.flatten())

    CRgood = CR.flatten()[subR]
    CLgood = CL.flatten()[subL]

    _,submatchR,submatchL = np.intersect1d(CRgood,CLgood,return_indices=True)

    matchR = subR[0][submatchR]
    matchL = subL[0][submatchL]

    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))

    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)
    
    num_pts = pts2L.shape[1]
    imageR=plt.imread('%s%2.2d.png'%(imprefixColorR,1))
    imageL=plt.imread('%s%2.2d.png'%(imprefixColorL,1))
    rgb_valuesR = rgb_valuesL = np.zeros((3,num_pts))
  
    for i in range(num_pts):
        xR,yR=int(pts2R[0,i]),int(pts2R[1,i])
        xL,yL=int(pts2L[0,i]),int(pts2L[1,i])
        rgb_valuesR[:,i]=imageR[yR,xR,:].T
        rgb_valuesL[:,i]=imageL[yL,xL,:].T
    rgb_values = (rgb_valuesR+rgb_valuesL)/2
    
    pts3 = triangulate(pts2L,camL,pts2R,camR)
    
    
    return pts2L,pts2R,pts3,rgb_values

def findNeighbors(vertex_id, tri):
    """
    Find neighbors of given index and triangle object

    Parameters
    ----------
    vertex_id : int
      the index of the vertex to find its neighbors

    tri : scipy.spatial.Delaunay
      Delaunay object created with triangulation


    Returns
    -------
    res : 1D numpy.array (dtype=int)

    """
    idx, indices = tri.vertex_neighbor_vertices
    res = indices[idx[vertex_id]:idx[vertex_id + 1]]
    return res
    
def mesh(pts2L,pts2R,pts3,boxlimits,trithresh,rgb):
    """
    Simple reconstruction based on triangulating matched pairs of points
    between to view which have been encoded with a 20bit gray code.

    Parameters
    ----------
    pts2L,pts2R : 2D numpy.array (dtype=float)
      2D points from left and right views
    
    pts3 : 2D numpy.array (dtype=float)
      3D points reconstructed from pts2L and pts2R
      
    trithresh : float
      triangulation threshold

    boxlimits : 1D numpy.array (dtype=float)
      mesh cleaning boundary with [-x,x,-y,y,-z,z]
    
    rgb : (3,n) 2D numpy.array (dtype=float)
    
    
    Returns
    -------
    
    pts3 : 2D numpy.array (dtype=float)
    
    tri : (n,3) 2D numpy.array (dtype=float)
    
    rgb : (3,n) 2D numpy.array (dtype=float)
    
    """
    in_bound = np.nonzero((pts3[0,:]>boxlimits[0])&(pts3[0,:]<boxlimits[1])&
                      (pts3[1,:]>boxlimits[2])&(pts3[1,:]<boxlimits[3])&
                      (pts3[2,:]>boxlimits[4])&(pts3[2,:]<boxlimits[5]))
    pts3 = pts3[:,in_bound[0]]
    pts2L = pts2L[:,in_bound[0]]  
    pts2R = pts2R[:,in_bound[0]]
    rgb = rgb[:,in_bound[0]]
    #
    # triangulate the 2D points to get the surface mesh
    #
    triangles = Delaunay(pts2L.T)
    tri = triangles.simplices
    for _ in range(3):
        for idx in range(pts3.shape[1]):
            pts3[:,idx]=np.mean(pts3[:,findNeighbors(idx,triangles)],axis=1)

    edge01 = np.sqrt(np.sum((pts3[:,tri[:,0]]-pts3[:,tri[:,1]])**2,axis=0))
    edge02 = np.sqrt(np.sum((pts3[:,tri[:,0]]-pts3[:,tri[:,2]])**2,axis=0))
    edge12 = np.sqrt(np.sum((pts3[:,tri[:,1]]-pts3[:,tri[:,2]])**2,axis=0))
    after_prune = (edge01<trithresh)&(edge02<trithresh)&(edge12<trithresh)
    tri = tri[after_prune,:]

    #
    # remove any points which are not referenced in any triangle
    #
    referenced_index=np.unique(tri)
    mapping = np.zeros(pts3.shape[1], dtype=int)
    pts3=pts3[:,referenced_index]
    rgb=rgb[:,referenced_index]
    
    mapping[referenced_index] = np.arange(0,referenced_index.shape[0])
    tri=mapping[tri]
    
    return pts3, tri, rgb
