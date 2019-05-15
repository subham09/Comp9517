import numpy as np
import cv2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')


print 'loading images...'
imgL = cv2.pyrDown(cv2.imread('image_left.png')) # downscale images for faster processing
imgR = cv2.pyrDown(cv2.imread('image_right.png'))

# disparity range is adjusted for bike image
window_size = 2
min_disp = 16
num_disp = 112-min_disp
stereo = cv2.StereoSGBM(minDisparity = min_disp, 
    numDisparities = num_disp, 
    SADWindowSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 1,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    fullDP = False
)

print 'computing disparity...'
disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

print 'generating 3d point cloud...',
h, w = imgL.shape[:2]
f = 3997.684                          #focal length
Q = np.float32([[1, 0, 0, -0.5*w],
                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis, 
                [0, 0, 0,     -f], # so that y-axis looks up
                [0, 0, 1,      0]])
points = cv2.reprojectImageTo3D(disp, Q)
colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
mask = disp > disp.min()
out_points = points[mask]
out_colors = colors[mask]
out_fn = '3D_image.ply'
write_ply('3D_image.ply', out_points, out_colors)
print '%s saved' % '3D_image.ply'
img = (disp-min_disp)/num_disp
z = points[:,:,2]
##    x = points[2,:,:]
##    y = points[:,2,:]
##    print(len(z[0]),len(y[0]))
yax = []
yaxis = []
##xax = []
##    xaxis = []
dist = []
distance = []
for i in z:
    dist = []
    for j in i:
        dist.append(abs(j))
    distance.append(dist)
##    for i in x:
##	xax = []
##        for j in i:
##            xax.append(abs(j))
##	xaxis.append(xax)
for i in points:
    yax = []
    for j in i:
        yax.append(abs(j[1]))
    yaxis.append(yax)
new = []
news = []
min_distance = []
for m,i in enumerate(distance):
    new = []
    for n,j in enumerate(i):
        if yaxis[m][n] > 3.7:
            new.append([255,255,255])
            continue
        elif j < 50:
            new.append([0,0,0])
            min_distance.append(j)
        else:
            new.append([255,255,255])
    news.append(new)
min_dist = min(min_distance)
print '\nThere is an obstacle in',min_dist,'CM\n'
dt = np.dtype('f8')
img2 = np.array(news,dtype=dt)                
cv2.imwrite('obstacle.png',img2)
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", img)
cv2.waitKey()
cv2.destroyAllWindows()

