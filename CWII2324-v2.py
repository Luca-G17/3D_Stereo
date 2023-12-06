'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse
import os.path
import sys


'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True


# print("here", flush=True)
if __name__ == '__main__': 

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6, 
                        help='number of spheres')    
    parser.add_argument('--sph_rad_min', dest='sph_rad_min', type=int, default=10, 
                        help='min sphere  radius x10')
    parser.add_argument('--sph_rad_max', dest='sph_rad_max', type=int, default=16, 
                        help='max sphere  radius x10')
    parser.add_argument('--sph_sep_min', dest='sph_sep_min', type=int, default=4, 
                       help='min sphere  separation')
    parser.add_argument('--sph_sep_max', dest='sph_sep_max', type=int, default=8, 
                       help='max sphere  separation')
    parser.add_argument('--coords', dest='bCoords', action='store_true')
    parser.add_argument('--camera_noise_t', dest='camera_noise_t', type=int, default=0)
    parser.add_argument('--camera_noise_r', dest='camera_noise_r', type=int, default=0)
    parser.add_argument('--iters', dest='iters', type=int, default=0)
    args = parser.parse_args()

    if args.num<=0:
        print('invalidnumber of spheres')
        exit()

    if args.sph_rad_min>=args.sph_rad_max or args.sph_rad_min<=0:
        print('invalid max and min sphere radii')
        exit()
    	
    if args.sph_sep_min>=args.sph_sep_max or args.sph_sep_min<=0:
        print('invalid max and min sphere separation')
        exit()
	
    #####################################################
    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.
    
    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width=640
    img_height=480
    f=415 # focal length
    # image centre in pixel coordinates
    ox=img_width/2-0.5 
    oy=img_height/2-0.5
    
    ###################################
    '''
    Task 3: Circle detection
    Hint: use cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    '''
    def HoughCircles(filename):
        if not(os.path.isfile(filename)):
            print("File not found")
            return
        image = cv2.imread(filename, 1)
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, 30, param1=70, param2=35, minRadius=10, maxRadius=60)
        circles = np.uint16(np.around(circles))
        if (len(circles) == 0):
            return
        for c in circles[0, :]:
            cv2.circle(image, (c[0], c[1]), c[2], (0, 255, 0), 2)
        cv2.imwrite(f"{filename.split('.png')[0]}HoughCircles.png", image)
        return circles
    ###################################


    ###################################
    '''
    Task 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here
    '''
    def CameraTransform(H0_wc, H1_wc):
        H_10 = np.matmul(H0_wc, np.linalg.inv(H1_wc))
        R = H_10[:3, :3].T
        T = H_10[:3, 3]
        return R, T

    def EpipolarLine(pt1, R, T, M, frame):
        S = np.array([
            [0, -T[2], T[1]],
            [T[2], 0, -T[0]],
            [-T[1], T[0], 0]
        ])

        E = np.matmul(R, S)
        F = np.matmul(np.matmul(M.T, E), M)

        u = np.matmul(F, pt1).reshape(3,)
        p0 = np.array([0, -f*u[2]/u[1]]).astype(int)
        p1 = np.array([img_width, -((f * u[2]) + (u[0] * img_width)) / u[1]]).astype(int)
        frame = cv2.line(frame, p0, p1, (255, 0, 0), 1)
        return [p0, p1]
        #cv2.imwrite(f"{filename.split('.png')[0]}ELine.png", frame)

    def DetectedCircleEpipolarLines(circles, R, T, M, filename):
        frame = cv2.imread(filename, 1)
        lines = []
        for c in circles[0, :]:
            pt1 = np.array([c[0], c[1], f])
            lines.append(EpipolarLine(pt1, R, T, M, frame))
        cv2.imwrite(f"{filename.split('.png')[0]}ELine.png", frame)
        return lines
    
    ###################################

    ###################################
    '''
    Task 5: Find correspondences

    Write your code here
    '''

    # ax + by + c = 0
    def LineFromPoints(p0, p1):
        a = p1[1] - p0[1]
        b = p0[0] - p1[0]
        c = (a * p0[0]) + (b * p0[1])
        return [a, b, -c]
    
    # Circles0 = Detected Circles from image 0
    # Circles1 = Detected Circles from image 1
    # Lines = Epipolar lines in image 1 from the centres of circles 0
    def Correspondences(circles0, circles1, lines):
        lines = [LineFromPoints(p0, p1) for [p0, p1] in lines]

        # Shortest distance between point and line:
        # d = |Ax + By + C| / sqrt(A^2 + B^2)
    
        # For each epipolar line build a list of circles that are closest to that line
        closest_circles = [[] for _ in lines]
        for c1 in circles1[0,:]:
            min_dist, min_l = sys.maxsize, -1
            for i in range(len(lines)):
                l = lines[i]
                dist = abs(l[0] * c1[0] + l[1] * c1[1] + l[2]) / math.sqrt(c1[0] ** 2 + c1[1] ** 2)
                if (dist < min_dist and dist < 150):
                    min_dist, min_l = dist, i
            if (min_l != -1):
                closest_circles[min_l].append([c1, min_dist])
        
        # For each epipolar line choose the circle with the minimum distance to it
        correspondences = []

        for l in range(len(closest_circles)):
            c = closest_circles[l]
            if (len(c) != 0):
                m = min(c, key=lambda x:x[1])
                correspondences.append([circles0[0,:][l], m[0]])
        return correspondences
    
    def DisplayCorrespondences(cs, filename0, filename1):
        frame0 = cv2.imread(filename0, 1)
        frame1 = cv2.imread(filename1, 1)
        colours = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (0, 152, 227),
            (227, 216, 0),
            (0, 227, 227),
            (191, 0, 169),
            (100, 0, 0),
            (0, 100, 0),
            (0, 0, 100)
        ]

        for i in range(len(cs)):
            circle0 = cs[i][0]
            circle1 = cs[i][1]
            if i == len(colours):
                print("NO MORE COLOURS")
            cv2.circle(frame0, (circle0[0], circle0[1]), circle0[2], colours[i], 2)
            cv2.circle(frame1, (circle1[0], circle1[1]), circle1[2], colours[i], 2)
        
        black_bar = np.zeros((img_height, 2, 3))
        horz = np.hstack((frame0, black_bar, frame1))
        cv2.imwrite("correspondences.png", horz)

    ###################################


    ###################################
    '''
    Task 6: 3-D locations of sphere centres

    Write your code here
    '''

    # M = Pixel to Image
    # R = Rotation applied to camera 1 to align it with camera 0
    # T = Translation from camera 0 to camera 1
    # cs = Correspondences in pixel coordindates (c0, c1)
    def SphereLocations(M, R, T, cs):
        # For each correspondence 
        # Form H = [p_l -R^Tp_r -p_lXR^Tp_r]
        PS = []
        for (c0, c1) in cs:
            c0 = np.array([c0[0], c0[1], f])
            c1 = np.array([c1[0], c1[1], f])

            # Convert to image coordinates
            p_l = np.matmul(M, c1.reshape(3, 1))
            p_r = np.matmul(M, c0.reshape(3, 1))

            H1 = np.matmul(R.T, p_r)
            H2 = np.cross(p_l, np.matmul(R.T, p_r), axis=0)
            H = np.hstack((p_l, H1, H2))

            [a, b, c] = np.matmul(np.linalg.inv(H), T)
            P = (((a * p_l) + (-b * H1) + T.reshape(3, 1)) / float(2))
            PS.append(P)
        return PS
    ###################################


    ###################################
    '''
    Task 7: Evaluate and Display the centres

    Write your code here
    '''

    def ImageToPixel(arr, M):
        Minv = np.linalg.inv(M)
        return np.array([np.matmul(Minv, (e * f / float(e[2])).reshape(3, 1)) for e in arr])
    
    def GroundTruthSphereLocations(H_list, H_wc, n):
        return np.array([transform_points(l[:3, 3].reshape(1,3), H_wc).reshape(3,) for l in H_list[-n:]])
    
    def DisplaySphereLocations(ground_truth, predicted, filename):
        frame = cv2.imread(filename, 1)
        radius = 5
        horz = np.array([[-radius,0],
                         [radius, 0]])
        vert = np.fliplr(horz)
        cross = np.vstack((horz, vert))
        for t in ground_truth:
            crossT = np.around([c + t[:2,0] for c in cross]).astype(int)
            cv2.line(frame, crossT[0], crossT[1], (0, 0, 255), 2)
            cv2.line(frame, crossT[2], crossT[3], (0, 0, 255), 2)

        for p in predicted:
            crossT = np.around([c + p[:2,0] for c in cross]).astype(int)
            cv2.line(frame, crossT[0], crossT[1], (0, 255, 0), 2)
            cv2.line(frame, crossT[2], crossT[3], (0, 255, 0), 2)

        cv2.imwrite("PredictedSphereLocations.png", frame)

    def EstimateErrors(ground_truth, ground_truth_p, predictions, GT_rads, radii, correspondences, verbose=True):
        # First Identify a matching between world spheres and hough circles in view1
        # The order of the hough circles never changes so the predicted sphere's will have the same indices as their corrosponding hough circle
        matches = []
        for t in range(len(ground_truth_p)):
            truth = ground_truth_p[t].T[0]
            min_dist, min_c = sys.maxsize, -1
            for i in range(len(correspondences)):
                c0, c1 = correspondences[i]
                # x, y, r
                dist = np.dot(c1[:2] - truth[:2], c1[:2] - truth[:2])
                if (dist < min_dist and dist < 150 ** 2):
                    min_dist, min_c = dist, i
            matches.append([t, min_c])
        
        # Error = Average distance between ground truth and its prediction
        total = 0
        total_radius_e = 0
        point_padding = 42
        float_padding = 20
        print()
        print("{:<{p}}| {:<{p}}| {:<{fp}}| {}".format("Prediction", "Ground Truth", "Centre Error", "Radius Error", p=point_padding, fp=float_padding))
        print("{:-<{p}}--{:-<{p}}--{:-<{p}}".format('','','', p=point_padding))
        for m in range(len(matches)):
            truth_i, hough_circle_i = matches[m]
            if (len(radii) <= hough_circle_i or len(predictions) <= hough_circle_i):
                continue

            prediction = predictions[hough_circle_i]
            truth = ground_truth[truth_i]
            radius_error = GT_rads[truth_i] - radii[hough_circle_i]
            dist = np.sqrt(np.dot(truth[:3] - prediction[:3], truth[:3] - prediction[:3]))
            print("{:<{p}}| {:<{p}}| {:.4f}{:<13} | {:.4f}{:<13}".format(str(prediction), str(truth), float(dist), '', float(radius_error), '', p=point_padding))
            total += dist
            total_radius_e += abs(radius_error)
        total = total / len(matches)
        total_radius_e = total / len(matches)
        print("{:-<{p}}--{:-<{p}}--{:-<{p}}".format('','','', p=point_padding))
        print("{:<{p}}| {:<{p}}| {:.4f}{:<14}| {:.4f}{:<14}".format('', 'Average', float(total), '', float(total_radius_e), '', p=point_padding))
        return (total, total_radius_e, len(matches))

    ###################################

    ###################################
    '''
    Task 8: 3-D radius of spheres

    Write your code here
    '''
    def EstimateSphereRadii(spheres, correspondences):
        new_spheres = []
        spheres = [s.T[0] for s in spheres]
        radii = []
        for i in range(len(spheres)):
            # Radius of circle in image
            r_image = correspondences[i][1][2]
            Z = spheres[i][2]
            R = Z * r_image / f
            radii.append(R)
        return spheres, radii
    ###################################


    ###################################
    '''
    Task 9: Display the spheres

    Write your code here:
    '''
    def CoordinatesToMeshes(Spheres):
        mesh_list_ext = []
        H_list_ext = []
        generated_meshes = []
        for i in range(len(Spheres)):
            centre, r = Spheres[i]
            if (r < 0):
                r = 1e-4
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=r)
            H = np.eye(4, 3)
            H = np.hstack((H, np.array([centre[0], centre[1], centre[2], 1]).reshape(4, 1)))
            H_list_ext.append(H)
            mesh_list_ext.append(mesh)
        
        for (mesh, H) in zip(mesh_list_ext, H_list_ext):
            mesh.vertices = o3d.utility.Vector3dVector(
                transform_points(np.asarray(mesh.vertices), H)
            )
            mesh.paint_uniform_color([0.0, 1.0, 0.0])
            mesh.compute_vertex_normals()
            generated_meshes.append(mesh)
        return generated_meshes

    def Generate3DSpheresFromCorrespondence(H0_wc, H1_wc, H1_wc_actual, H_list, GT_rads):
        M = np.array([
            [1, 0, -ox / f],
            [0, 1, -oy / f],
            [0, 0, 1]
        ])
        R, T = CameraTransform(H0_wc, H1_wc)
        reference_view = "view0.png"
        other_view = "view1.png"
        circles0 = HoughCircles(reference_view)
        circles1 = HoughCircles(other_view)
        if (len(circles0[:,0]) == 0):
            exit()
        lines = DetectedCircleEpipolarLines(circles0, R, T, M, other_view)
        correspondences = Correspondences(circles0, circles1, lines)
        DisplayCorrespondences(correspondences, reference_view, other_view)

        # Get Transform from H0 to H1
        R, T = CameraTransform(H1_wc, H0_wc)
        spheres = SphereLocations(M, R, T, correspondences)

        ground_truth = GroundTruthSphereLocations(H_list, H1_wc_actual, args.num)
        ground_truth_p = ImageToPixel(ground_truth, M)
        predicted_spheres_p = ImageToPixel(spheres, M)
        DisplaySphereLocations(ground_truth_p, predicted_spheres_p, other_view)

        spheres, radii = EstimateSphereRadii(spheres, correspondences)
        error = EstimateErrors(ground_truth, ground_truth_p, spheres, GT_rads, radii, correspondences)

        # Transform to world coordinates
        coords = transform_points(np.array(spheres), np.linalg.inv(H1_wc))
        spheres = list(zip(coords, radii))
        return spheres, CoordinatesToMeshes(spheres), error
    ###################################

    ###################################
    '''
    Task 10: Investigate impact of noise added to relative pose

    Write your code here:
    '''
    ###################################

    def Centres(original, generated):
        orignal_centres = o3d.geometry.PointCloud()
        orignal_centres.points = o3d.utility.Vector3dVector(np.array(original)[:, :3])
        orignal_centres.paint_uniform_color([1., 0., 0.])

        generated_centres = o3d.geometry.PointCloud()
        generated_centres.points = o3d.utility.Vector3dVector(np.array(generated))
        generated_centres.paint_uniform_color([0., 1., 0.])
        return [orignal_centres, generated_centres]


    def GenerateSpheres():
        # create plane to hold all spheres
        h, w = 24, 12
        # place the support plane on the x-z plane
        box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
        box_H=np.array(
                    [[1, 0, 0, -h/2],
                    [0, 1, 0, -0.05],
                    [0, 0, 1, -w/2],
                    [0, 0, 0, 1]]
                    )
        box_rgb = [0.7, 0.7, 0.7]
        name_list = ['plane']
        mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

        # create spheres
        prev_loc = []
        GT_cents, GT_rads = [], []
        for i in range(args.num):
            # add sphere name
            name_list.append(f'sphere_{i}')

            # create sphere with random radius
            size = random.randrange(args.sph_rad_min, args.sph_rad_max, 2)/10
            sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
            mesh_list.append(sph_mesh)
            RGB_list.append([0., 0.5, 0.5])

            # create random sphere location
            step = random.randrange(args.sph_sep_min,args.sph_sep_max,1)
            x = random.randrange(-h/2+2, h/2-2, step)
            z = random.randrange(-w/2+2, w/2-2, step)
            while check_dup_locations(x, z, prev_loc):
                x = random.randrange(-h/2+2, h/2-2, step)
                z = random.randrange(-w/2+2, w/2-2, step)
            prev_loc.append((x, z))

            GT_cents.append(np.array([x, size, z, 1.]))
            GT_rads.append(size)
            sph_H = np.array(
                        [[1, 0, 0, x],
                        [0, 1, 0, size],
                        [0, 0, 1, z],
                        [0, 0, 0, 1]]
                    )
            H_list.append(sph_H)
        # arrange plane and sphere in the space
        obj_meshes = []
        for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
            # apply location
            mesh.vertices = o3d.utility.Vector3dVector(
                transform_points(np.asarray(mesh.vertices), H)
            )
            # paint meshes in uniform colours here
            mesh.paint_uniform_color(rgb)
            mesh.compute_vertex_normals()
            obj_meshes.append(mesh)

        # add optional coordinate system
        if args.bCoords:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
            obj_meshes = obj_meshes+[coord_frame]
            RGB_list.append([1., 1., 1.])
            name_list.append('coords')
        return (obj_meshes, GT_cents, GT_rads, H_list)
    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    def RandomUniformWithExclusionInterval(delta, interval):
        (min, max) = interval
        n = random.uniform(-delta, delta)
        if (min == 0):
            return n
        while (n > max or n < min):
            n = random.uniform(-delta, delta)
        return n
    
    def RandomTranslation(delta, min):
        x_offset = RandomUniformWithExclusionInterval(delta, (-min, min))
        y_offset = RandomUniformWithExclusionInterval(delta, (-min, min))
        z_offset = RandomUniformWithExclusionInterval(delta, (-min, min))
        H = np.array([
            [1, 0, 0, x_offset],
            [0, 1, 0, y_offset],
            [0, 0, 1, z_offset],
            [0, 0, 0, 1]
        ])
        return H
    
    def RandomRotation(delta, min):
        t_x = (np.pi / 180) * RandomUniformWithExclusionInterval(delta, (-min, min))
        R_x = np.array([
            [1,             0,              0, 0],
            [0, math.cos(t_x), -math.sin(t_x), 0],
            [0, math.sin(t_x),  math.cos(t_x), 0],
            [0,             0,              0, 1]
        ])
        t_y = (np.pi / 180) * RandomUniformWithExclusionInterval(delta, (-min, min))
        R_y = np.array([
            [ math.cos(t_y), 0, math.sin(t_y), 0],
            [             0, 1,             0, 0],
            [-math.sin(t_y), 0, math.cos(t_y), 0],
            [             0, 0,             0, 1]
        ])
        t_z = (np.pi / 180) * RandomUniformWithExclusionInterval(delta, (-min, min))
        R_z = np.array([
            [math.cos(t_z), -math.sin(t_z), 0, 0],
            [math.sin(t_z),  math.cos(t_z), 0, 0],
            [            0,              0, 1, 0],
            [            0,              0, 0, 1]
        ])
        return np.matmul(np.matmul(R_x, R_y), R_z)

    def GenerateCameras():
        theta = np.pi * (45*5+random.uniform(-5, 5))/180.
        H0_wc = np.array(
                    [[1,            0,              0,  0],
                    [0, np.cos(theta), -np.sin(theta),  0], 
                    [0, np.sin(theta),  np.cos(theta), 20], 
                    [0, 0, 0, 1]]
                )

        # camera_1 (world to camera)
        theta = np.pi * (80+random.uniform(-10, 10))/180.
        H1_0 = np.array(
                    [[np.cos(theta),  0, np.sin(theta), 0],
                    [0,              1, 0,             0],
                    [-np.sin(theta), 0, np.cos(theta), 0],
                    [0, 0, 0, 1]]
                )
        theta = np.pi * (45*5+random.uniform(-5, 5))/180.
        H1_1 = np.array(
                    [[1, 0,            0,              0],
                    [0, np.cos(theta), -np.sin(theta), -4],
                    [0, np.sin(theta), np.cos(theta),  20],
                    [0, 0, 0, 1]]
                )
        H1_wc = np.matmul(H1_1, H1_0)

        return H0_wc, H1_wc

    def GenerateRotationCameraNoise(H0_wc, H1_wc):
        r = args.camera_noise_r
        H0_wc_actual = np.matmul(H0_wc, RandomRotation(r, r - 0.5))
        H1_wc_actual = np.matmul(H1_wc, RandomRotation(r, r - 0.5))
        return H0_wc_actual, H1_wc_actual
    
    def GenerateTranslationCameraNoise(H0_wc, H1_wc):
        t = args.camera_noise_t
        H0_wc_actual = np.matmul(H0_wc, RandomTranslation(t, t - 0.5))
        H1_wc_actual = np.matmul(H1_wc, RandomTranslation(t, t - 0.5))
        return H0_wc_actual, H1_wc_actual

    def RenderMeshes(scene, cameras):
        (H0_wc, H1_wc) = cameras
        (obj_meshes, GT_cents, GT_rads, H_list) = scene
        H0_wc_actual = np.copy(H0_wc)
        H1_wc_actual = np.copy(H1_wc)
        if (args.camera_noise_r != 0):
            H0_wc_actual, H1_wc_actual = GenerateRotationCameraNoise(H0_wc, H1_wc)
        if (args.camera_noise_t != 0):
            H0_wc_actual, H1_wc_actual = GenerateTranslationCameraNoise(H0_wc, H1_wc)

        render_list = [(H0_wc_actual, 'view0.png', 'depth0.png'), 
                       (H1_wc_actual, 'view1.png', 'depth1.png')]
        
        K = o3d.camera.PinholeCameraIntrinsic(img_width,img_height,f,f,ox,oy)

        # Rendering RGB-D frames given camera poses
        # create visualiser and get rendered views
        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = K
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=img_width, height=img_height, left=0, top=0)
        for m in obj_meshes:
            vis.add_geometry(m)
        ctr = vis.get_view_control()
        for (H_wc, name, dname) in render_list:
            cam.extrinsic = H_wc
            ctr.convert_from_pinhole_camera_parameters(cam)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(name, True)
            vis.capture_depth_image(dname, True)

        spheres, sphere_meshes, error = Generate3DSpheresFromCorrespondence(H0_wc, H1_wc, H1_wc_actual, H_list, GT_rads)
        sphere_meshes.append(obj_meshes[0])
        centres = Centres(GT_cents, [sphere[0] for sphere in spheres])
        centres.append(obj_meshes[0])

        scenes = [obj_meshes, sphere_meshes, centres]

        viewI = 0
        def switch_meshes(vis):
            nonlocal viewI
            viewI = (viewI + 1) % len(scenes)
            meshes = scenes[viewI]
            vis.clear_geometries()
            for m in meshes:
                vis.add_geometry(m)
            ctr = vis.get_view_control()
            cam.extrinsic = H1_wc
            ctr.convert_from_pinhole_camera_parameters(cam)
            vis.poll_events()
            vis.update_renderer()
        
        if (args.iters == 0):
            vis.register_key_callback(65, switch_meshes)
            vis.run()
        vis.destroy_window()
        return error
    
    def RunMultipleSimulations(scene, cameras):
        total_centre_e = 0
        total_radius_e = 0
        total_n = 0
        padding = 20
        data = []
        for _ in range(args.iters):
            data.append(RenderMeshes(scene, cameras))
        
        print()
        print('{:<14}| {:<{p}}| {:<{p}}| {:<{p}}'.format('', '# of Predictions', 'Centre Error', 'Radius Error', p=padding))
        print('{:-<{p}}--{:-<{p}}--{:-<{p}}--{:-<{p}}'.format('','','','', p=padding))
        for i in range(args.iters):
            (centre_e, radius_e, n) = data[i]
            print('{:<14}| {:<{p}}| {:.4f}{:<14}| {:.4f}{:<14}'.format('', n, centre_e, '', radius_e, '', p=padding))
            total_centre_e += centre_e * n
            total_radius_e += radius_e * n
            total_n += n
        average_centre_e = total_centre_e / total_n
        average_radius_e = total_radius_e / total_n
        print('{:-<{p}}--{:-<{p}}--{:-<{p}}--{:-<{p}}'.format('','','','', p=padding))
        print('{:<14}| {:.4f}{:<14}| {:.4f}{:<14}| {:.4f}{:<14}'.format('Average', total_n / args.iters, '', average_centre_e, '', average_radius_e, '', p=padding))

        with open('test_log', 'a') as log:
            log.write('{}, {}, {}, {:.4f}, {:.4f}\n'.format(args.camera_noise_t, args.camera_noise_r, args.iters, average_centre_e, average_radius_e))

    if (args.iters == 0):
        RenderMeshes(GenerateSpheres(), GenerateCameras())
    else:
        max_noise = 40
        resolution = 1
        spheres = GenerateSpheres()
        cameras = GenerateCameras()            
        for i in range(int(max_noise / resolution)):
            args.camera_noise_r = i * resolution
            RunMultipleSimulations(spheres, cameras)
