import numpy as np

def project_lidar_to_camera(points, image_shape):
    # KITTI Calibration Matrix (Fixed for image_02)
    P_rect = np.array([[7.21e+02, 0.00e+00, 6.09e+02, 4.48e+01],
                       [0.00e+00, 7.21e+02, 1.72e+02, 2.16e-01],
                       [0.00e+00, 0.00e+00, 1.00e+00, 2.74e-03]])
    
    # Transformation from Velodyne to Camera coordinates
    Tr_velo_to_cam = np.array([[7.53e-03, -9.99e-01, -3.22e-04, -4.06e-03],
                               [1.48e-02, 7.28e-04, -9.99e-01, -7.63e-02],
                               [9.99e-01, 7.50e-03, 1.48e-02, -2.71e-01],
                               [0, 0, 0, 1]])

    # Filter points in front of the car
    points = points[points[:, 0] > 0]
    
    # Transform points
    pts_3d = np.column_stack((points[:, :3], np.ones(len(points))))
    pts_cam = pts_3d @ Tr_velo_to_cam.T
    
    # Project to 2D
    pts_2d = pts_cam @ P_rect.T
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    
    # Final image boundary mask
    mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < image_shape[1]) & \
           (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < image_shape[0])
           
    return pts_2d[mask], points[mask, 0]