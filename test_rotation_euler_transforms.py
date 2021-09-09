import numpy as np
from calibrate_projection import euler_angles_to_rotation_matrix, rotation_matrix_to_euler_angles

def test_rotation_euler_transforms():
    tolerance= 1e-8
    m= np.empty(3)
    scale= np.array([2*np.pi, np.pi, 2*np.pi])
    offset= np.array([-np.pi, 0, -np.pi])
    for i in range(10):
        m= np.random.uniform(size=3)
        m= m*scale+offset
        M= euler_angles_to_rotation_matrix(m)
        m2= rotation_matrix_to_euler_angles(M)
        assert np.linalg.norm(m-m2) < tolerance
        
if __name__ == '__main__':
    test_rotation_euler_transforms()

    
