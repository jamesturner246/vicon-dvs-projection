
from vicon_dssdk import ViconDataStream
import numpy as np

def test_wand():
    prop_name = 'jt_wand'
    server = '127.0.0.1:801'

    client = ViconDataStream.Client()
    client.Connect(server)
    client.EnableSegmentData()

    while True:
        if not client.GetFrame():
            continue

        try:
            prop_quality = client.GetObjectQuality(prop_name)
        except ViconDataStream.DataStreamException:
            prop_quality = None

        if prop_quality is not None:
            root_segment = client.GetSubjectRootSegmentName(prop_name)
            rotation = client.GetSegmentGlobalRotationEulerXYZ(prop_name, root_segment)
            translation = client.GetSegmentGlobalTranslation(prop_name, root_segment)
            #break
            print('rotation:', np.round(rotation[0], 2))


    client.Disconnect()

    print()
    print('prop name:', prop_name)
    print('rotation:', rotation)
    print('translation:', translation)


"""
Vicon Euler anglesappear to be Tait-Bryan angles:
m[0] - rotation around x axis in [-pi, pi]
m[1] - rotation around (new) y axis [-pi/2, pi/2]
m[2] - rotation around (new) z axis [-pi, pi]
positive angles are counter-clockwise if looking towards the origin
"""

def tait_bryan_angles_to_rotation_matrix(m):
    M= np.array([
        [ 1, 0, 0],
        [ 0, np.cos(m[0]), np.sin(m[0])],
        [ 0, -np.sin(m[0]), np.cos(m[0])]])
    M= np.dot(M, np.array([
        [ np.cos(m[1]), 0, np.sin(m[1])],
        [ 0, 1, 0 ],
        [ -np.sin(m[1]), 0, np.cos(m[1])]]))
    M= np.dot(M, np.array([
        [ np.cos(m[2]), np.sin(m[2]), 0],
        [ -np.sin(m[2]), np.cos(m[2]), 0],
        [ 0, 0, 1]]))
    return M



if __name__ == '__main__':
    test_wand()
