
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import pyvicon as pv
import numpy as np
import stl
import cv2
import dv


def projection():

    prop_mesh = stl.mesh.Mesh.from_file('./screwdriver-decimated.stl')

    dv_space_coefficients_file = './data/dv_space_coefficients.npy'
    dv_space_constants_file = './data/dv_space_constants.npy'

    dv_space_coefficients = np.load(dv_space_coefficients_file)
    dv_space_constants = np.load(dv_space_constants_file)

    mesh_marker_names = {}
    mesh_marker_coordinates = {}
    vicon_marker_coordinates = {}

    # # screwdriver.stl: marker names and coordinates
    # mesh_marker_names['screwdriver'] = [
    #     'handle_1', 'handle_2', 'shaft_base', 'shaft_tip'
    # ]
    # mesh_marker_coordinates['screwdriver'] = np.array([
    #     [ 0.0,  78.0,   13.5], # handle_1
    #     [ 0.0,  78.0,  -13.5], # handle_2
    #     [ 5.0,  119.9,  0.0 ], # shaft_base
    #     [-5.0,  163.9,  0.0 ], # shaft_tip
    # ])
    # vicon_marker_coordinates['screwdriver'] = np.empty(
    #     mesh_marker_coordinates['screwdriver'].shape
    # )



    # PROTOTYPE: screwdriver.stl marker coordinates
    mesh_marker_names['screwdriver'] = [
        'handle_1', 'handle_2', 'shaft_base', 'shaft_tip'
    ]
    mesh_marker_coordinates['screwdriver'] = np.array([
        [ 0.0,  78.0,   13.5], # handle_1
        [ 0.0,  78.0,  -13.5], # handle_2
        [ 7.5,  100.0,  0.0 ], # shaft_base alternate position
        [-5.0,  163.9,  0.0 ], # shaft_tip
    ])
    vicon_marker_coordinates['screwdriver'] = np.empty(
        mesh_marker_coordinates['screwdriver'].shape
    )


    vicon_address, vicon_port = '192.168.1.1', 801

    dv_address, dv_event_port, dv_frame_port = '192.168.1.100', 36000, 36001
    dv_frame_shape = (260, 346, 3)

    dv_frame_f = dv.NetworkFrameInput(address=dv_address, port=dv_frame_port)
    dv_event_f = dv.NetworkEventInput(address=dv_address, port=dv_event_port)

    prop_mask = np.zeros(dv_frame_shape[:2], dtype='uint8')
    frame_image = np.zeros(dv_frame_shape, dtype='uint8')
    event_image = np.zeros(dv_frame_shape, dtype='uint8')

    frame_image_video_file = './frame_image_video.avi'
    event_image_video_file = './event_image_video.avi'

    frame_image_video = cv2.VideoWriter(
        frame_image_video_file, cv2.VideoWriter_fourcc(*'MJPG'),
        2, dv_frame_shape[1::-1])
    event_image_video = cv2.VideoWriter(
        event_image_video_file, cv2.VideoWriter_fourcc(*'MJPG'),
        2, dv_frame_shape[1::-1])


    vicon_client = pv.PyVicon()
    #print('version: ' + vicon_client.__version__)

    result = vicon_client.connect(f'{vicon_address}:{vicon_port}')
    #print('connect:', result)

    result = vicon_client.enable_marker_data()
    #print('enable_marker_data:', result)


    done = False
    dv_event = next(dv_event_f)

    for dv_frame in dv_frame_f:
        if done:
            break

        result = vicon_client.get_frame()
        #print('get_frame:', result)

        if result == pv.Result.NoFrame:
            continue

        prop_count = vicon_client.get_subject_count()
        #print('props:', prop_count)

        for i in range(prop_count):
            prop_name = vicon_client.get_subject_name(i)
            #print('prop name:', prop_name)

            prop_quality = vicon_client.get_subject_quality(prop_name)
            #print(' ', prop_name, 'quality:', prop_quality)

            if prop_quality is not None:
                marker_count = vicon_client.get_marker_count(prop_name)
                #print(' ', prop_name, 'markers:', marker_count)

                assert(marker_count == mesh_marker_coordinates[prop_name].shape[0])

                for j in range(marker_count):

                    marker_name = vicon_client.get_marker_name(prop_name, j)
                    #print('   ', prop_name, 'marker', j, 'name:', marker_name)

                    assert(marker_name == mesh_marker_names[prop_name][j])

                    marker_coordinates = vicon_client.get_marker_global_translation(prop_name, marker_name)
                    #print('   ', prop_name, 'marker', j, 'coordinates', marker_coordinates)

                    vicon_marker_coordinates[prop_name][j, :] = marker_coordinates

                #print()

                # estimate Vicon space transformation
                x = mesh_marker_coordinates[prop_name]
                y = vicon_marker_coordinates[prop_name]
                regressor = MultiOutputRegressor(
                    estimator=LinearRegression(),
                ).fit(x, y)

                vicon_space_coefficients = np.array([re.coef_ for re in regressor.estimators_]).T
                vicon_space_constants = np.array([[re.intercept_ for re in regressor.estimators_]])

                # print('Vicon space coefficients')
                # print(vicon_space_coefficients)
                # print('Vicon space intercepts')
                # print(vicon_space_constants)
                # print()


                # transform STL mesh space to Vicon space, then
                # transform from Vicon space to DV camera space
                vicon_space_p = np.matmul(prop_mesh.vectors, vicon_space_coefficients) + vicon_space_constants
                dv_space_p = np.matmul(vicon_space_p, dv_space_coefficients) + dv_space_constants
                dv_space_p = dv_space_p[:, :, :2] * (1.0 / dv_space_p[:, :, 2, np.newaxis])
                dv_space_p_int = np.rint(dv_space_p).astype('int32')

                frame_image[:, :, :] = dv_frame.image
                cv2.fillPoly(frame_image, dv_space_p_int, (255, 0, 0))
                cv2.imshow('frame image', frame_image)
                frame_image_video.write(frame_image)





                prop_mask.fill(0)
                cv2.fillPoly(prop_mask, dv_space_p_int, 255)
                event_image[event_image[prop_mask.astype('bool')] == (0, 0, 255)] = (255, 255, 0)
                event_image[event_image[prop_mask.astype('bool')] == (0, 255, 0)] = (255, 0,   0)
                event_image_video.write(event_image)
                event_image.fill(0)





                # dt IS dv_frame INTERVAL


                while dv_event.timestamp < dv_frame.timestamp + dt:

                    # IF dv_event not None, ADD dv_event to image

                    if dv_event:

                        # ADD EVENT TO IMAGE

                        pass


                    dv_event = next(dv_event_f)






                k = cv2.waitKey(1)
                if k == 27 or k == ord('q'):
                    done = True


    result = vicon_client.disconnect()
    #print('disconnect:', result)

    event_image_video.release()
    frame_image_video.release()

    return


if __name__ == '__main__':
    projection()
