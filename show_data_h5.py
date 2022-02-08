import tables
import numpy as np
import cv2

def show_data_h5(path, n_cameras=1):
    f = tables.open_file(path, 'r')
    props_names = [item[0] for item in f.root.props._v_children.items()]
    image = [np.zeros((260, 346, 3), dtype='uint8') for i in range(n_cameras)]

    iter_timestamp = f.root.timestamp.iterrows()
    iter_events = []
    iter_xy = []
    for i in range(n_cameras):
        iter_events.append(f.root[f'camera_{i}_events'].iterrows())
        iter_xy.append({})
    for i in range(n_cameras):
        for prop_name in props_names:
            iter_xy[i][prop_name] = f.root.props[prop_name][f'camera_{i}_xy'].iterrows()

    done = False
    while not done:
        timestamp = next(iter_timestamp)
        print('time:', timestamp)

        for i in range(n_cameras):
            events = next(iter_events[i])
            image[i].fill(0)
            image[i][:, :, :2] = events
            image[i][image[i] > 0] = 255

            for prop_name in props_names:
                xy = next(iter_xy[i][prop_name])
                xy_int = np.rint(xy).astype('int32')
                cv2.circle(image[i], (xy_int[0], xy_int[1]), 3, (0, 0, 255), -1)

            # show event image
            cv2.imshow(f'camera {i} image', image[i])
            k = cv2.waitKey(10)
            if k == ord('q'):
                cv2.destroyAllWindows()
                done = True
