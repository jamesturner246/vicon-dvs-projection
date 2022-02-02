import cv2

'''
Manually find first DVS event and frame after start signal
'''

def seek_dvs_start(
        n_cameras,
        events_file, events_iter,
        frames_file, frames_iter,
        start_signal_delay_secs=3):

    dvs_start_timestamp = [None for i in range(n_cameras)]

    for i in range(n_cameras):
        search_image = np.empty((dv_cam_height[i], dv_cam_width[i] * 2, 3), dtype='uint8')
        event_search_image = search_image[:, :dv_cam_width[i]]
        frame_search_image = search_image[:, dv_cam_width[i]:]

        batch = 30
        i_event = 0
        n_event = len(events_file[i].root[f'timestamp_{i}'])
        event_timestamp = events_file[i].root[f'timestamp_{i}']
        event_xy_undistorted = events_file[i].root[f'xy_undistorted_{i}']

        i_frame = 0
        n_frame = len(frames_file[i].root[f'timestamp_{i}'])
        frame_timestamp = frames_file[i].root[f'timestamp_{i}']
        frame_image_undistorted = frames_file[i].root[f'image_undistorted_{i}']

        while True:
            # search events
            event_search_image.fill(0)
            for xy in event_xy_undistorted[i_event:(i_event + batch)]:
                xy_int = np.rint(xy).astype('int32')
                xy_bounded = all(xy_int >= 0) and all(xy_int < [dv_cam_width[i], dv_cam_height[i]])
                if xy_bounded:
                    event_search_image[xy_int[1], xy_int[0]] = 255

            # search frames
            if frame_timestamp[i_frame] < event_timestamp[i_event]:
                while True:
                    i_frame += 1
                    if i_frame == n_frame:
                        i_frame -= 1
                        break
                    if frame_timestamp[i_frame] > event_timestamp[i_event]:
                        i_frame -= 1
                        break
            elif frame_timestamp[i_frame] > event_timestamp[i_event]:
                while True:
                    i_frame -= 1
                    if i_frame < 0:
                        i_frame = 0
                        break
                    if frame_timestamp[i_frame] <= event_timestamp[i_event]:
                        break
            frame_search_image[:] = frame_image_undistorted[i_frame]

            print(f'dv {i} timestamps (usec):'
                  f'  event: {event_timestamp[i_event]:,}:'
                  f'  frame: {frame_timestamp[i_frame]:,}', end='\r')

            cv2.imshow(f'find first dv {i} event and frame', search_image)
            k = cv2.waitKey(0)
            if k == ord(' '):
                cv2.destroyWindow(f'find first dv {i} event and frame')
                break
            elif k == ord(','):
                i_event = max(i_event - batch, 0)
            elif k == ord('<'):
                i_event = max(i_event - (50 * batch), 0)
            elif k == ord('.'):
                i_event = min(i_event + batch, n_event - 1)
            elif k == ord('>'):
                i_event = min(i_event + (50 * batch), n_event - 1)

        print()

        dvs_start_timestamp[i] = np.uint64(event_timestamp[i_event] + 3000000)

    return dvs_start_timestamp
