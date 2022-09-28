from nuscenes.nuscenes import NuScenes


def search_nearby_cameras(nusc,
                          init_camera,
                          max_translation,
                          direction,
                          lidar_P_inv,
                          nearby_camera_token_list):
    init_camera_direction_token = init_camera[direction]
    if init_camera_direction_token == '':
        return nearby_camera_token_list

    camera = nusc.get('sample_data', init_camera_direction_token)
    while True:
        camera_token = camera[direction]
        if camera_token == '':
            break
        camera = nusc.get('sample_data', camera_token)
        camera_P = get_sample_data_ego_pose_P(nusc, camera)
        P_lc = np.dot(lidar_P_inv, camera_P)
        t_lc = P_lc[0:3, 3]
        t_lc_norm = np.linalg.norm(t_lc)

        if t_lc_norm < max_translation:
            nearby_camera_token_list.append(camera_token)
        else:
            break
    return nearby_camera_token_list


def get_nearby_camera_token_list(nusc,
                                 lidar_token,
                                 max_translation,
                                 camera_name):
    lidar = nusc.get('sample_data', lidar_token)
    lidar_P = get_sample_data_ego_pose_P(nusc, lidar)
    lidar_P_inv = np.linalg.inv(lidar_P)

    lidar_sample_token = lidar['sample_token']
    lidar_sample = nusc.get('sample', lidar_sample_token)

    init_camera_token = lidar_sample['data'][camera_name]
    init_camera = nusc.get('sample_data', init_camera_token)
    nearby_camera_token_list = [init_camera_token]

    nearby_camera_token_list = search_nearby_cameras(
        nusc,
        init_camera,
        max_translation,
        'next',
        lidar_P_inv,
        nearby_camera_token_list)
    nearby_camera_token_list = search_nearby_cameras(
        nusc,
        init_camera,
        max_translation,
        'prev',
        lidar_P_inv,
        nearby_camera_token_list)

    return nearby_camera_token_list

def get_nearby_camera(nusc, lidar_token, max_translation):
    cam_list = ['CAM_FRONT',
                'CAM_FRONT_LEFT',
                'CAM_FRONT_RIGHT',
                'CAM_BACK',
                'CAM_BACK_LEFT',
                'CAM_BACK_RIGHT']
    nearby_cam_token_dict = {}
    for camera_name in cam_list:
        nearby_cam_token_dict[camera_name] \
            = get_nearby_camera_token_list(nusc,
                                           lidar_token,
                                           max_translation,
                                           camera_name)
    return nearby_cam_token_dict
def make_nuscenes_dataset(nusc, frame_skip, max_translation):
    dataset = []

    lidar_token_list = get_lidar_token_list(nusc,
                                            frame_skip)
    for i, lidar_token in enumerate(lidar_token_list):
        # begin_t = time.time()
        nearby_camera_token_dict = get_nearby_camera(nusc,
                                                     lidar_token,
                                                     max_translation)

        dataset.append((lidar_token, nearby_camera_token_dict))

        # print('lidar %s takes %f' % (lidar_token, time.time()-begin_t))
        if i % 100 == 0:
            print('%d done...' % i)

    return dataset


if __name__=='__main__':
    nusc=NuScenes(version='v1.0-mini',dataroot='G:\\nuscene\\nuscenes\\mini')
