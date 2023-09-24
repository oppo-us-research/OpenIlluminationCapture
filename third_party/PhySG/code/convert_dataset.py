import os
import json

import numpy as np
import json
import copy
import open3d as o3d


def get_tf_cams(cam_dict, target_radius=1.):
    cam_centers = []
    for im_name in cam_dict:
        W2C = np.array(cam_dict[im_name]['W2C']).reshape((4, 4))
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    scale = target_radius / radius

    return translate, scale


def normalize_cam_dict(combined_data, target_radius=1., in_geometry_file=None, out_geometry_file=None):


    combined_data = convert_data(combined_data)


    translate, scale = get_tf_cams(combined_data, target_radius=target_radius)

    if in_geometry_file is not None and out_geometry_file is not None:
        # check this page if you encounter issue in file io: http://www.open3d.org/docs/0.9.0/tutorial/Basic/file_io.html
        geometry = o3d.io.read_triangle_mesh(in_geometry_file)
        
        tf_translate = np.eye(4)
        tf_translate[:3, 3:4] = translate
        tf_scale = np.eye(4)
        tf_scale[:3, :3] *= scale
        tf = np.matmul(tf_scale, tf_translate)

        geometry_norm = geometry.transform(tf)
        o3d.io.write_triangle_mesh(out_geometry_file, geometry_norm)
  
    def transform_pose(W2C, translate, scale):
        C2W = np.linalg.inv(W2C)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        return np.linalg.inv(C2W)

    out_cam_dict = copy.deepcopy(combined_data)
    for img_name in out_cam_dict:
        W2C = np.array(out_cam_dict[img_name]['W2C']).reshape((4, 4))
        W2C = transform_pose(W2C, translate, scale)
        assert(np.isclose(np.linalg.det(W2C[:3, :3]), 1.))
        out_cam_dict[img_name]['W2C'] = list(W2C.flatten())

    return out_cam_dict


def calculate_K(camera_angle_x, img_width,img_height):
    focal_length = img_width / (2 * np.tan(camera_angle_x / 2))
    K = [focal_length, 0, img_width / 2, 0, 
        0, focal_length, img_height / 2, 0,
        0, 0, 1, 0,
        0, 0, 0, 1]
    return K


def convert_data(old_data):
    new_data = {}

    for key, frame in old_data.items():
        new_format = {}
        K = calculate_K(frame['camera_angle_x'], w, h) 
        new_format['K'] = K
        # Copy the transform matrix to W2C
        C2W = np.array(frame['transform_matrix'])
        camera_position = C2W[:3, 3]
        C2W[:3, 3] =  1 * camera_position
        W2C = np.linalg.inv(C2W)
        new_format['W2C'] = [item for sublist in W2C for item in sublist]

        # Calculate K values from camera_angle_x and calib_imgw, then assign them to K
        

        # Copy calib_imgw to img_size in two places
        new_format['img_size'] = [w, h]

        new_data[key] = new_format

    return new_data

def traverse_and_edit(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.split(dirpath)[-1] == "output":
            for filename in filenames:
                # if the file is not a json file, skip
                test_filepath = os.path.join(dirpath, "transforms_test.json")
                train_filepath = os.path.join(dirpath, "transforms_train.json")
                if os.path.exists(test_filepath) and os.path.exists(train_filepath):
                # read the json file
                    with open(test_filepath, "r") as test_file, open(train_filepath, "r") as train_file:
                        test_data = json.load(test_file)
                        train_data = json.load(train_file)
                        if "frames" in test_data and "frames" in train_data:
                            test_data = test_data["frames"]
                            train_data = train_data["frames"]
                            combined_data = {**test_data, **train_data}

                            modified_data = normalize_cam_dict(combined_data, target_radius=1.)
                            test_keys = test_data.keys()
                            train_keys = train_data.keys()

                            test_result = {key: modified_data[key] for key in test_keys}
                            train_result = {key: modified_data[key] for key in train_keys}

                            # write back to the json files
                            with open(test_filepath, "w") as test_file, open(train_filepath, "w") as train_file:
                                json.dump(test_result, test_file, indent=4)
                                json.dump(train_result, train_file, indent=4)

# output img size, for caculating K
w,h = 800,1200
# replace this with your openillumination dataset path
traverse_and_edit("./lightstage_dataset")
