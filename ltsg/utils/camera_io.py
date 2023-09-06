#  created by Isabella Liu (lal005@ucsd.edu).
import json
from ltsg.structures.camera import Camera


def read_cameras(json_file):
    with open(json_file, 'r') as f:
        cams_json = json.load(f)
    cameras = {}
    for cam_id in cams_json:
        cam = cams_json[cam_id]
        cameras[cam_id] = Camera(cam['K'], cam['dist'], cam['width'], cam['height'], cam['pose'], cam['newK'])
    return cameras


def write_cameras(camera, out_path):
    camera_jsons = {}
    for imgid, cam in camera.items():
        cam_dict = {
            "K": cam.K.tolist(),
            "dist": cam.dist.tolist(),
            "width": cam.width,
            "height": cam.height,
            "newK": cam.newK.tolist(),
            "pose": cam.pose.tolist(),
        }
        camera_jsons[imgid] = cam_dict
    with open(out_path, 'w') as f:
        json.dump(camera_jsons, f, indent=4)
