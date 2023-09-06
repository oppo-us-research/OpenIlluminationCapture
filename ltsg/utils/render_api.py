#  created by Isabella Liu (lal005@ucsd.edu)

import os
import numpy as np
import trimesh


from easyhec.utils.utils_3d import rotx, transform_points

class PyrenderRenderMeshApiHelper:
    _renderer = None
    H, W = None, None

    # def __init__(self):
    #     self._renderer = None

    @staticmethod
    def get_renderer(H, W):
        import pyrender
        if PyrenderRenderMeshApiHelper._renderer is None or H != PyrenderRenderMeshApiHelper.H or W != PyrenderRenderMeshApiHelper.W:
            PyrenderRenderMeshApiHelper._renderer = pyrender.OffscreenRenderer(W, H)
        return PyrenderRenderMeshApiHelper._renderer


def pyrender_render_mesh_api(mesh: trimesh.Trimesh, object_pose, H, W, K, return_depth=False):
    import pyrender
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    camera = pyrender.IntrinsicsCamera(fx=fu, fy=fv, cx=cu, cy=cv)
    coord_convert = np.eye(4)
    coord_convert[:3, :3] = rotx(-np.pi)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0)
    object_pose = np.asarry(object_pose)
    vertices = transform_points(mesh.vertices, object_pose)
    mesh = trimesh.Trimesh(vertices, mesh.faces)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    scene.add(camera, pose=coord_convert)
    scene.add(light, pose=coord_convert)
    flags = pyrender.constants.RenderFlags.DEPTH_ONLY
    r = PyrenderRenderMeshApiHelper.get_renderer(H, W)
    depth = r.render(scene, flags)
    r.delete()
    if return_depth:
        return depth
    mask = depth > 0
    return mask

