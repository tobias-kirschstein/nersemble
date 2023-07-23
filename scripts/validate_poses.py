import tyro
import pyvista as pv

from dreifus.pyvista import add_camera_frustum, add_coordinate_axes

from nersemble.data_manager.multi_view_data import NeRSembleDataManager


def main():
    data_manager = NeRSembleDataManager(18, "EMO-1-shout+laugh")
    camera_params = data_manager.load_camera_params()

    p = pv.Plotter()
    add_coordinate_axes(p, scale=0.1)
    for serial, cam_pose in camera_params.world_2_cam.items():
        image = data_manager.load_image(0, serial)
        add_camera_frustum(p, cam_pose, camera_params.intrinsics, label=serial, image=image)

    p.show()


if __name__ == '__main__':
    tyro.cli(main)
