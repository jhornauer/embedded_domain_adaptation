import os
import argparse
from data.kitti_utils import generate_depth_map
import h5py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, help='path to kitti dataset')
    args = parser.parse_args()

    root_dir = os.path.join(args.data_dir, 'kitti')
    for folder in os.listdir(root_dir):
        calib_path = os.path.join(root_dir, folder)
        for scene in os.listdir(os.path.join(root_dir, folder)):
            path = os.path.join(root_dir, folder, scene)
            if os.path.isdir(path):
                os.mkdir(os.path.join(path, 'depth'))
                os.mkdir(os.path.join(path, 'depth', 'data'))
                depth_dir = os.path.join(path, 'depth', 'data')
                velodyne_dir = os.path.join(path, 'velodyne_points', 'data')
                for file in os.listdir(velodyne_dir):
                    file_dir = os.path.join(velodyne_dir, file)
                    depth = generate_depth_map(calib_path, file_dir)
                    depth_file = file.replace('.bin', '.h5')

                    hf = h5py.File(os.path.join(depth_dir, depth_file), "w")
                    hf.create_dataset('depth', data=depth, compression="gzip")
                    hf.close()
