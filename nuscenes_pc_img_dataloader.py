import open3d
import torch.utils.data as data
import random
import numbers
import os
import os.path
import numpy as np
import struct
import math
import torch
import torchvision
import cv2
from PIL import Image
from torchvision import transforms
import pickle
from pyquaternion import Quaternion

from nuScenes import options

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.nuscenes import NuScenes

from scipy.sparse import coo_matrix


def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def camera_matrix_scaling(K: np.ndarray, s: float):
    K_scale = s * K
    K_scale[2, 2] = 1
    return K_scale


def camera_matrix_cropping(K: np.ndarray, dx: float, dy: float):
    K_crop = np.copy(K)
    K_crop[0, 2] -= dx
    K_crop[1, 2] -= dy
    return K_crop


class FarthestSampler:
    def __init__(self, dim=3):
        self.dim = dim

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=0)

    def sample(self, pts, k):
        farthest_pts = np.zeros((self.dim, k))
        farthest_pts_idx = np.zeros(k, dtype=np.int)
        init_idx = np.random.randint(len(pts))
        farthest_pts[:, 0] = pts[:, init_idx]
        farthest_pts_idx[0] = init_idx
        distances = self.calc_distances(farthest_pts[:, 0:1], pts)
        for i in range(1, k):
            idx = np.argmax(distances)
            farthest_pts[:, i] = pts[:, idx]
            farthest_pts_idx[i] = idx
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i + 1], pts))
        return farthest_pts, farthest_pts_idx


def downsample_with_reflectance(pointcloud, reflectance, voxel_grid_downsample_size):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np.transpose(pointcloud[0:3, :]))
    reflectance_max = np.max(reflectance)

    fake_colors = np.zeros((pointcloud.shape[1], 3))
    fake_colors[:, 0] = reflectance / reflectance_max
    pcd.colors = open3d.utility.Vector3dVector(fake_colors)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
    down_pcd_points = np.transpose(np.asarray(down_pcd.points))  # 3xN
    pointcloud = down_pcd_points
    reflectance = np.asarray(down_pcd.colors)[:, 0] * reflectance_max

    return pointcloud, reflectance


def load_dataset_info(filepath):
    with open(filepath, 'rb') as f:
        dataset_read = pickle.load(f)
    return dataset_read


def make_nuscenes_dataset(root_path):
    dataset = load_dataset_info(os.path.join(root_path, 'dataset_info_new.list'))
    return dataset


def get_sample_data_ego_pose_P(nusc, sample_data):
    sample_data_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    sample_data_pose_R = np.asarray(Quaternion(sample_data_pose['rotation']).rotation_matrix).astype(np.float32)
    sample_data_pose_t = np.asarray(sample_data_pose['translation']).astype(np.float32)
    sample_data_pose_P = get_P_from_Rt(sample_data_pose_R, sample_data_pose_t)
    return sample_data_pose_P


def get_calibration_P(nusc, sample_data):
    calib = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    R = np.asarray(Quaternion(calib['rotation']).rotation_matrix).astype(np.float32)
    t = np.asarray(calib['translation']).astype(np.float32)
    P = get_P_from_Rt(R, t)
    return P


def get_P_from_Rt(R, t):
    P = np.identity(4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t
    return P


def get_camera_K(nusc, camera):
    calib = nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])
    return np.asarray(calib['camera_intrinsic']).astype(np.float32)


def transform_pc_np(P, pc_np):
    """

    :param pc_np: 3xN
    :param P: 4x4
    :return:
    """
    pc_homo_np = np.concatenate((pc_np,
                                 np.ones((1, pc_np.shape[1]), dtype=pc_np.dtype)),
                                axis=0)
    P_pc_homo_np = np.dot(P, pc_homo_np)
    return P_pc_homo_np[0:3, :]


class nuScenesLoader(data.Dataset):
    def __init__(self, root, mode, opt: options.Options):
        super(nuScenesLoader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        # farthest point sample
        self.farthest_sampler = FarthestSampler(dim=3)

        # list of (traversal, pc_timestamp, pc_timestamp_idx, traversal_pc_num)
        if mode == 'train':
            self.pc_path=os.path.join(root,'train','PC')
            self.img_path = os.path.join(root,'train', 'img')
            self.K_path = os.path.join(root,'train', 'K')
        else:
            self.pc_path = os.path.join(root, 'test', 'PC')
            self.img_path = os.path.join(root, 'test', 'img')
            self.K_path = os.path.join(root, 'test', 'K')

        self.length=len(os.listdir(self.pc_path))


    def augment_img(self, img_np):
        """

        :param img: HxWx3, np.ndarray
        :return:
        """
        # color perturbation
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(np.uint8(img_np))))

        return img_color_aug_np

    def generate_random_transform(self,
                                  P_tx_amplitude, P_ty_amplitude, P_tz_amplitude,
                                  P_Rx_amplitude, P_Ry_amplitude, P_Rz_amplitude):
        """

        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-P_tx_amplitude, P_tx_amplitude),
             random.uniform(-P_ty_amplitude, P_ty_amplitude),
             random.uniform(-P_tz_amplitude, P_tz_amplitude)]
        angles = [random.uniform(-P_Rx_amplitude, P_Rx_amplitude),
                  random.uniform(-P_Ry_amplitude, P_Ry_amplitude),
                  random.uniform(-P_Rz_amplitude, P_Rz_amplitude)]

        rotation_mat = angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        return P_random.astype(np.float32)

    def downsample_np(self, pc_np, intensity_np, k):
        if pc_np.shape[1] >= k:
            choice_idx = np.random.choice(pc_np.shape[1], k, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < k:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], k - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]

        return pc_np, intensity_np

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        pc_data=np.load(os.path.join(self.pc_path,'%06d.npy'%index))
        pc_np=pc_data[0:3,:]
        intensity_np=pc_data[3:,:]

        # load point cloud

        # random sampling
        pc_np, intensity_np = self.downsample_np(pc_np, intensity_np, self.opt.input_pt_num)

        img = np.load(os.path.join(self.img_path,'%06d.npy'%index))
        K = np.load(os.path.join(self.K_path,'%06d.npy'%index))
        # random crop into input size
        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.opt.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.opt.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.opt.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.opt.img_H) / 2)
        # crop image
        img = img[img_crop_dy:img_crop_dy + self.opt.img_H,
              img_crop_dx:img_crop_dx + self.opt.img_W, :]
        K = camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)

        #  ------------- apply random transform on points under the NWU coordinate ------------
        # if 'train' == self.mode:

        # -------------- augmentation ----------------------
        # pc_np, intensity_np = self.augment_pc(pc_np, intensity_np)
        #print(img.shape)
        if 'train' == self.mode:
            img = self.augment_img(img)

        # random rotate pc_np
        # pc_np = transform_pc_np(Pr, pc_np)

        # å››åˆ†ä¹‹ä¸€çš„å°ºå¯¸
        K_4 = camera_matrix_scaling(K, 0.25)

        pc_ = np.dot(K_4, pc_np)
        pc_mask = np.zeros((1, np.shape(pc_np)[1]), dtype=np.float32)
        pc_[0:2, :] = pc_[0:2, :] / pc_[2:, :]
        xy = np.floor(pc_[0:2, :])
        is_in_picture = (xy[0, :] >= 0) & (xy[0, :] <= (self.opt.img_W * 0.25 - 1)) & (xy[1, :] >= 0) & (
                xy[1, :] <= (self.opt.img_H * 0.25 - 1)) & (pc_[2, :] > 0)
        pc_mask[:, is_in_picture] = 1.

        pc_kpt_idx = np.where(pc_mask.squeeze() == 1)[0]
        index = np.random.permutation(len(pc_kpt_idx))[0:self.opt.num_kpt]
        pc_kpt_idx = pc_kpt_idx[index]

        pc_outline_idx = np.where(pc_mask.squeeze() == 0)[0]
        index = np.random.permutation(len(pc_outline_idx))[0:self.opt.num_kpt]
        pc_outline_idx = pc_outline_idx[index]

        xy2 = xy[:, is_in_picture]
        img_mask = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])),
                              shape=(int(self.opt.img_H * 0.25), int(self.opt.img_W * 0.25))).toarray()

        img_mask = np.array(img_mask)
        img_mask[img_mask > 0] = 1.

        img_kpt_index = xy[1, pc_kpt_idx] * self.opt.img_W * 0.25 + xy[0, pc_kpt_idx]

        img_outline_index = np.where(img_mask.squeeze().reshape(-1) == 0)[0]
        index = np.random.permutation(len(img_outline_index))[0:self.opt.num_kpt]
        img_outline_index = img_outline_index[index]

        P_np = self.generate_random_transform(self.opt.P_tx_amplitude, self.opt.P_ty_amplitude,
                                              self.opt.P_tz_amplitude,
                                              self.opt.P_Rx_amplitude, self.opt.P_Ry_amplitude,
                                              self.opt.P_Rz_amplitude)

        '''r_max=np.max(np.sqrt(np.sum(pc_np**2,axis=0)))
        print('max range',r_max)'''


        pc_np = np.dot(P_np[0:3, 0:3], pc_np) + P_np[0:3, 3:]
        P_inv = np.linalg.inv(P_np)

        node_a_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                              int(self.opt.node_a_num * 8),
                                                                              replace=False)],
                                                    k=self.opt.node_a_num)
        node_b_np, _ = self.farthest_sampler.sample(pc_np[:, np.random.choice(pc_np.shape[1],
                                                                              int(self.opt.node_b_num * 8),
                                                                              replace=False)],
                                                    k=self.opt.node_b_num)

        # visualize nodes
        # ax = vis_tools.plot_pc(pc_np, size=1)
        # ax = vis_tools.plot_pc(node_a_np, size=10, ax=ax)
        # plt.show()

        # -------------- convert to torch tensor ---------------------
        pc = torch.from_numpy(pc_np.astype(np.float32))  # 3xN
        intensity = torch.from_numpy(intensity_np.astype(np.float32))  # 1xN
        sn = torch.zeros(pc.size(), dtype=pc.dtype, device=pc.device)

        P = torch.from_numpy(P_inv.astype(np.float32))  # 3x4

        img = torch.from_numpy(img.astype(np.float32)/255).permute(2, 0, 1).contiguous()  # 3xHxW
        K = torch.from_numpy(K_4.astype(np.float32))  # 3x3

        # print(P)

        # print(pc)
        # print(intensity)

        return {'pc': pc,
                'intensity': intensity,
                'sn': sn,
                'P': P,
                'img': img,
                'K': K,

                'pc_mask': torch.from_numpy(pc_mask).float(),
                'img_mask': torch.from_numpy(img_mask).float(),  # (40,128)

                'pc_kpt_idx': torch.from_numpy(pc_kpt_idx),  # 512
                'pc_outline_idx': torch.from_numpy(pc_outline_idx),  # 512
                'img_kpt_idx': torch.from_numpy(img_kpt_index).long(),  # 512
                'img_outline_index': torch.from_numpy(img_outline_index).long(),
                'node_a': torch.from_numpy(node_a_np).float(),
                'node_b': torch.from_numpy(node_b_np).float()
                }


if __name__ == '__main__':
    root_path = 'F:\\nuscenes'
    opt = options.Options()
    dataset = nuScenesLoader(root_path, 'train', opt)
    #print(dataset[100]['img'].size())
    #assert False
    for i in range(0, len(dataset), 1000):
        print('--- %d ---' % i)

        data = dataset[i]
        P = data['P'].numpy()
        img = data['img'].numpy().transpose(1, 2, 0)
        print(np.max(img))
        '''print(img.shape)
        H = img.shape[0]
        W = img.shape[1]
        K = data[7].numpy()
        pc = data[0].numpy()

        pointcloud2 = open3d.geometry.PointCloud()
        pointcloud2.points = open3d.utility.Vector3dVector(pc.T)
        open3d.io.write_point_cloud('pc_rot.ply', pointcloud2)

        pc = np.dot(P[0:3, 0:3], pc) + P[0:3, 3:]

        pointcloud = open3d.geometry.PointCloud()
        pointcloud.points = open3d.utility.Vector3dVector(pc.T)
        open3d.io.write_point_cloud('pc.ply', pointcloud)

        uvz = np.dot(K, pc)
        depth = uvz[2, :]
        uv = uvz[0:2, :] / uvz[2:, :]
        in_img = (depth > 0) & (uv[0, :] >= 0) & (uv[0, :] <= W - 1) & (uv[1, :] >= 0) & (uv[1, :] <= H - 1)
        print(np.sum(in_img))
        print(P)
        uv = uv[:, in_img]
        depth = depth[in_img]
        # print(np.max(img))
        plt.figure(1)
        plt.imshow(img)
        plt.scatter(uv[0, :], uv[1, :], c=[depth], s=1)
        #plt.show()
        pointcloud = open3d.geometry.PointCloud()
        pointcloud.points = open3d.utility.Vector3dVector(data[0].numpy().T)
        #open3d.visualization.draw_geometries([pointcloud])'''
