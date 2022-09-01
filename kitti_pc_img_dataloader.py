import os
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import math
import open3d as o3d
import cv2
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import coo_matrix


class KittiCalibHelper:
    def __init__(self, root_path):
        self.root_path = root_path
        self.calib_matrix_dict = self.read_calib_files()

    def read_calib_files(self):
        seq_folders = [name for name in os.listdir(
            os.path.join(self.root_path, 'calib'))]
        calib_matrix_dict = {}
        for seq in seq_folders:
            calib_file_path = os.path.join(
                self.root_path, 'calib', seq, 'calib.txt')
            with open(calib_file_path, 'r') as f:
                for line in f.readlines():
                    seq_int = int(seq)
                    if calib_matrix_dict.get(seq_int) is None:
                        calib_matrix_dict[seq_int] = {}

                    key = line[0:2]
                    mat = np.fromstring(line[4:], sep=' ').reshape(
                        (3, 4)).astype(np.float32)
                    if 'Tr' == key:
                        P = np.identity(4)
                        P[0:3, :] = mat
                        calib_matrix_dict[seq_int][key] = P
                    else:
                        K = mat[0:3, 0:3]
                        calib_matrix_dict[seq_int][key + '_K'] = K
                        fx = K[0, 0]
                        fy = K[1, 1]
                        cx = K[0, 2]
                        cy = K[1, 2]
                        # mat[0, 3] = fx*tx + cx*tz
                        # mat[1, 3] = fy*ty + cy*tz
                        # mat[2, 3] = tz
                        tz = mat[2, 3]
                        tx = (mat[0, 3] - cx * tz) / fx
                        ty = (mat[1, 3] - cy * tz) / fy
                        P = np.identity(4)
                        P[0:3, 3] = np.asarray([tx, ty, tz])
                        calib_matrix_dict[seq_int][key] = P
        return calib_matrix_dict

    def get_matrix(self, seq: int, matrix_key: str):
        return self.calib_matrix_dict[seq][matrix_key]

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
            distances = np.minimum(distances, self.calc_distances(farthest_pts[:, i:i+1], pts))
        return farthest_pts, farthest_pts_idx


class kitti_pc_img_dataset(data.Dataset):
    def __init__(self, root_path, mode, num_pc,
                 P_tx_amplitude=5, P_ty_amplitude=5, P_tz_amplitude=5,
                 P_Rx_amplitude=0, P_Ry_amplitude=2.0 * math.pi, P_Rz_amplitude=0,num_kpt=512,is_front=False):
        super(kitti_pc_img_dataset, self).__init__()
        self.root_path = root_path
        self.mode = mode
        self.dataset = self.make_kitti_dataset(root_path, mode)
        self.calibhelper = KittiCalibHelper(root_path)
        self.num_pc = num_pc
        self.img_H = 160
        self.img_W = 512

        self.P_tx_amplitude = P_tx_amplitude
        self.P_ty_amplitude = P_ty_amplitude
        self.P_tz_amplitude = P_tz_amplitude
        self.P_Rx_amplitude = P_Rx_amplitude
        self.P_Ry_amplitude = P_Ry_amplitude
        self.P_Rz_amplitude = P_Rz_amplitude
        self.num_kpt=num_kpt
        self.farthest_sampler = FarthestSampler(dim=3)

        self.node_a_num=256
        self.node_b_num=256
        self.is_front=is_front
        print('load data complete')

    def read_velodyne_bin(self, path):

        pc_list = []
        with open(path, 'rb') as f:
            content = f.read()
            pc_iter = struct.iter_unpack('ffff', content)
            for idx, point in enumerate(pc_iter):
                pc_list.append([point[0], point[1], point[2], point[3]])
        return np.asarray(pc_list, dtype=np.float32).T

    def make_kitti_dataset(self, root_path, mode):
        dataset = []

        if mode == 'train':
            seq_list = list(range(9))
        elif 'val' == mode:
            seq_list = [9, 10]
        else:
            raise Exception('Invalid mode.')

        skip_start_end = 0
        for seq in seq_list:
            img2_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'img_P2')
            img3_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'img_P3')
            pc_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'pc_npy_with_normal')

            K2_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'K_P2')
            K3_folder = os.path.join(
                root_path, 'sequences', '%02d' % seq, 'K_P3')

            sample_num = round(len(os.listdir(img2_folder)))

            for i in range(skip_start_end, sample_num - skip_start_end):
                dataset.append((img2_folder, pc_folder,
                                K2_folder, seq, i, 'P2', sample_num))
                dataset.append((img3_folder, pc_folder,
                                K3_folder, seq, i, 'P3', sample_num))
        return dataset


    def downsample_with_intensity_sn(self, pointcloud, intensity, sn, voxel_grid_downsample_size):
        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(np.transpose(pointcloud))
        intensity_max=np.max(intensity)

        fake_colors=np.zeros((pointcloud.shape[1],3))
        fake_colors[:,0:1]=np.transpose(intensity)/intensity_max

        pcd.colors=o3d.utility.Vector3dVector(fake_colors)
        pcd.normals=o3d.utility.Vector3dVector(np.transpose(sn))

        down_pcd=pcd.voxel_down_sample(voxel_size=voxel_grid_downsample_size)
        down_pcd_points=np.transpose(np.asarray(down_pcd.points))
        pointcloud=down_pcd_points

        intensity=np.transpose(np.asarray(down_pcd.colors)[:,0:1])*intensity_max
        sn=np.transpose(np.asarray(down_pcd.normals))

        return pointcloud, intensity, sn

    def downsample_np(self, pc_np, intensity_np, sn_np):
        if pc_np.shape[1] >= self.num_pc:
            choice_idx = np.random.choice(pc_np.shape[1], self.num_pc, replace=False)
        else:
            fix_idx = np.asarray(range(pc_np.shape[1]))
            while pc_np.shape[1] + fix_idx.shape[0] < self.num_pc:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(pc_np.shape[1]))), axis=0)
            random_idx = np.random.choice(pc_np.shape[1], self.num_pc - fix_idx.shape[0], replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
        pc_np = pc_np[:, choice_idx]
        intensity_np = intensity_np[:, choice_idx]
        sn_np=sn_np[:,choice_idx]
        return pc_np, intensity_np, sn_np

    def camera_matrix_cropping(self, K: np.ndarray, dx: float, dy: float):
        K_crop = np.copy(K)
        K_crop[0, 2] -= dx
        K_crop[1, 2] -= dy
        return K_crop

    def camera_matrix_scaling(self, K: np.ndarray, s: float):
        K_scale = s * K
        K_scale[2, 2] = 1
        return K_scale

    def augment_img(self, img_np):
        brightness = (0.8, 1.2)
        contrast = (0.8, 1.2)
        saturation = (0.8, 1.2)
        hue = (-0.1, 0.1)
        color_aug = transforms.ColorJitter(
            brightness, contrast, saturation, hue)
        img_color_aug_np = np.array(color_aug(Image.fromarray(img_np)))

        return img_color_aug_np

    def angles2rotation_matrix(self, angles):
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

    def generate_random_transform(self):
        """
        :param pc_np: pc in NWU coordinate
        :return:
        """
        t = [random.uniform(-self.P_tx_amplitude, self.P_tx_amplitude),
             random.uniform(-self.P_ty_amplitude, self.P_ty_amplitude),
             random.uniform(-self.P_tz_amplitude, self.P_tz_amplitude)]
        angles = [random.uniform(-self.P_Rx_amplitude, self.P_Rx_amplitude),
                  random.uniform(-self.P_Ry_amplitude, self.P_Ry_amplitude),
                  random.uniform(-self.P_Rz_amplitude, self.P_Rz_amplitude)]

        rotation_mat = self.angles2rotation_matrix(angles)
        P_random = np.identity(4, dtype=np.float32)
        P_random[0:3, 0:3] = rotation_mat
        P_random[0:3, 3] = t

        # print('t',t)
        # print('angles',angles)

        return P_random

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_folder, pc_folder, K_folder, seq, seq_i, key, _ = self.dataset[index]
        img = np.load(os.path.join(img_folder, '%06d.npy' % seq_i))
        data = np.load(os.path.join(pc_folder, '%06d.npy' % seq_i))
        intensity = data[3:4, :]
        sn = data[4:, :]
        pc = data[0:3, :]


        P_Tr = np.dot(self.calibhelper.get_matrix(seq, key),
                      self.calibhelper.get_matrix(seq, 'Tr'))

        pc = np.dot(P_Tr[0:3, 0:3], pc) + P_Tr[0:3, 3:]
        sn = np.dot(P_Tr[0:3, 0:3], sn)



        K = np.load(os.path.join(K_folder, '%06d.npy' % seq_i))

        pc, intensity, sn = self.downsample_with_intensity_sn(pc, intensity, sn, voxel_grid_downsample_size=0.1)

        pc, intensity, sn = self.downsample_np(pc, intensity,sn)

        img = cv2.resize(img,
                         (int(round(img.shape[1] * 0.5)),
                          int(round((img.shape[0] * 0.5)))),
                         interpolation=cv2.INTER_LINEAR)
        K = self.camera_matrix_scaling(K, 0.5)

        if 'train' == self.mode:
            img_crop_dx = random.randint(0, img.shape[1] - self.img_W)
            img_crop_dy = random.randint(0, img.shape[0] - self.img_H)
        else:
            img_crop_dx = int((img.shape[1] - self.img_W) / 2)
            img_crop_dy = int((img.shape[0] - self.img_H) / 2)
        img = img[img_crop_dy:img_crop_dy + self.img_H,
              img_crop_dx:img_crop_dx + self.img_W, :]
        K = self.camera_matrix_cropping(K, dx=img_crop_dx, dy=img_crop_dy)


        #1/4 scale
        K_4=self.camera_matrix_scaling(K,0.25)

        if 'train' == self.mode:
            img = self.augment_img(img)

        
        pc_ = np.dot(K_4, pc)
        pc_mask = np.zeros((1, np.shape(pc)[1]), dtype=np.float32)
        pc_[0:2, :] = pc_[0:2, :] / pc_[2:, :]
        xy = np.floor(pc_[0:2, :])
        is_in_picture = (xy[0, :] >= 0) & (xy[0, :] <= (self.img_W*0.25 - 1)) & (xy[1, :] >= 0) & (xy[1, :] <= (self.img_H*0.25 - 1)) & (pc_[2, :] > 0)
        pc_mask[:, is_in_picture] = 1.

        pc_kpt_idx=np.where(pc_mask.squeeze()==1)[0]
        index=np.random.permutation(len(pc_kpt_idx))[0:self.num_kpt]
        pc_kpt_idx=pc_kpt_idx[index]

        pc_outline_idx=np.where(pc_mask.squeeze()==0)[0]
        index=np.random.permutation(len(pc_outline_idx))[0:self.num_kpt]
        pc_outline_idx=pc_outline_idx[index]

        xy2 = xy[:, is_in_picture]
        img_mask = coo_matrix((np.ones_like(xy2[0, :]), (xy2[1, :], xy2[0, :])), shape=(int(self.img_H*0.25), int(self.img_W*0.25))).toarray()
        img_mask = np.array(img_mask)
        img_mask[img_mask > 0] = 1.

        img_kpt_index=xy[1,pc_kpt_idx]*self.img_W*0.25 +xy[0,pc_kpt_idx]


        img_outline_index=np.where(img_mask.squeeze().reshape(-1)==0)[0]
        index=np.random.permutation(len(img_outline_index))[0:self.num_kpt]
        img_outline_index=img_outline_index[index]

        P = self.generate_random_transform()

        pc = np.dot(P[0:3, 0:3], pc) + P[0:3, 3:]

        sn = np.dot(P[0:3, 0:3], sn)

        node_a_np, _ = self.farthest_sampler.sample(pc[:, np.random.choice( pc.shape[1],
                                                                            self.node_a_num * 8,
                                                                            replace=False)],
                                                                            k=self.node_a_num)

        node_b_np, _ = self.farthest_sampler.sample(pc[:, np.random.choice( pc.shape[1],
                                                                            self.node_b_num * 8,
                                                                            replace=False)],
                                                                            k=self.node_b_num)

        return {'img': torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1).contiguous(),
                'pc': torch.from_numpy(pc.astype(np.float32)),
                'intensity': torch.from_numpy(intensity.astype(np.float32)),
                'sn': torch.from_numpy(sn.astype(np.float32)),
                'K': torch.from_numpy(K_4.astype(np.float32)),
                'P': torch.from_numpy(np.linalg.inv(P).astype(np.float32)),

                'pc_mask': torch.from_numpy(pc_mask).float(),       #(1,20480)
                'img_mask': torch.from_numpy(img_mask).float(),     #(40,128)
                
                'pc_kpt_idx': torch.from_numpy(pc_kpt_idx),         #512
                'pc_outline_idx':torch.from_numpy(pc_outline_idx),  #512
                'img_kpt_idx':torch.from_numpy(img_kpt_index).long() ,      #512
                'img_outline_index':torch.from_numpy(img_outline_index).long(),
                'node_a':torch.from_numpy(node_a_np).float(),
                'node_b':torch.from_numpy(node_b_np).float()
                }
               


if __name__ == '__main__':
    dataset = kitti_pc_img_dataset('/gpfs1/scratch/siyuren2/dataset/', 'val', 20480)
    data = dataset[4000]
    
    '''img=data[0].numpy()              #full size
    pc=data[1].numpy()
    intensity=data[2].numpy()
    sn=data[3].numpy()
    K=data[4].numpy()
    P=data[5].numpy()
    pc_mask=data[6].numpy()      
    img_mask=data[7].numpy()    #1/4 size

    pc_kpt_idx=data[8].numpy()                #(B,512)
    pc_outline_idx=data[9].numpy()
    img_kpt_idx=data[10].numpy()
    img_outline_idx=data[11].numpy()

    np.save('./test_data/img.npy',img)
    np.save('./test_data/pc.npy',pc)
    np.save('./test_data/intensity.npy',intensity)
    np.save('./test_data/sn.npy',sn)
    np.save('./test_data/K.npy',K)
    np.save('./test_data/P.npy',P)
    np.save('./test_data/pc_mask.npy',pc_mask)
    np.save('./test_data/img_mask.npy',img_mask)
    '''



    '''for i,data in enumerate(dataset):
        print(i,data['pc'].size())'''

    print(len(dataset))
    print(data['pc'].size())
    print(data['img'].size())
    print(data['pc_mask'].size())
    print(data['intensity'].size())
    np.save('./test_data/pc.npy', data['pc'].numpy())
    np.save('./test_data/P.npy', data['P'].numpy())
    np.save('./test_data/img.npy', data['img'].numpy())
    np.save('./test_data/img_mask.npy', data['img_mask'].numpy())
    np.save('./test_data/pc_mask.npy', data['pc_mask'].numpy())
    np.save('./test_data/K.npy', data['K'].numpy())
    

    """
    img = dict['img'].numpy()
    img_mask = dict['img_mask'].numpy()
    img = img.transpose(1, 2, 0)
    cv2.imwrite('img.png',np.uint8(img*255))
    cv2.imwrite('img_mask.png', np.uint8(img_mask * 255))
    cv2.imshow('img', cv2.resize(img,(512,160)))
    cv2.imshow('img_mask', cv2.resize(img_mask,(512,160)))

    color = []
    for i in range(np.shape(pc_data)[1]):
        if pc_mask[0, i] > 0:
            color.append([0, 1, 1])
        else:
            color.append([0, 0, 1])
    color = np.asarray(color, dtype=np.float64)
    print(color.shape)

    print(np.sum(pc_mask), np.sum(img_mask))
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pc_data.T)
    pointcloud.colors = o3d.utility.Vector3dVector(color)

    o3d.visualization.draw_geometries([pointcloud])
    # plt.imshow(dict['img'].permute(1,2,0).numpy())
    # plt.show()"""
