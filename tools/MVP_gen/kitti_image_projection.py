import numpy as np
import pickle
import torch
import os
import matplotlib.pyplot as plt


def to_tensor(x, device='cuda:0', dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, device=device)


def get_obj(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def get_data_list(path):
    set_path = os.path.join(path, 'ImageSets', 'train.txt')
    with open(set_path, 'r') as f:
        data_str = f.read()
        data_list = data_str.split('\n')

    image_list = [d + '.png' for d in data_list]
    point_list = [d + '.bin' for d in data_list]
    label_list = [d + '.txt' for d in data_list]
    calib_list = [d + '.txt' for d in data_list]
    return image_list, point_list, label_list, calib_list

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        data_str = f.read()
        line_list = data_str.split('\n')

    label_dict = {}
    for l in line_list:
        if l != '':
            label_list = l.split(' ')
            key = label_list[0]
            value_list = label_list[1:]
            value_list = [float(v) for v in value_list if v]
            label_dict[key] = value_list
            print(label_dict)

def to_batch_tensor(tensor, device='cuda:0', dtype=torch.float32):
    return torch.stack([to_tensor(x, device=device, dtype=dtype) for x in tensor], dim=0)


def batch_view_points(points, view, normalize, device='cuda:0'):
    # points: batch x 3 x N
    # view: batch x 3 x 3
    batch_size, _, nbr_points = points.shape

    viewpad = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    viewpad[:, :view.shape[1], :view.shape[2]] = view

    points = torch.cat((points, torch.ones([batch_size, 1, nbr_points], device=device)), dim=1)

    # (6 x 4 x 4) x (6 x 4 x N)   -> 6 x 4 x N
    points = torch.bmm(viewpad, points)
    # points = torch.einsum('abc,def->abd', viewpad, points)

    points = points[:, :3]

    if normalize:
        # 6 x 1 x N
        points = points / points[:, 2:3].repeat(1, 3, 1)

    return points


def reverse_view_points(points, depths, view, device='cuda:0'):
    # TODO: write test for reverse projection
    nbr_points = points.shape[1]
    points = points * depths.repeat(3, 1).reshape(3, nbr_points)

    points = torch.cat((points, torch.ones([1, nbr_points]).to(device)), dim=0)

    viewpad = torch.eye(4).to(device)
    viewpad[:view.shape[0], :view.shape[1]] = torch.inverse(view)

    points = torch.matmul(viewpad, points)
    points = points[:3, :]

    return points


def read_file(path, num_point_feature=4):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


@torch.no_grad()
def projectionV2(points, cams_from_lidar, cams_intrinsic, H=900, W=1600, device='cuda:0'):
    num_lidar_point = points.shape[0]
    projected_points = torch.zeros((points.shape[0], 4), device=device)

    # 因为外参矩阵是4*4的，所以点云坐标需要补全一个全为1的维度
    point_padded = torch.cat([
        points.transpose(1, 0)[:3, :],
        torch.ones(1, num_lidar_point, dtype=points.dtype, device=device)
    ], dim=0)

    # (4 x 4) x (4 x N) 雷达坐标到相机坐标的转换
    transform_points = torch.einsum('bc,cd->bd', cams_from_lidar, point_padded)[:3, :]
    depths = transform_points[2, :]

    # (4 x 4) x (4 x N) 相机坐标到图像平面的转换
    transform_points_padded = torch.cat([
        transform_points,
        torch.ones([1, num_lidar_point], dtype=transform_points.dtype, device=device)
    ], dim=0)
    points_2d = torch.einsum('bc,cd->bd', cams_intrinsic, transform_points_padded)[:3, :]
    points_2d = torch.floor(points_2d)

    points_x, points_y = points_2d[0, :].long(), points_2d[1, :].long()

    valid_mask = (points_x > 0) & (points_x < W) & (points_y > 0) & (points_y < H) & (depths > 0)

    valid_projected_points = projected_points[valid_mask]

    valid_projected_points[:, :2] = points_2d[valid_mask]
    valid_projected_points[:, 2] = depths[valid_mask]
    valid_projected_points[:, 3] = 1  # indicate that there is a valid projection

    projected_points[valid_mask] = valid_projected_points
    return


def projectionV3(points, P2, R0_rect, Tr_velo_to_cam, instance_mask_list, image, IMG_H, IMG_W):
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

    raw_points = points[:, 0:3]  # lidar xyz (front, left, up)
    velo_points = np.insert(raw_points, 3, 1, axis=1).T

    velo_valid_mask = velo_points[0, :] >= 0
    velo_points = np.delete(velo_points, np.where(velo_points[0, :] < 0), axis=1)
    cam_points = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo_points)))
    deep_valid_mask = cam_points[2, :] >= 0
    cam_points = np.delete(cam_points, np.where(cam_points[2, :] < 0), axis=1)

    # get u,v,z
    cam_points[:2] /= cam_points[2, :]
    u, v, z = cam_points
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    range_valid_mask = np.logical_or(u_out, v_out)
    valid_points = np.delete(cam_points, np.where(range_valid_mask), axis=1)
    u, v, z = valid_points
    # plt.axis([0, IMG_W, IMG_H, 0])
    # plt.imshow(image)
    # plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
    # plt.title('projection')
    # plt.show()

    instance_point_list = []
    for im in instance_mask_list:
        ins_label = np.max(im)
        instance_point_mask = im[v.astype(int), u.astype(int)] == ins_label
        instance_point = points[velo_valid_mask]
        instance_point = instance_point[deep_valid_mask]
        instance_point = instance_point[~range_valid_mask]
        instance_point = instance_point[instance_point_mask]
        # add label info
        instance_point = np.insert(instance_point, 4, ins_label, axis=1)
        instance_point_list.append(instance_point)
        # plt.axis([0, IMG_W, IMG_H, 0])
        # plt.imshow(image)
        # u, v, z = valid_points[:, instance_point_mask]
        # plt.scatter([u.astype(int)], [v.astype(int)], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
        # plt.title('instance projection')
        # plt.show()
    return instance_point_list


if __name__ == '__main__':
    # get_data_list('/data/szy4017/code/OpenPCDet/data/kitti')
    # read_txt_file('/data/szy4017/code/OpenPCDet/data/kitti/training/label_2/000000.txt')
    import sys
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import numpy as np

    sn = int(sys.argv[1]) if len(sys.argv) > 1 else 7  # default 0-7517
    name = '%06d' % sn  # 6 digit zeropadding
    img = f'/data/szy4017/code/OpenPCDet/data/kitti/testing/image_2/{name}.png'
    binary = f'/data/szy4017/code/OpenPCDet/data/kitti/testing/velodyne/{name}.bin'
    with open(f'/data/szy4017/code/OpenPCDet/data/kitti/testing/calib/{name}.txt', 'r') as f:
        calib = f.readlines()

    # P2 (3 x 4) for left eye
    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
    R0_rect = np.array([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    Tr_velo_to_cam = np.array([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

    # read raw data from binary
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]  # lidar xyz (front, left, up)
    # TODO: use fov filter?
    velo = np.insert(points, 3, 1, axis=1).T
    # velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
    cam = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(velo))).T
    cam = np.delete(cam, np.where(cam[2, :] < 0), axis=1)
    # get u,v,z
    cam[:2] /= cam[2, :]
    # do projection staff
    plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
    png = mpimg.imread(img)
    IMG_H, IMG_W, _ = png.shape
    # restrict canvas in range
    plt.axis([0, IMG_W, IMG_H, 0])
    plt.imshow(png)
    # filter point out of canvas
    u, v, z = cam
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam, np.where(outlier), axis=1)
    # generate color map from depth
    u, v, z = cam
    plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
    plt.title(name)
    plt.savefig(f'.{name}.png', bbox_inches='tight')
    plt.show()