from kitti_image_projection import read_file, to_batch_tensor, to_tensor, projectionV3, reverse_view_points, get_obj, get_data_list
from detector import detector_inference
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import os

H = 900
W = 1600


class DataSet(Dataset):
    def __init__(self, root_path):
        '''
        :param root_path: 数据集根目录
        '''
        self.image_list, self.point_list, self.label_list, self.calib_list = get_data_list(root_path)
        self.image_root = os.path.join(root_path, 'training', 'image_2')
        self.point_root = os.path.join(root_path, 'training', 'velodyne')
        self.label_root = os.path.join(root_path, 'training', 'label_2')
        self.calib_root = os.path.join(root_path, 'training', 'calib')
        self.len = len(self.image_list)

    @torch.no_grad()
    def __getitem__(self, index):
        image_name = self.image_list[index]
        point_name = self.point_list[index]
        label_name = self.label_list[index]
        calib_name = self.calib_list[index]

        image, height, width = self.read_png_file(os.path.join(self.image_root, image_name))
        point = self.read_bin_file(os.path.join(self.point_root, point_name))
        label = self.read_txt_file(os.path.join(self.label_root, label_name), flag='label')
        calib = self.read_txt_file(os.path.join(self.calib_root, calib_name), flag='calib')

        all_data = {'image': image, 'height': height, 'width': width,
                    'point': point, 'label': label, 'calib': calib,
                    'tag': point_name}

        return all_data

    def __len__(self):
        return len(self.image_list)

    def read_png_file(self, file_path):
        img = cv2.imread(file_path)
        img = img[:, :, ::-1]   # convert BGR to RGB
        h, w = img.shape[:2]
        return img, h, w

    def read_bin_file(self, file_path):
        scan = np.fromfile(file_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def read_txt_file(self, file_path, flag):
        with open(file_path, 'r') as f:
            data_str = f.read()
            line_list = data_str.split('\n')

        if flag == 'calib':
            calib_dict = {}
            for l in line_list:
                if l != '':
                    key, value = l.split(':')
                    value_list = value.split(' ')
                    value_list = [float(v) for v in value_list if v]
                    calib_dict[key] = value_list
            return calib_dict
        elif flag == 'label':
            label_dict = {}
            for l in line_list:
                if l != '':
                    label_list = l.split(' ')
                    key = label_list[0]
                    value_list = label_list[1:]
                    value_list = [float(v) for v in value_list if v]
                    label_dict[key] = value_list
                    return label_dict
        else:
            raise Exception("Wrong flag!")



def is_within_mask(points_xyc, masks, H=900, W=1600):
    seg_mask = masks[:, :-1].reshape(-1, W, H)
    camera_id = masks[:, -1]
    points_xyc = points_xyc.long()
    valid = seg_mask[:, points_xyc[:, 0], points_xyc[:, 1]] * (camera_id[:, None] == points_xyc[:, -1][None])
    return valid.transpose(1, 0)


@torch.no_grad()
def add_virtual_mask(masks, labels, points, raw_points, num_virtual=50, dist_thresh=3000, num_camera=6, intrinsics=None,
                     transforms=None):
    points_xyc = points.reshape(-1, 5)[:, [0, 1, 4]]  # x, y, z, valid_indicator, camera id

    valid = is_within_mask(points_xyc, masks)
    valid = valid * points.reshape(-1, 5)[:, 3:4]

    # remove camera id from masks
    camera_ids = masks[:, -1]
    masks = masks[:, :-1]

    box_to_label_mapping = torch.argmax(valid.float(), dim=1).reshape(-1, 1).repeat(1, 11)
    point_labels = labels.gather(0, box_to_label_mapping)
    point_labels *= (valid.sum(dim=1, keepdim=True) > 0)

    foreground_real_point_mask = (valid.sum(dim=1, keepdim=True) > 0).reshape(num_camera, -1).sum(dim=0).bool()

    offsets = []
    for mask in masks:
        indices = mask.reshape(W, H).nonzero()
        selected_indices = torch.randperm(len(indices), device=masks.device)[:num_virtual]
        if len(selected_indices) < num_virtual:
            selected_indices = torch.cat([selected_indices, selected_indices[
                selected_indices.new_zeros(num_virtual - len(selected_indices))]])

        offset = indices[selected_indices]
        offsets.append(offset)

    offsets = torch.stack(offsets, dim=0)
    virtual_point_instance_ids = torch.arange(1, 1 + masks.shape[0],
                                              dtype=torch.float32, device='cuda:0').reshape(masks.shape[0], 1,
                                                                                            1).repeat(1, num_virtual, 1)

    virtual_points = torch.cat([offsets, virtual_point_instance_ids], dim=-1).reshape(-1, 3)
    virtual_point_camera_ids = camera_ids.reshape(-1, 1, 1).repeat(1, num_virtual, 1).reshape(-1, 1)

    valid_mask = valid.sum(dim=1) > 0
    real_point_instance_ids = (torch.argmax(valid.float(), dim=1) + 1)[valid_mask]
    real_points = torch.cat([points_xyc[:, :2][valid_mask], real_point_instance_ids[..., None]], dim=-1)

    # avoid matching across instances
    real_points[:, -1] *= 1e4
    virtual_points[:, -1] *= 1e4

    if len(real_points) == 0:
        return None

    dist = torch.norm(virtual_points.unsqueeze(1) - real_points.unsqueeze(0), dim=-1)
    nearest_dist, nearest_indices = torch.min(dist, dim=1)
    mask = nearest_dist < dist_thresh

    indices = valid_mask.nonzero(as_tuple=False).reshape(-1)

    nearest_indices = indices[nearest_indices[mask]]
    virtual_points = virtual_points[mask]
    virtual_point_camera_ids = virtual_point_camera_ids[mask]
    all_virtual_points = []
    all_real_points = []
    all_point_labels = []

    for i in range(num_camera):
        camera_mask = (virtual_point_camera_ids == i).squeeze()
        per_camera_virtual_points = virtual_points[camera_mask]
        per_camera_indices = nearest_indices[camera_mask]
        per_camera_virtual_points_depth = points.reshape(-1, 5)[per_camera_indices, 2].reshape(1, -1)

        per_camera_virtual_points = per_camera_virtual_points[:, :2]  # remove instance id
        per_camera_virtual_points_padded = torch.cat(
            [per_camera_virtual_points.transpose(1, 0).float(),
             torch.ones((1, len(per_camera_virtual_points)), device=per_camera_indices.device, dtype=torch.float32)],
            dim=0
        )
        per_camera_virtual_points_3d = reverse_view_points(per_camera_virtual_points_padded,
                                                           per_camera_virtual_points_depth, intrinsics[i])

        per_camera_virtual_points_3d[:3] = torch.matmul(torch.inverse(transforms[i]),
                                                        torch.cat([
                                                            per_camera_virtual_points_3d[:3, :],
                                                            torch.ones(1, per_camera_virtual_points_3d.shape[1],
                                                                       dtype=torch.float32,
                                                                       device=per_camera_indices.device)
                                                        ], dim=0)
                                                        )[:3]

        all_virtual_points.append(per_camera_virtual_points_3d.transpose(1, 0))
        all_real_points.append(
            raw_points.reshape(1, -1, 4).repeat(num_camera, 1, 1).reshape(-1, 4)[per_camera_indices][:, :3])
        all_point_labels.append(point_labels[per_camera_indices])

    all_virtual_points = torch.cat(all_virtual_points, dim=0)
    all_real_points = torch.cat(all_real_points, dim=0)
    all_point_labels = torch.cat(all_point_labels, dim=0)

    all_virtual_points = torch.cat([all_virtual_points, all_point_labels], dim=1)

    real_point_labels = point_labels.reshape(num_camera, raw_points.shape[0], -1)
    real_point_labels = torch.max(real_point_labels, dim=0)[0]

    all_real_points = torch.cat(
        [raw_points[foreground_real_point_mask.bool()], real_point_labels[foreground_real_point_mask.bool()]], dim=1)

    return all_virtual_points, all_real_points, foreground_real_point_mask.bool().nonzero(as_tuple=False).reshape(-1)


# def init_detector(args):
#     from CenterNet2.train_net import setup
#     from detectron2.engine import DefaultPredictor
#
#     cfg = setup(args)
#     predictor = DefaultPredictor(cfg)
#     return predictor


def postprocess(res):
    result = res['instances']
    labels = result.pred_classes
    scores = result.scores
    masks = result.pred_masks.reshape(scores.shape[0], 1600 * 900)
    boxes = result.pred_boxes.tensor

    # remove empty mask and their scores / labels
    empty_mask = masks.sum(dim=1) == 0

    labels = labels[~empty_mask]
    scores = scores[~empty_mask]
    masks = masks[~empty_mask]
    boxes = boxes[~empty_mask]
    masks = masks.reshape(-1, 900, 1600).permute(0, 2, 1).reshape(-1, 1600 * 900)
    return labels, scores, masks


def exter_mat_transform(Tr_velo_to_cam, R0_rect):
    tr_mat = np.array(Tr_velo_to_cam).reshape(3, 4)
    r_mat = np.array(R0_rect).reshape(3, 3)
    mask = np.diag([1.0, 1.0, 1.0, 1.0])
    mask[:-1, :] = tr_mat
    tr_mat_expand = mask
    mask = np.diag([1.0, 1.0, 1.0, 1.0])
    mask[:-1, :-1] = r_mat
    r_mat_expand = mask
    exter_mat = np.dot(r_mat_expand, tr_mat_expand)
    return exter_mat


def inter_mat_transform(P2):
    p_mat = np.array(P2).reshape(3, 4)
    mask = np.diag([1.0, 1.0, 1.0, 1.0])
    mask[:-1, :] = p_mat
    inter_mat = mask
    return inter_mat


@torch.no_grad()
def process_one_frame(data):
    calib = data['calib']
    lidar_point = data['point']
    image = data['image']
    height = data['height']
    width = data['width']

    Tr_velo_to_cam = calib['Tr_velo_to_cam']
    R0_rect = calib['R0_rect']
    P2 = calib['P2']
    tr_mat = np.array(Tr_velo_to_cam).reshape(3, 4)
    r0_mat = np.array(R0_rect).reshape(3, 3)
    p2_mat = np.array(P2).reshape(3, 4)

    one_hot_labels = []
    for i in range(3):
        one_hot_label = torch.zeros(3, device='cuda:0', dtype=torch.float32)
        one_hot_label[i] = 1
        one_hot_labels.append(one_hot_label)

    one_hot_labels = torch.stack(one_hot_labels, dim=0)

    masks = []
    labels = []

    img_bbox_res = detector_inference.inference(image)
    bbox_instance_list = []
    for ki, k in enumerate(img_bbox_res.keys()):
        if img_bbox_res[k] is not None:
            for vi, v in enumerate(img_bbox_res[k]):
                bbox_instance = v[:-1]
                bbox_instance = np.append(bbox_instance, ki+1)
                bbox_instance_list.append(bbox_instance)
    print(bbox_instance_list)

    instance_mask_list = []
    for ins in bbox_instance_list:
        ins_label = ins[-1]
        ins_box = ins[:-1]
        ins_mask = np.zeros((height, width))
        # plt.imshow(ins_mask)
        # plt.show()
        ins_mask[int(ins_box[1]): int(ins_box[3]), int(ins_box[0]): int(ins_box[2])] = ins_label
        plt.imshow(ins_mask)
        # plt.show()
        instance_mask_list.append(ins_mask)

    instance_point_list = projectionV3(lidar_point, p2_mat, r0_mat, tr_mat, instance_mask_list, image, height, width)

    if instance_point_list != []:
        return lidar_point, instance_point_list
    else:
        return lidar_point, None

def simple_collate(batch_list):
    assert len(batch_list) == 1
    batch_list = batch_list[0]
    return batch_list


def main(args):
    # predictor = init_detector(args)
    detector = None
    data_loader = DataLoader(
        DataSet(args.root_path),
        batch_size=1,
        num_workers=8,
        collate_fn=simple_collate,
        pin_memory=False,
        shuffle=False
    )

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        if len(data) == 0:
            continue
        tokens = data['tag']
        output_path = os.path.join('instance_pcl', tokens)
        print(tokens)

        raw_pcl, instance_pcl_list = process_one_frame(data)

        raw_pcl.tofile('raw_pcl_' + tokens)
        instance_pcl = np.concatenate(instance_pcl_list)
        instance_pcl = instance_pcl[:, :-1]
        instance_pcl.tofile('instance_pcl_' + tokens)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--root_path', type=str, default='/data/szy4017/code/OpenPCDet/data/kitti')
    parser.add_argument('--config-file', type=str, default='c2_config/nuImages_CenterNet2_DLA_640_8x.yaml')
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    main(args)