"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import json
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

center_colormap = [
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
]

def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(background, points, gt_boxes=None, ref_boxes=None, ref_centers=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    # if isinstance(points, torch.Tensor):
    #     points = points.cpu().numpy()
    # if isinstance(gt_boxes, torch.Tensor):
    #     gt_boxes = gt_boxes.cpu().numpy()
    # if isinstance(ref_boxes, torch.Tensor):
    #     ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    if background is not None:
        bgd_pts = open3d.geometry.PointCloud()
        bgd_pts.points = open3d.utility.Vector3dVector(background[:, :3])
        bgd_pts.colors = open3d.utility.Vector3dVector(np.ones((background.shape[0], 3)))
        vis.add_geometry(bgd_pts)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(np.repeat(point_colors, points.shape[0], axis=0))

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

        ref_centers = ref_boxes[:, 0:3]
        vis = draw_center(vis, ref_centers, ref_labels)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis


def draw_center(vis, center, ref_labels):
    for i in range(center.shape[0]):
        center3d = open3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=30)
        center3d.compute_vertex_normals()
        center_color = center_colormap[ref_labels[i]]
        center3d.paint_uniform_color(center_color)
        center3d.translate(center[i])
        vis.add_geometry(center3d)
    return vis


if __name__ == '__main__':
    draw_json_file = './demo_json/CenterPoint-KittiDataset-ID-0.json'
    draw_instance_json_file = './demo_json/CenterPoint-KittiDataset-ID-0_instance.json'
    with open(draw_json_file, 'r') as draw_file:
        draw_data = json.load(draw_file)
    with open(draw_instance_json_file, 'r') as draw_file:
        draw_instance_data = json.load(draw_file)
    draw_scenes(background=np.array(draw_data['points']),
                points=np.array(draw_instance_data['points']), ref_boxes=np.array(draw_instance_data['ref_boxes']),
                ref_scores=np.array(draw_instance_data['ref_scores']), ref_labels=np.array(draw_instance_data['ref_labels']),
                point_colors=[[0, 1, 0]]
    )
    # draw_scenes(background=None,
    #             points=np.array(draw_instance_data['points']), ref_boxes=np.array(draw_instance_data['ref_boxes']),
    #             ref_scores=np.array(draw_instance_data['ref_scores']), ref_labels=np.array(draw_instance_data['ref_labels']),
    #             point_colors=[[1, 1, 1]]
    # )