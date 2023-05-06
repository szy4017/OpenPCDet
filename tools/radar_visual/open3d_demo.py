import open3d as o3d
import numpy as np
import json
import matplotlib.pyplot as plt
from io import BytesIO


def read_radar_json(json_file='./radar_test_2/50-300_meters.json'):
    radar_point_list = []
    radar_object_list = []
    with open(json_file, 'r', encoding='utf-8') as radar_file:
        for radar_str in radar_file.readlines():
            radar_data = json.loads(radar_str)
            # print(radar_data)
            if radar_data['__type__'] == 'DetectionList':
                frame_point_list = []
                for detection in radar_data['List_Detections']:
                    # print(detection)
                    det_Range = detection['f_Range']    # 获取检测点径向距离Range
                    det_RangeRate = detection['f_RangeRate']    # 获取目标射程率RangeRate，用于计算速度
                    det_AzimuthAngle = detection['f_AzimuthAngle']  # 获取检测点方位角AzimuthAngle
                    det_ElevationAngle = detection['f_ElevationAngle']  # 获取检测点仰角ElevationAngle
                    det_RCS = detection['s_RCS']    # 获取检测点反射强度RCS
                    # if det_Range >= 100.0:
                    #     continue
                    # print('det_Range: ', det_Range)
                    # print('det_AzimuthAngle: ', det_AzimuthAngle)
                    # print('det_ElevationAngle: ', det_ElevationAngle)
                    # print('det_RCS: ' , det_RCS)
                    det_X = np.sin(det_AzimuthAngle) * np.cos(det_ElevationAngle) * det_Range
                    det_Y = np.cos(det_AzimuthAngle) * np.cos(det_ElevationAngle) * det_Range
                    det_Height = np.sin(det_ElevationAngle) * det_Range
                    det_V = np.cos(det_AzimuthAngle) * np.cos(det_ElevationAngle) * det_RangeRate
                    print('det_Y: ', det_Y)
                    print('det_X: ', det_X)
                    print('det_Height: ', det_Height)
                    print('det_V: ', det_V)
                    print('det_RCS: ', det_RCS)
                    frame_point_list.append(np.array([det_X, det_Y, det_Height, det_RCS]))
                print('one frame points num: ', len(frame_point_list))
                print('---------------')
                if len(frame_point_list) != 0:
                    radar_point_list.append(frame_point_list)
            if radar_data['__type__'] == 'ObjectList':
                frame_object_list = []
                for object in radar_data['ObjectList_Objects']:
                    obj_X = object['f_Position_Y']
                    obj_Y = object['f_Position_X']
                    obj_Height = object['f_Position_Z']
                    obj_V = object['f_Dynamics_RelVel_X']
                    print('obj_X: ', obj_X)
                    print('obj_Y: ', obj_Y)
                    print('obj_Height: ', obj_Height)
                    print('obj_V: ', obj_V)
                    frame_object_list.append(np.array([obj_X, obj_Y, obj_Height]))
                print('one frame objects num: ', len(frame_object_list))
                print('---------------')
                radar_object_list.append(frame_object_list)
            print('################')
        print('frame num: ', len(radar_point_list))
    return [radar_point_list, radar_object_list]


def heatmap():
    data = np.arange(0, 100)
    data = np.reshape(data, (10, 10))
    # plt.imshow(data, cmap='hot')
    cmap = plt.get_cmap('hot')
    # plt.show()

    color_list = []
    for i in range(100):
        color = cmap.__call__(i)
        color_list.append(color[:3])
    return color_list

def main():
    raw_point = read_radar_json()[0]
    raw_point = np.vstack(raw_point)
    # raw_point = np.load('1.npy')  # 读取1.npy数据  N*[x,y,z]
    # raw_point = np.array([[10, 10, 10], [30, 30, 30], [100, 100, 100]])
    # raw_point = [np.array([10, 10, 10]), np.array([30, 30, 30]), np.array([100, 100, 100])]

    # 创建窗口对象
    vis = o3d.visualization.Visualizer()
    # 设置窗口标题
    vis.create_window(window_name="radar")
    # 设置点云大小
    vis.get_render_option().point_size = 5
    # 设置颜色背景为黑色
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    # 设置坐标系
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=(0, 0, 0))
    # 设置网格线
    polygon_points = np.array([[1, 0, 1], [1, 0, -1], [-1, 0, -1], [-1, 0, 1]])
    lines_connect = [[0, 1], [1, 2], [2, 3], [3, 0]]
    lines_color = [[0, 1, 1] for i in range(len(lines_connect))]
    line1_pcd = o3d.geometry.LineSet()
    line1_pcd.lines = o3d.utility.Vector2iVector(lines_connect)
    line1_pcd.colors = o3d.utility.Vector3dVector(lines_color)
    line1_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    polygon_points = np.array([[-10, 0, 0], [-10, 10, 0], [10, 10, 0], [10, 0, 0]])
    lines_connect = [[0, 1], [1, 2], [2, 3], [3, 0]]
    lines_color = [[0, 1, 1] for i in range(len(lines_connect))]
    line2_pcd = o3d.geometry.LineSet()
    line2_pcd.lines = o3d.utility.Vector2iVector(lines_connect)
    line2_pcd.colors = o3d.utility.Vector3dVector(lines_color)
    line2_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    # polygon_points = np.array([[-10, 0, 0], [-10, 20, 0], [10, 20, 0], [10, 0, 0]])
    # lines_connect = [[0, 1], [1, 2], [2, 3], [3, 0]]
    # lines_color = [[0, 1, 1] for i in range(len(lines_connect))]
    # line3_pcd = o3d.geometry.LineSet()
    # line3_pcd.lines = o3d.utility.Vector2iVector(lines_connect)
    # line3_pcd.colors = o3d.utility.Vector3dVector(lines_color)
    # line3_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    polygon_points = np.array([[-10, 0, 0], [-10, 50, 0], [10, 50, 0], [10, 0, 0]])
    lines_connect = [[0, 1], [1, 2], [2, 3], [3, 0]]
    lines_color = [[0, 1, 1] for i in range(len(lines_connect))]
    line4_pcd = o3d.geometry.LineSet()
    line4_pcd.lines = o3d.utility.Vector2iVector(lines_connect)
    line4_pcd.colors = o3d.utility.Vector3dVector(lines_color)
    line4_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    polygon_points = np.array([[-10, 0, 0], [-10, 100, 0], [10, 100, 0], [10, 0, 0]])
    lines_connect = [[0, 1], [1, 2], [2, 3], [3, 0]]
    lines_color = [[0, 1, 1] for i in range(len(lines_connect))]
    line5_pcd = o3d.geometry.LineSet()
    line5_pcd.lines = o3d.utility.Vector2iVector(lines_connect)
    line5_pcd.colors = o3d.utility.Vector3dVector(lines_color)
    line5_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    polygon_points = np.array([[-10, 0, 0], [-10, 150, 0], [10, 150, 0], [10, 0, 0]])
    lines_connect = [[0, 1], [1, 2], [2, 3], [3, 0]]
    lines_color = [[0, 1, 1] for i in range(len(lines_connect))]
    line6_pcd = o3d.geometry.LineSet()
    line6_pcd.lines = o3d.utility.Vector2iVector(lines_connect)
    line6_pcd.colors = o3d.utility.Vector3dVector(lines_color)
    line6_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    # 创建点云对象
    pcd = o3d.open3d.geometry.PointCloud()
    # 将点云数据转换为Open3d可以直接使用的数据类型
    pcd.points = o3d.open3d.utility.Vector3dVector(raw_point[:, :3])
    # 设置点的颜色
    color_array = np.array(heatmap())[:, :, None]
    pcd_colors = np.repeat(color_array, raw_point.shape[0], axis=-1)
    pcd_colors = np.transpose(pcd_colors, axes=(0, 2, 1))
    color_indices = abs(raw_point[:, -1])
    color_indices = (((color_indices - color_indices.min()) / (color_indices.max() - color_indices.min())) * 99).astype(np.uint).tolist()
    pcd_colors = pcd_colors[color_indices, list(range(0, len(color_indices))), :]
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    # pcd.paint_uniform_color([1, 1, 1])
    # 将点云加入到窗口中
    vis.add_geometry(pcd)
    # 将坐标系加入窗口中
    vis.add_geometry(axis_pcd)
    # 将网格线加入窗口中
    vis.add_geometry(line1_pcd)
    vis.add_geometry(line2_pcd)
    # vis.add_geometry(line3_pcd)
    vis.add_geometry(line4_pcd)
    vis.add_geometry(line5_pcd)
    vis.add_geometry(line6_pcd)

    # 设置视角
    ctr = vis.get_view_control()
    print(ctr.get_field_of_view())
    # ctr.change_field_of_view(step=30.0)
    print(ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    heatmap()
    main()