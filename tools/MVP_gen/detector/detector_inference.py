from mmdet.apis import init_detector, inference_detector
import pprint

def inference(input_image, score_threshold=0.3):
    config_file = 'detector/fcos_r50_caffe_fpn_gn-head_1x_kitti.py'
    checkpoint_file = 'detector/fcos_r50_caffe_fpn_gn-head_1x_kitti_20230410_mine.pth'
    model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
    results = inference_detector(model, input_image)

    result_dict = {}
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')
    for i, re in enumerate(results):
        mask = re[:, -1] >= score_threshold
        re = re[mask]
        if re.size > 0:
            result_dict[CLASSES[i]] = re
        else:
            result_dict[CLASSES[i]] = None
    print('inference results:')
    pprint.pprint(result_dict)
    return result_dict

if __name__ == '__main__':
    inference('demo_kitti.png')