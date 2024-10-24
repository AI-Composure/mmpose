import os
import os.path as osp
from glob import glob
import cv2
import argparse
import json
import numpy as np
from tqdm import tqdm

# setup: https://mmpose.readthedocs.io/en/latest/installation.html

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, dest='dataset_path')
    args = parser.parse_args()
    assert args.dataset_path, "Please set dataset_path."
    return args

args = parse_args()
dataset_path = args.dataset_path

output_root = './vis_results'
os.system('rm -rf ' + output_root)

# run mmpose
cmd = 'python demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py dw-ll_ucoco_384.pth  --input ' + osp.join(dataset_path, 'images') + ' --output-root ' + output_root + ' --save-predictions'
print(cmd)
os.system(cmd)

# move predictions to the root path
os.makedirs(osp.join(dataset_path, 'keypoints_whole_body'), exist_ok=True)
img_path_list = sorted(glob(osp.join(dataset_path, 'images', '*.png')) + glob(osp.join(dataset_path, 'images', '*.jpg')))
frame_idx_list = [int(x.split('/')[-1][:-4]) for x in img_path_list]
output_path_list = glob(osp.join(output_root, '*.json'))
for output_path in output_path_list:
    frame_idx = output_path.split('/')[-1].split('results_')[1][:-5]
    with open(output_path) as f:
        out = json.load(f)

    kpt_save = None
    for i in range(len(out['instance_info'])):
        xy = np.array(out['instance_info'][i]['keypoints'], dtype=np.float32).reshape(-1,2)
        score = np.array(out['instance_info'][i]['keypoint_scores'], dtype=np.float32).reshape(-1,1)
        kpt = np.concatenate((xy, score),1) # x, y, score
        if (kpt_save is None) or (kpt_save[:,2].mean() < kpt[:,2].mean()):
            kpt_save = kpt
    with open(osp.join(dataset_path, 'keypoints_whole_body', frame_idx + '.json'), 'w') as f:
        json.dump(kpt_save.tolist(), f)

# add original image and frame index to the video
output_path_list = glob(osp.join(output_root, '*.png')) + glob(osp.join(output_root, '*.jpg'))
output_path_list = sorted(output_path_list)
img_height, img_width = cv2.imread(output_path_list[0]).shape[:2]
video_save = cv2.VideoWriter(osp.join(dataset_path, 'keypoints_whole_body.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height)) 
for i, output_path in enumerate(output_path_list):
    frame_idx = output_path.split('/')[-1][:-4]
    input_img_path = img_path_list[i]
    img = cv2.imread(input_img_path)
    output = cv2.imread(output_path)
    vis = np.concatenate((img, output),1)
    vis = cv2.putText(vis, frame_idx, (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
    video_save.write(vis)
video_save.release()

os.system('rm -rf ' + output_root)
