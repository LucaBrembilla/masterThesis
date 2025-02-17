import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

from tools.streaming_utilis.temporal_state import update_temporal_state, predict_from_state
from tools.streaming_utilis.crop import crop_point_cloud

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion 
import math

def quaternion_yaw(q):
    """
    Extracts yaw (rotation around z-axis) from a quaternion.
    Assumes q is a list/array in the format [w, x, y, z].
    """
    quat = Quaternion(q)
    # Returns (yaw, pitch, roll) in radians.
    yaw, _, _ = quat.yaw_pitch_roll
    return yaw

def rotation_matrix_to_yaw(R):
    """
    Compute yaw from a 3x3 rotation matrix.
    Yaw is defined as the angle between the x-axis and the projection of the 
    rotated x-axis onto the ground plane.
    """
    return math.atan2(R[1, 0], R[0, 0])

def get_lidar_pose(nusc, sample_token):
    # Get sample and LiDAR sample data.
    sample = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    
    # Get ego pose (in the world frame).
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    # Get LiDAR extrinsics (calibration relative to ego).
    calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    
    # Convert rotation lists to Quaternion objects.
    from pyquaternion import Quaternion
    lidar_quat = Quaternion(calib['rotation'])
    ego_quat = Quaternion(ego_pose['rotation'])
    
    # Compute transformation matrices.
    lidar_to_ego = transform_matrix(calib['translation'], rotation=lidar_quat)
    ego_to_world = transform_matrix(ego_pose['translation'], rotation=ego_quat)
    
    # Compose to get the LiDAR-to-world transformation matrix.
    lidar_to_world = ego_to_world @ lidar_to_ego  # 4x4 transformation matrix
    
    return lidar_to_world



def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    final_output_dir = result_dir / 'final_result' / timestamp / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    prev_detections = None
    tracker = None
    tot_num_points = 0 # sum of the number of points of all the pointclouds in the batch
    points_analyzed = 0
    current_timestamp = 0
    previous_timestamp = 0

    nusc = NuScenes(version='v1.0-mini', dataroot='/home/brembilla/exp/private_datasets/nuscenes/v1.0-mini', verbose=True)
    
    for i, batch_dict in enumerate(dataloader):
        
        sample_token = batch_dict['metadata'][0]['token']
        current_pose_mat = get_lidar_pose(nusc, sample_token)

        tot_num_points += batch_dict['points'].shape[0]

        print(f"Frame {i}. Id: {batch_dict['frame_id']}. Metadata: {batch_dict['metadata']}. # points {len(batch_dict['points'])}")
        
        states = predict_from_state(tracker, current_pose_mat, time_step=0.05)
        # if states:
            # print(f"States: {states[:5]}")
        # timestamp = batch_dict['frame_id'][0].split('__')[-1].spit('.')[0]
        # print(timestamp)

        if i == 40:
            print("New sequence")
        
        if i != 40 and i % 10:  # Frame 40 is a new sequence
            # Crop current frame using previous detections
            points = crop_point_cloud(
                batch_dict['points'], 
                # np.array([track['box'] for track in tracker['track_states']]),
                np.array(states),
                expand_ratio=2
            )
            
            data_dict = dataset.process_pointcloud(points = points, frame_id = i)
            new_col = np.zeros((data_dict['voxel_coords'].shape[0], 1))
            data_dict['voxel_coords'] = np.concatenate((new_col, data_dict['voxel_coords']), axis=1)
            data_dict['metadata'] = batch_dict['metadata']
            data_dict['frame_id'] = batch_dict['frame_id']
            batch_dict = data_dict
            batch_dict['batch_size'] =  1
            points_analyzed += len(batch_dict['points'])
            print(f"Cropping for frame {i}, new #points: {len(batch_dict['points'])}") 
        else:
            points_analyzed += len(batch_dict['points'])
    
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
            # print(f"Number of predicted objects: {len(pred_dicts[0]['pred_boxes'])}")
            """print(f"Predicted objects: {pred_dicts[0]['pred_boxes'][:5]}")   
            print(f"Predicted scores: {pred_dicts[0]['pred_scores'][:5]}")
            print(f"Predicted classes: {pred_dicts[0]['pred_labels'][:5]}")  """

        # Update tracking 
        prev_detections, tracker = update_temporal_state(
            pred_dicts,
            tracker,
            time_step=0.05,
            current_pose_mat=current_pose_mat
        )

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Percentage points analyzed: %.2f' % (points_analyzed / tot_num_points * 100))

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
