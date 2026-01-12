"""
Dataset class for KITTI single-agent source domain training.
"""
import os
from collections import OrderedDict

import numpy as np
import torch

import opencood.data_utils.post_processor as post_processor
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils import box_utils
from opencood.utils.pcd_utils import mask_points_by_range, mask_ego_points, \
    shuffle_points


class KittiIntermediateFusionDataset(torch.utils.data.Dataset):
    def __init__(self, params, visualize, train=True, isSim=False):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.isSim = isSim

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)
        self.augment_config = params['data_augment']
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train,
                                            intermediate=True)

        self.max_cav = params['train_params'].get('max_cav', 1)

        root_dir = self._get_root_dir()
        self.velodyne_dir = os.path.join(root_dir, 'velodyne')
        self.label_dir = os.path.join(root_dir, 'label_2')
        self.calib_dir = os.path.join(root_dir, 'calib')

        self.frame_ids = sorted([
            os.path.splitext(fname)[0]
            for fname in os.listdir(self.velodyne_dir)
            if fname.endswith('.bin')
        ])

    def _get_root_dir(self):
        if self.train:
            return self.params['root_dir'] if self.isSim \
                else self.params['root_dir_target']
        return self.params['validate_dir']

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        lidar_np = self._load_lidar(frame_id)
        calib = self._load_calib(frame_id)
        object_bbx_center, object_ids = self._load_labels(frame_id, calib)

        object_bbx_center, object_bbx_mask = \
            self._pad_boxes(object_bbx_center)

        flip, noise_rotation, noise_scale = self.generate_augment()

        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_ego_points(lidar_np)
        lidar_np, object_bbx_center, object_bbx_mask = self.augment(
            lidar_np, object_bbx_center, object_bbx_mask,
            flip, noise_rotation, noise_scale
        )

        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])

        object_bbx_center_valid = object_bbx_center[object_bbx_mask == 1]
        object_bbx_center_valid, range_mask = \
            box_utils.mask_boxes_outside_range_numpy(
                object_bbx_center_valid,
                self.params['preprocess']['cav_lidar_range'],
                self.params['postprocess']['order']
            )
        object_ids = [int(x) for x in list(np.array(object_ids)[range_mask])]

        object_bbx_center, object_bbx_mask = \
            self._pad_boxes(object_bbx_center_valid)

        processed_features = self.pre_processor.preprocess(lidar_np)
        merged_feature_dict = self.merge_features_to_dict(
            [processed_features])

        anchor_box = self.post_processor.generate_anchor_box()
        label_dict = self.post_processor.generate_label(
            gt_box_center=object_bbx_center,
            anchors=anchor_box,
            mask=object_bbx_mask)

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {
            'object_bbx_center': object_bbx_center,
            'object_bbx_mask': object_bbx_mask,
            'object_ids': object_ids,
            'anchor_box': anchor_box,
            'processed_lidar': merged_feature_dict,
            'label_dict': label_dict,
            'cav_num': 1,
            'velocity': [0.0],
            'time_delay': [0.0],
            'infra': [0.0],
            'spatial_correction_matrix': np.eye(4)[None],
            'pairwise_t_matrix': self.get_pairwise_transformation(
                self.max_cav),
        }

        if self.visualize:
            processed_data_dict['ego'].update({'origin_lidar': lidar_np})

        return processed_data_dict

    def _load_lidar(self, frame_id):
        lidar_file = os.path.join(self.velodyne_dir, f'{frame_id}.bin')
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def _load_calib(self, frame_id):
        calib_file = os.path.join(self.calib_dir, f'{frame_id}.txt')
        calib = {}
        with open(calib_file, 'r', encoding='utf-8') as calib_handle:
            for line in calib_handle:
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                calib[key] = np.array(
                    [float(x) for x in value.strip().split()],
                    dtype=np.float32
                )

        rect = calib.get('R0_rect')
        if rect is None:
            rect = calib.get('R_rect')
        rect = rect.reshape(3, 3) if rect is not None else np.eye(3)
        rect_4x4 = np.eye(4, dtype=np.float32)
        rect_4x4[:3, :3] = rect

        velo_to_cam = calib.get('Tr_velo_to_cam')
        if velo_to_cam is None:
            velo_to_cam = calib.get('Tr_velo_cam')
        velo_to_cam = velo_to_cam.reshape(3, 4)
        velo_to_cam_4x4 = np.eye(4, dtype=np.float32)
        velo_to_cam_4x4[:3, :4] = velo_to_cam

        rect_to_cam = rect_4x4 @ velo_to_cam_4x4
        cam_to_velo = np.linalg.inv(rect_to_cam)
        return cam_to_velo

    def _load_labels(self, frame_id, cam_to_velo):
        label_file = os.path.join(self.label_dir, f'{frame_id}.txt')
        boxes = []
        object_ids = []
        if not os.path.exists(label_file):
            return np.array(boxes, dtype=np.float32), object_ids

        with open(label_file, 'r', encoding='utf-8') as label_handle:
            for line in label_handle:
                fields = line.strip().split()
                if len(fields) < 15:
                    continue
                obj_type = fields[0]
                if obj_type != 'Car':
                    continue
                h, w, l = map(float, fields[8:11])
                x, y, z = map(float, fields[11:14])
                rotation_y = float(fields[14])
                location_cam = np.array([x, y - h / 2.0, z, 1.0],
                                        dtype=np.float32)
                location_velo = cam_to_velo @ location_cam
                yaw = -rotation_y - np.pi / 2.0

                if self.params['postprocess']['order'] == 'hwl':
                    dims = [h, w, l]
                else:
                    dims = [l, w, h]
                boxes.append([
                    location_velo[0],
                    location_velo[1],
                    location_velo[2],
                    *dims,
                    yaw
                ])
                object_ids.append(len(object_ids))

        return np.array(boxes, dtype=np.float32), object_ids

    def _pad_boxes(self, object_bbx_center):
        object_bbx_center_padded = \
            np.zeros((self.params['postprocess']['max_num'], 7))
        object_bbx_mask = np.zeros(self.params['postprocess']['max_num'])
        num_boxes = min(object_bbx_center.shape[0],
                        self.params['postprocess']['max_num'])
        if num_boxes > 0:
            object_bbx_center_padded[:num_boxes, :] = \
                object_bbx_center[:num_boxes]
            object_bbx_mask[:num_boxes] = 1
        return object_bbx_center_padded, object_bbx_mask

    def generate_augment(self):
        flip = [None, None, None]
        noise_rotation = None
        noise_scale = None

        for aug_ele in self.augment_config:
            if 'random_world_rotation' in aug_ele['NAME']:
                rot_range = aug_ele['WORLD_ROT_ANGLE']
                if not isinstance(rot_range, list):
                    rot_range = [-rot_range, rot_range]
                noise_rotation = np.random.uniform(rot_range[0],
                                                   rot_range[1])

            if 'random_world_flip' in aug_ele['NAME']:
                for i, cur_axis in enumerate(aug_ele['ALONG_AXIS_LIST']):
                    enable = np.random.choice([False, True], replace=False,
                                              p=[0.5, 0.5])
                    flip[i] = enable

            if 'random_world_scaling' in aug_ele['NAME']:
                scale_range = aug_ele['WORLD_SCALE_RANGE']
                noise_scale = np.random.uniform(scale_range[0],
                                                scale_range[1])

        return flip, noise_rotation, noise_scale

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask,
                flip=None, rotation=None, scale=None):
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask,
                    'flip': flip,
                    'noise_rotation': rotation,
                    'noise_scale': scale}
        tmp_dict = self.data_augmentor.forward(tmp_dict)
        return tmp_dict['lidar_np'], tmp_dict['object_bbx_center'], \
            tmp_dict['object_bbx_mask']

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        merged_feature_dict = OrderedDict()
        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)
        return merged_feature_dict

    def collate_batch_train(self, batch):
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        object_ids = []
        processed_lidar_list = []
        record_len = []
        label_dict_list = []

        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix_list = []
        pairwise_t_matrix_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            object_ids.append(ego_dict['object_ids'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            record_len.append(ego_dict['cav_num'])
            label_dict_list.append(ego_dict['label_dict'])
            velocity.append(ego_dict['velocity'])
            time_delay.append(ego_dict['time_delay'])
            infra.append(ego_dict['infra'])
            spatial_correction_matrix_list.append(
                ego_dict['spatial_correction_matrix'])
            pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)
        record_len = torch.from_numpy(np.array(record_len, dtype=int))
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)

        pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))
        velocity = torch.from_numpy(np.array(velocity))
        time_delay = torch.from_numpy(np.array(time_delay))
        infra = torch.from_numpy(np.array(infra))
        spatial_correction_matrix_list = torch.from_numpy(
            np.array(spatial_correction_matrix_list))
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()

        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   'label_dict': label_torch_dict,
                                   'object_ids': object_ids[0],
                                   'prior_encoding': prior_encoding,
                                   'spatial_correction_matrix': spatial_correction_matrix_list,
                                   'pairwise_t_matrix': pairwise_t_matrix})

        if self.visualize:
            origin_lidar = np.array(origin_lidar)
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(
                    batch[0]['ego']['anchor_box']))})

        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch})

        return output_dict

    def post_process(self, data_dict, output_dict):
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor

    def get_pairwise_transformation(self, max_cav):
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))
        pairwise_t_matrix[:, :] = np.identity(4)
        return pairwise_t_matrix
