# KITTI ➜ V2V4Real 训练与测试流程（Car 类）

本文档说明如何使用 KITTI 原始数据结构训练 Car 类模型，并在 V2V4Real 上进行测试。

## 1) 准备 KITTI 训练配置

推荐使用示例配置 `opencood/hypes_yaml/point_pillar_S2Rformer_kitti.yaml`，确保路径指向 KITTI 原始目录结构：

- `root_dir` / `root_dir_target`: 指向 `KITTI/training`
- `validate_dir`: 指向 KITTI 验证集（可用自定义划分）

需要确保目录结构包含：

```
KITTI/
  training/
    velodyne/
    label_2/
    calib/
```

## 2) 训练（只用 KITTI）

```bash
CUDA_VISIBLE_DEVICES=0 python opencood/tools/train.py \
  --hypes_yaml opencood/hypes_yaml/point_pillar_S2Rformer_kitti.yaml
```

训练完成后会在 `save_path` 下生成 `config.yaml` 和模型权重。

## 3) 用 V2V4Real 进行测试

因为 `inference.py` 会读取 `model_dir/config.yaml`，需要将其中的：

- `fusion.core_method` 设置为 `IntermediateFusionDataset`
- `validate_dir` 指向 `V2V4Real/test`

你可以手动修改，也可以用脚本：

```bash
opencood/tools/run_kitti_to_v2v4real_eval.sh \
  opencood/hypes_yaml/point_pillar_S2Rformer_kitti.yaml \
  /path/to/model_dir \
  /path/to/V2V4Real/test
```

脚本会自动训练、更新 `config.yaml` 并运行：

```bash
python opencood/tools/inference.py \
  --model_dir /path/to/model_dir \
  --fusion_method intermediate
```
