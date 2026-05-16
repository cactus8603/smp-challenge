# yaml_controlled_experiment_patch

This patch centralizes experiment knobs into YAML.

Main YAML-controlled knobs:
- `fusion.type`
- `fusion.use_clip_similarity`
- `fusion.clip_similarity_mode`: `raw` or `projected`
- `model.head_type`: `moe` or `regression`
- `image.normalize_feature`
- `meta.use_semantic_groups`
- `meta.use_user_desc`
- `meta.user_desc_scale`
- `preprocess.use_user_desc_lite`
- `preprocess.user_desc_lite_debug`
- `tag_embedding.use`
- all hybrid loss weights
- `monitor.gbdt`
- `code_snapshot.enabled`

Copy:
```bash
cp /mnt/data/yaml_controlled_experiment_patch/base.yaml /code/smp-challenge/configs/base.yaml
cp /mnt/data/yaml_controlled_experiment_patch/text_meta_image_v2.yaml /code/smp-challenge/configs/text_meta_image_v2.yaml

cp /mnt/data/yaml_controlled_experiment_patch/train.py /code/smp-challenge/scripts/train.py
cp /mnt/data/yaml_controlled_experiment_patch/fusion_model.py /code/smp-challenge/src/models/fusion_model.py
cp /mnt/data/yaml_controlled_experiment_patch/meta_encoder.py /code/smp-challenge/src/models/meta_encoder.py
cp /mnt/data/yaml_controlled_experiment_patch/metadata_preprocessor.py /code/smp-challenge/src/datasets/metadata_preprocessor.py
cp /mnt/data/yaml_controlled_experiment_patch/fasttext_tag_encoder.py /code/smp-challenge/src/models/fasttext_tag_encoder.py
```

Recommended also copy:
```bash
cp /mnt/data/yaml_controlled_experiment_patch/fusion.py /code/smp-challenge/src/models/fusion.py
cp /mnt/data/yaml_controlled_experiment_patch/head.py /code/smp-challenge/src/models/head.py
cp /mnt/data/yaml_controlled_experiment_patch/text_encoder.py /code/smp-challenge/src/models/text_encoder.py
cp /mnt/data/yaml_controlled_experiment_patch/image_encoder.py /code/smp-challenge/src/models/image_encoder.py
```

Current default experiment:
- PairwiseGatedFusion
- raw CLIP sim enabled
- MoE head
- semantic metadata groups off
- 768-d user_desc modality off
- user_desc_lite metadata on
- fastText tag embedding off by default
- GBDT off
- code snapshot on
