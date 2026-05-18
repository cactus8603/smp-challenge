from __future__ import annotations

from typing import Any, Dict

from src.models.fusion_model import SMPFusionModel


def build_smp_fusion_model(
    cfg: Dict[str, Any],
    preprocessor,
    tag_encoder=None,
    dataset_user_desc_dim: int = 0,
    dataset_loc_desc_dim: int = 0,
) -> SMPFusionModel:
    model_cfg = cfg["model"]
    text_cfg = cfg["text"]
    image_cfg = cfg["image"]
    meta_cfg = cfg["meta"]
    fusion_cfg = cfg["fusion"]
    tag_embedding_cfg = cfg.get("tag_embedding", {}) or {}

    use_text = bool(model_cfg["use_text"])
    use_meta = bool(model_cfg["use_meta"])
    use_image = bool(model_cfg["use_image"])
    use_tag_embedding = bool(tag_embedding_cfg.get("use", tag_embedding_cfg.get("enabled", False)))

    cat_cardinalities = [
        int(preprocessor.cat_cardinalities[col])
        for col in preprocessor.cat_cols
    ]

    return SMPFusionModel(
        text_model_name=text_cfg["model_name"],
        meta_num_dim=len(preprocessor.transformed_num_cols),
        meta_cat_cardinalities=cat_cardinalities,
        meta_bin_dim=len(preprocessor.transformed_bin_cols),
        meta_num_cols=preprocessor.transformed_num_cols,
        meta_cat_cols=preprocessor.transformed_cat_cols,
        meta_bin_cols=preprocessor.transformed_bin_cols,
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        use_text=use_text,
        use_tag_embedding=use_tag_embedding,
        tag_encoder=tag_encoder,
        use_meta=use_meta,
        use_image=use_image,
        image_model_name=image_cfg["model_name"],
        text_pooling=text_cfg["pooling"],
        text_trainable=bool(text_cfg["trainable"]),
        image_pretrained=bool(image_cfg.get("pretrained", True)),
        image_trainable=bool(image_cfg["trainable"]),
        fusion_type=str(fusion_cfg.get("type", "pairwise_gated")),
        use_clip_similarity=bool(fusion_cfg.get("use_clip_similarity", True)),
        clip_similarity_mode=str(fusion_cfg.get("clip_similarity_mode", "raw")),
        head_type=str(model_cfg.get("head_type", "moe")),
        use_meta_aux_head=bool(model_cfg.get("use_meta_aux_head", False)),
        meta_aux_scale=float(model_cfg.get("meta_aux_scale", 0.0)),
        use_meta_residual=bool(model_cfg.get("use_meta_residual", True)),
        meta_residual_init=float(model_cfg.get("meta_residual_init", 0.0)),
        meta_residual_dropout=float(model_cfg.get("meta_residual_dropout", 0.0)),
        head_hidden_mult=float(model_cfg.get("head_hidden_mult", 1.0)),
        head_num_layers=int(model_cfg.get("head_num_layers", 2)),
        fusion_num_heads=int(fusion_cfg.get("num_heads", 4)),
        fusion_meta_res_scale=float(fusion_cfg.get("meta_res_scale", 0.75)),
        fusion_text_res_scale=float(fusion_cfg.get("text_res_scale", 0.50)),
        fusion_image_res_scale=float(fusion_cfg.get("image_res_scale", 0.50)),
        fusion_tm_res_scale=float(fusion_cfg.get("tm_res_scale", 0.75)),
        fusion_order=str(fusion_cfg.get("order", "text_meta_first")),
        normalize_image_feature=bool(image_cfg.get("normalize_feature", False)),
        meta_branch_dim=int(meta_cfg["branch_dim"]),
        use_semantic_meta_groups=bool(meta_cfg.get("use_semantic_groups", False)),
        meta_encoder_type=str(
            meta_cfg.get(
                "encoder_type",
                "semantic_groups" if meta_cfg.get("use_semantic_groups", False) else "legacy",
            )
        ),
        ft_token_dim=int(meta_cfg.get("ft_token_dim", 192)),
        ft_num_layers=int(meta_cfg.get("ft_num_layers", 3)),
        ft_num_heads=int(meta_cfg.get("ft_num_heads", 8)),
        ft_ffn_mult=int(meta_cfg.get("ft_ffn_mult", 2)),
        ft_dropout=meta_cfg.get("ft_dropout", None),
        use_user_desc=bool(meta_cfg.get("use_user_desc", False)),
        user_desc_dim=int(dataset_user_desc_dim or meta_cfg.get("user_desc_dim", 768)),
        use_loc_desc=bool(meta_cfg.get("use_loc_desc", False)),
        loc_desc_dim=int(meta_cfg.get("loc_desc_dim", dataset_loc_desc_dim or 400)),
        desc_bottleneck_dim=int(meta_cfg.get("desc_bottleneck_dim", 32)),
        user_desc_scale=float(meta_cfg.get("user_desc_scale", 0.0)),
        user_desc_mode=str(meta_cfg.get("user_desc_mode", "residual")),
        user_desc_gate_init=float(meta_cfg.get("user_desc_gate_init", -4.0)),
        user_desc_res_scale=float(meta_cfg.get("user_desc_res_scale", 0.35)),
        loc_desc_scale=float(meta_cfg.get("loc_desc_scale", 0.0)),
    )
