from pathlib import Path


def build_backbone(args):
    import open_clip

    kind = args.get("convnet_type", "clip").lower()
    defaults = {
        "clip": "laion400m_e32",
        "clip_laion2b": "laion2b_s34b_b88k",
        "openai_clip": "openai",
    }
    if kind not in defaults:
        raise ValueError(f"Unsupported backbone: {kind}")
    pretrained = args.get("clip_checkpoint") or args.get(
        "clip_pretrained", defaults[kind]
    )
    if args.get("clip_checkpoint"):
        checkpoint = Path(pretrained).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint}")
        pretrained = str(checkpoint)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=pretrained
    )
    model.out_dim = 512
    return model, preprocess, open_clip.get_tokenizer("ViT-B-16")
