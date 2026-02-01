import torch

def initialize(model_path, device):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )

    if "landscape_only" not in args:
        args = args[:-1] + ", landscape_only=False)"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )

    net = eval(args)
    net.load_state_dict(ckpt["model"], strict=False)
    return net.to(device)
