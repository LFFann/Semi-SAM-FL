import torch
import torch.nn.functional as F


def _sobel_edges(x):
    gray = x.mean(dim=1, keepdim=True)
    kernel_x = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    kernel_y = gray.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-8)


def _sample_xy_from_weight(weight):
    b, _, h, w = weight.shape
    flat = weight.flatten(1).clamp_min(1e-8)
    flat = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-8)
    index = torch.multinomial(flat, 1).squeeze(1)
    y = torch.div(index, w, rounding_mode="floor").float()
    x = (index % w).float()
    return torch.stack(((x + 0.5) / float(w), (y + 0.5) / float(h)), dim=1).unsqueeze(1)


def build_structure_prompts(image, student_prob=None, num_prompts=4):
    """Build prompt descriptors from random points, boxes, edges, and student confidence."""
    num_prompts = max(int(num_prompts), 4)
    b, _, h, w = image.shape
    device = image.device
    prompts = []

    prompts.append({"kind": "point", "coords": torch.rand(b, 1, 2, device=device), "source": "random_point"})

    xy1 = torch.rand(b, 2, device=device) * 0.75
    wh = torch.rand(b, 2, device=device) * 0.35 + 0.15
    xy2 = (xy1 + wh).clamp(max=0.98)
    box = torch.stack((xy1, xy2), dim=1)
    prompts.append({"kind": "box", "coords": box, "source": "random_box"})

    edge_xy = _sample_xy_from_weight(_sobel_edges(image))
    prompts.append({"kind": "point", "coords": edge_xy, "source": "image_edge_point"})

    if student_prob is not None:
        confidence = student_prob.detach().max(dim=1, keepdim=True).values
    else:
        confidence = torch.ones(b, 1, h, w, device=device)
    conf_xy = _sample_xy_from_weight(confidence)
    prompts.append({"kind": "point", "coords": conf_xy, "source": "student_high_confidence"})

    while len(prompts) < num_prompts:
        prompts.append({"kind": "point", "coords": torch.rand(b, 1, 2, device=device), "source": "random_point_extra"})
    return prompts[:num_prompts]


def _point_embedding(sam_model, coords):
    prompt_encoder = sam_model.prompt_encoder
    emb = prompt_encoder.pe_layer.forward_with_coords(coords, prompt_encoder.input_image_size)
    emb = emb + prompt_encoder.point_embeddings[1].weight.view(1, 1, -1)
    labels = torch.ones(coords.shape[:2], dtype=torch.long, device=coords.device)
    return (emb, labels), None


def _box_embedding(sam_model, coords):
    prompt_encoder = sam_model.prompt_encoder
    emb = prompt_encoder.pe_layer.forward_with_coords(coords, prompt_encoder.input_image_size)
    emb[:, 0, :] = emb[:, 0, :] + prompt_encoder.point_embeddings[2].weight
    emb[:, 1, :] = emb[:, 1, :] + prompt_encoder.point_embeddings[3].weight
    return None, emb


def _mask_gradient(mask_stack):
    dx = mask_stack[..., :, 1:] - mask_stack[..., :, :-1]
    dy = mask_stack[..., 1:, :] - mask_stack[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-8)


@torch.no_grad()
def compute_structure_reliability(image, prompts):
    """Compute SSRF reliability maps from frozen SAM prompt-response stability.

    Args:
        image: Tensor shaped [B, C, H, W].
        prompts: Dict with keys:
            sam_model: frozen SAM model.
            items: list returned by build_structure_prompts.
            affinity_size: downsampled side length for R_a.

    Returns:
        R_b: boundary reliability [B, 1, H, W].
        R_a: downsampled pairwise region consistency [B, S*S, S*S].
        R_u: normalized uncertainty map [B, 1, H, W].
    """
    sam_model = prompts["sam_model"]
    prompt_items = prompts["items"]
    affinity_size = int(prompts.get("affinity_size", 16))

    sam_model.eval()
    for param in sam_model.parameters():
        param.requires_grad_(False)

    b, _, h, w = image.shape
    image_embeddings = sam_model.image_encoder(image)
    masks = []
    for prompt in prompt_items:
        if prompt["kind"] == "box":
            points, boxes = _box_embedding(sam_model, prompt["coords"].to(image.device))
        else:
            points, boxes = _point_embedding(sam_model, prompt["coords"].to(image.device))
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=points, boxes=boxes, masks=None)
        low_res_masks, _ = sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        mask = F.interpolate(low_res_masks, size=(h, w), mode="bilinear", align_corners=False).sigmoid()
        masks.append(mask.squeeze(1))

    mask_stack = torch.stack(masks, dim=0).clamp(1e-6, 1.0 - 1e-6)
    grad_var = _mask_gradient(mask_stack).var(dim=0, unbiased=False)
    R_b = torch.exp(-grad_var).unsqueeze(1).clamp(0.0, 1.0)

    prob = mask_stack.mean(dim=0)
    R_u = (-(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log()) / 0.6931471805599453)
    R_u = R_u.unsqueeze(1).clamp(0.0, 1.0)

    small = F.interpolate(mask_stack.flatten(0, 1).unsqueeze(1), size=(affinity_size, affinity_size),
                          mode="bilinear", align_corners=False)
    small = small.view(len(masks), b, -1).permute(1, 0, 2).clamp(1e-6, 1.0 - 1e-6)
    R_a = torch.einsum("bkn,bkm->bnm", small, small)
    R_a = R_a + torch.einsum("bkn,bkm->bnm", 1.0 - small, 1.0 - small)
    R_a = (R_a / float(len(masks))).clamp(0.0, 1.0)

    return R_b, R_a, R_u
