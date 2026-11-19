from __future__ import annotations

import torch
import torch.nn as nn
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks


def build_pi05_prefix_embs(
    pi05_model,
    img_masks,
    tokens,
    masks,
    image_embs,
    images,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact image+language prefix embeddings for PI05 language prefill."""
    per_camera_batch = int(images[0].shape[0])
    image_embs_list = list(
        image_embs.reshape(len(images), per_camera_batch, -1, image_embs.shape[-1])
    )

    embs: list[torch.Tensor] = []
    pad_masks: list[torch.Tensor] = []

    for img_emb, img_mask in zip(image_embs_list, img_masks, strict=True):
        bsize, num_img_embs = img_emb.shape[:2]
        embs.append(img_emb)
        img_mask = img_mask.to(device=img_emb.device, dtype=torch.bool)
        pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

    lang_emb = pi05_model.paligemma_with_expert.embed_language_tokens(tokens)
    embs.append(lang_emb)
    pad_masks.append(masks.to(device=lang_emb.device, dtype=torch.bool))

    prefix_embs = torch.cat(embs, dim=1)
    prefix_pad_masks = torch.cat(pad_masks, dim=1)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    valid = prefix_pad_masks.to(device=prefix_embs.device, dtype=torch.bool)
    valid_counts = valid.sum(dim=1)
    if not torch.equal(valid_counts, valid_counts[:1].expand_as(valid_counts)):
        raise ValueError(
            "build_pi05_prefix_embs requires equal valid token counts across the batch"
        )

    compact_len = int(valid_counts[0].item())
    compact_embs = torch.stack(
        [prefix_embs[b, valid[b], :] for b in range(prefix_embs.shape[0])],
        dim=0,
    )
    compact_position_ids = torch.stack(
        [prefix_position_ids[b, valid[b]] for b in range(prefix_position_ids.shape[0])],
        dim=0,
    )
    compact_pad_mask = torch.ones(
        prefix_embs.shape[0],
        compact_len,
        device=prefix_pad_masks.device,
        dtype=torch.bool,
    )
    compact_attention_mask = torch.zeros(
        prefix_embs.shape[0],
        1,
        compact_len,
        compact_len,
        device=prefix_embs.device,
        dtype=torch.float32,
    )
    return compact_embs, compact_pad_mask, compact_attention_mask, compact_position_ids


def pi05_compact_index(
    img_masks,
    images: list,
    seq_len_per_image: int,
    lang_masks: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Gather index for leftover ``fuse_prefix`` (same valid rows as ``build_pi05_prefix_embs``)."""
    batch = int(lang_masks.shape[0])
    pads: list[torch.Tensor] = []
    for img_mask in img_masks:
        mask = img_mask.to(device=device, dtype=torch.bool)
        pads.append(mask[:, None].expand(batch, int(seq_len_per_image)))
    vision_pad = torch.cat(pads, dim=1)
    lang_pad = lang_masks.to(device=device, dtype=torch.bool)
    valid = torch.cat([vision_pad, lang_pad], dim=1)
    counts = valid.sum(dim=1)
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError(
            "pi05_compact_index requires equal valid token counts across the batch"
        )
    return torch.stack(
        [torch.nonzero(valid[b], as_tuple=False).squeeze(-1) for b in range(batch)],
        dim=0,
    )


def pi05_seq_len_per_image(image_embs: torch.Tensor, images: list) -> int:
    """Vision token count per image view after PI05 camera reshape."""
    per_camera_batch = int(images[0].shape[0])
    num_images = len(images)
    reshaped = image_embs.reshape(
        num_images, per_camera_batch, -1, image_embs.shape[-1]
    )
    return int(reshaped.shape[2])


def pi05_prefix_max_seq_len(
    *,
    num_images: int,
    seq_len_per_image: int,
    tokenizer_max_length: int,
) -> int:
    """Upper bound on compact PI05 prefix length (vision slots + language tokens)."""
    return int(num_images) * int(seq_len_per_image) + int(tokenizer_max_length)


def pi05_compact_prefix_max_seq_len(
    image_embs: torch.Tensor,
    images: list,
    tokenizer_max_length: int,
) -> int:
    """Static TRT prefill length for PI05 compact prefix (vision + language slots)."""
    return pi05_prefix_max_seq_len(
        num_images=len(images),
        seq_len_per_image=pi05_seq_len_per_image(image_embs, images),
        tokenizer_max_length=tokenizer_max_length,
    )


def pad_pi05_compact_prefix(
    prefix_embs: torch.Tensor,
    prefix_pad_mask: torch.Tensor,
    prefix_attention_mask: torch.Tensor,
    prefix_position_ids: torch.Tensor,
    *,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Right-pad a compact PI05 prefix to ``max_seq_len`` for static TRT engines.

    Returns padded tensors and the valid (unpadded) sequence length.
    """
    valid_len = int(prefix_embs.shape[1])
    max_seq_len = int(max_seq_len)
    if valid_len > max_seq_len:
        raise ValueError(
            f"PI05 compact prefix length {valid_len} exceeds max_seq_len {max_seq_len}"
        )
    if valid_len == max_seq_len:
        return (
            prefix_embs,
            prefix_pad_mask,
            prefix_attention_mask,
            prefix_position_ids,
            valid_len,
        )

    batch_size = int(prefix_embs.shape[0])
    pad_len = max_seq_len - valid_len
    device = prefix_embs.device
    dtype = prefix_embs.dtype

    padded_embs = torch.zeros(
        batch_size, max_seq_len, prefix_embs.shape[-1], device=device, dtype=dtype
    )
    padded_embs[:, :valid_len, :] = prefix_embs

    padded_pad_mask = torch.zeros(
        batch_size, max_seq_len, device=device, dtype=torch.bool
    )
    padded_pad_mask[:, :valid_len] = prefix_pad_mask.to(device=device, dtype=torch.bool)

    padded_position_ids = torch.zeros(
        batch_size, max_seq_len, device=device, dtype=prefix_position_ids.dtype
    )
    padded_position_ids[:, :valid_len] = prefix_position_ids

    # Bidirectional padding mask: valid x valid block only.
    padded_attention_mask = torch.zeros(
        batch_size,
        1,
        max_seq_len,
        max_seq_len,
        device=device,
        dtype=prefix_attention_mask.dtype,
    )
    padded_attention_mask[:, :, :valid_len, :valid_len] = prefix_attention_mask[
        :, :, :valid_len, :valid_len
    ]

    return (
        padded_embs,
        padded_pad_mask,
        padded_attention_mask,
        padded_position_ids,
        valid_len,
    )


def make_pi05_suffix_position_and_mask(core, prefix_pad_masks, x_t, device):
    """Suffix position ids and 4D attention mask for PI05 diffusion."""
    batch_size, suffix_len = x_t.shape[:2]
    prefix_pad_masks = prefix_pad_masks.to(device=device)
    prefix_len = prefix_pad_masks.shape[1]

    suffix_pad_masks = torch.ones(
        batch_size, suffix_len, dtype=torch.bool, device=device
    )
    suffix_att_masks = torch.tensor(
        [1] + [0] * (suffix_len - 1),
        dtype=torch.int64,
        device=device,
    )[None, :].expand(batch_size, -1)

    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
        batch_size, suffix_len, prefix_len
    )
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

    attention_mask = core._prepare_attention_masks_4d(full_att_2d_masks)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    return position_ids, attention_mask


def _core(model: nn.Module) -> nn.Module:
    if hasattr(model, "paligemma_with_expert"):
        return model
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module) and hasattr(inner, "paligemma_with_expert"):
        return inner
    raise RuntimeError("PI05 spec expected a policy or paligemma_with_expert module")


def _nchw_to_hwc(pixel_values):
    if pixel_values.ndim != 4:
        return pixel_values
    if pixel_values.shape[1] in (1, 3, 4) and pixel_values.shape[-1] not in (1, 3, 4):
        return pixel_values.permute(0, 2, 3, 1).contiguous()
    return pixel_values
