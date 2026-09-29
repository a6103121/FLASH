import math
import types
from functools import lru_cache
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import torch_dct as dct


@lru_cache(maxsize=32)
def _dct_frequency_distance(side, device, dtype):
   
    with torch.inference_mode(False), torch.no_grad():
        u = torch.linspace(0, side - 1, side, device=device, dtype=dtype)
        v = torch.linspace(0, side - 1, side, device=device, dtype=dtype)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        return torch.sqrt(u_grid ** 2 + v_grid ** 2)


@lru_cache(maxsize=32)
def _fft_low_frequency_mask(side, device):
    with torch.inference_mode(False), torch.no_grad():
        y = torch.arange(side, device=device).float()
        x = torch.arange(side, device=device).float()
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        center = side / 2.0
        dist = torch.sqrt((grid_y - center) ** 2 + (grid_x - center) ** 2)
        return dist <= 1


def apply_spectral_alignment_v(v_states, img_start_idx, img_end_idx, index=None, boost_factor=1.2, threshold=0.6,eps=1e-9):

    vis_v = v_states[:, index, img_start_idx:img_end_idx, :]

    bsz, num_v_heads, num_vis, head_dim = vis_v.shape
    side = int(math.sqrt(num_vis))

    x = vis_v.transpose(2, 3).reshape(bsz * num_v_heads, head_dim, side, side).float()
    orig_norm = torch.norm(x, dim=(-1, -2), keepdim=True)

    dct_coeffs = dct.dct_2d(x, norm='ortho')

    dist = _dct_frequency_distance(side, x.device, torch.get_default_dtype())
    limit = side * (1 - threshold)

    soft_mask = torch.sigmoid((dist - limit) )
    soft_mask[..., 0, 0] = 1.0

    magnitude = torch.abs(dct_coeffs)
    log_mag = torch.log(magnitude + eps)
    log_mag_enhanced = log_mag * (1.0 + (boost_factor - 1.0) * soft_mask)

    phase_sign = torch.sign(dct_coeffs)
    new_dct_coeffs = phase_sign * torch.exp(log_mag_enhanced)

    filtered_x = dct.idct_2d(new_dct_coeffs, norm='ortho')

    new_norm = torch.norm(filtered_x, dim=(-1, -2), keepdim=True)
    filtered_x = filtered_x * (orig_norm / (new_norm + 1e-9))

    filtered_v = filtered_x.reshape(bsz, num_v_heads, head_dim, num_vis).transpose(2, 3)
    v_states[:, index, img_start_idx:img_end_idx, :] = filtered_v.to(v_states.dtype)

    return v_states

def apply_soft_lowpass_logit(logits, img_start_idx, img_end_idx, threshold=0.7, strength=0.6, index=None):

    vis_logits = logits[:, index, :, img_start_idx:img_end_idx]
    bsz, num_h, q_sub, num_vis = vis_logits.shape
    side = int(math.sqrt(num_vis))

    x = vis_logits.reshape(bsz * num_h, q_sub, side, side).float()
    orig_norm = torch.norm(x, dim=(-1, -2), keepdim=True)

    dct_coeffs = dct.dct_2d(x, norm='ortho')

    dist = _dct_frequency_distance(side, x.device, torch.get_default_dtype())

    limit = side * (1 - threshold)
    soft_mask = torch.sigmoid((limit - dist) )
    final_mask = (1-strength)+strength * soft_mask
    final_mask[..., 0, 0] = 1.0

    smoothed_coeffs = dct_coeffs * final_mask

    filtered_x = dct.idct_2d(smoothed_coeffs, norm='ortho')

    new_norm = torch.norm(filtered_x, dim=(-1, -2), keepdim=True)
    filtered_x = filtered_x * (orig_norm / (new_norm + 1e-9))

    filtered_logits = filtered_x.reshape(bsz, num_h, q_sub, num_vis)
    logits[:, index, :, img_start_idx:img_end_idx] = filtered_logits.to(logits.dtype)

    return logits

def compute_spectral_features(attn_map, img_start_idx, img_end_idx, head_threshold=0.1, topk=1):

    device = attn_map.device
    probs = attn_map
    vis_probs = probs[:, img_start_idx:img_end_idx]  # [num_heads, num_tokens]
    visual_weight_sum = vis_probs.sum(dim=-1)  # [num_heads]

    is_visual_head = visual_weight_sum >= head_threshold

    num_heads, num_tokens = vis_probs.shape
    side = int(num_tokens ** 0.5)
    vis_probs_2d = vis_probs.view(num_heads, side, side).float()

    low_freq_mask = _fft_low_frequency_mask(side, device)

    f_shift = torch.fft.fftshift(torch.fft.fft2(vis_probs_2d), dim=(-2, -1))

    mag_db = 20 * torch.log10(torch.abs(f_shift) + 1e-8)

    total_energy = torch.sum(mag_db, dim=(1, 2)) + 1e-10
    low_freq_energy = torch.sum(mag_db * low_freq_mask, dim=(1, 2))
    low_freq_energy_ratio = low_freq_energy / total_energy

    gy, gx = torch.gradient(mag_db, dim=(-2, -1))

    grad_mag = torch.sqrt(gx ** 2 + gy ** 2) + 1e-10
    grad_ori = torch.atan2(gy, gx)


    cos_2theta = torch.cos(2 * grad_ori)
    sin_2theta = torch.sin(2 * grad_ori)

    sum_mag = torch.sum(grad_mag, dim=(1, 2)) + 1e-10
    weighted_cos = torch.sum(grad_mag * cos_2theta, dim=(1, 2)) / sum_mag
    weighted_sin = torch.sum(grad_mag * sin_2theta, dim=(1, 2)) / sum_mag
    orientation_coherence = torch.sqrt(weighted_cos ** 2 + weighted_sin ** 2)

    oc = 1.0 - orientation_coherence

    lr = low_freq_energy_ratio

    vortex_score = (lr + oc)

    v_head_count = is_visual_head.sum().item()
    topk_n = min(v_head_count, topk)


    vortex_score = torch.where(is_visual_head, vortex_score, vortex_score * 0.00001)

    score, index = torch.topk(vortex_score, k=topk_n, dim=-1)

    return score, index

def llama_new_forward_filtered_keys(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
):
    bsz, q_len, _ = hidden_states.size()
    img_start_idx = self.img_start_idx
    img_end_idx = self.img_end_idx

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    apply_v = self.v_start_layer <= self.layer_idx <= self.v_end_layer
    apply_s = self.s_start_layer <= self.layer_idx <= self.s_end_layer

    if apply_v or apply_s:
        # Both branches select heads from the same unfiltered attention logits.
        # Selection uses only batch 0 / the last query; other rows are unused.
        selection_weights = attn_weights[0:1, :, -1:, :]
        if attention_mask is not None:
            # Preserve broadcasting (including singleton batch/head/query axes).
            selection_mask = attention_mask.expand_as(attn_weights)[0:1, :, -1:, :]
            selection_weights = selection_weights + selection_mask
            selection_weights = torch.max(
                selection_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )
        temp_probs = nn.functional.softmax(selection_weights, dim=-1)
        _, index = compute_spectral_features(
            temp_probs[0, :, -1, :], img_start_idx, img_end_idx,
            head_threshold=self.tau, topk=self.k,
        )

        if len(index) > 0:
            if apply_v:
                value_states = apply_spectral_alignment_v(
                    v_states=value_states,
                    img_start_idx=img_start_idx,
                    img_end_idx=img_end_idx,
                    index=index,
                    boost_factor=self.lambda_v,
                    threshold=0.5,
                )
            if apply_s:
                attn_weights = apply_soft_lowpass_logit(
                    attn_weights,
                    img_start_idx,
                    img_end_idx,
                    index=index,
                    strength=self.lambda_s,
                    threshold=0.5,
                )

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, past_key_value


def llama_modify(model,
                 s_start_layer,
                 s_end_layer,
                 v_start_layer,
                 v_end_layer,
                 lambda_v,
                 lambda_s,
                 tau,
                 k,
                 img_start_idx,
                 img_end_idx):
    if not hasattr(model, "spectral_records"):
        model.spectral_records = []

    for i in range(0, 32):
        target_layer = model.model.layers[i].self_attn
        target_layer.layer_idx = i
        target_layer.lambda_v = lambda_v
        target_layer.lambda_s = lambda_s
        target_layer.k = k
        target_layer.tau = tau

        target_layer.img_start_idx = img_start_idx
        target_layer.img_end_idx = img_end_idx

        target_layer.s_start_layer = s_start_layer
        target_layer.s_end_layer = s_end_layer
        target_layer.v_start_layer = v_start_layer
        target_layer.v_end_layer = v_end_layer

        target_layer.results_logger = model.spectral_records
        target_layer.forward = types.MethodType(llama_new_forward_filtered_keys, target_layer)


if __name__ == "__main__":
    print("Adaptive Spectral Calibration (ASC) Module Integrated for ICML Submission.")
