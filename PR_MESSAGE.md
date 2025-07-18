# Pull Request – Thumbnail Inpainting Refactor / NPU Support

## Overview
This PR modernises and streamlines our fork of **LaMa** for thumbnail inpainting, adds hand-mask training support, and introduces NPU-friendly tensor helpers.

## Key Points
1. **Repository baseline**
   * Forked from the official LaMa repo (<https://github.com/advimman/lama>).
   * ONNX export logic inspired by the Carve-Photos fork (<https://github.com/Carve-Photos/lama>).
2. **Hand-mask training**  
   Contributed by **@dohoon19-kim** – training pipeline can now exclude hand regions by providing `train_hand_mask/` PNGs and setting `kind: hand_mask[_multi]` in the data config.
3. **NPU conversion fixes**  
   Running PyTorch → Verisilicon **VIPNano-Ql.7120** NPU revealed that ops such as `torch.flip`, `torch.tensordot`, `torch.matmul` fail to convert.  We therefore added lightweight replacements in `saicinpainting/training/module/ffc.py`:

```python
"""
Custom tensor helpers.
Some PyTorch functions ('torch.flip', 'torch.tensordot', 'torch.matmul')
are custom re-implementations because the native ops fail during
A311D / VIPNano NPU conversion.
"""

# ----------------------------------------------------------------------

def flip_axis_static(x: torch.Tensor, axis: int) -> torch.Tensor:
    dim = axis if axis >= 0 else x.ndim + axis
    rev_idx = torch.arange(1, x.size(dim) - 1, device=x.device).flip(0)
    return x.index_select(dim, rev_idx)

# ----------------------------------------------------------------------

def matmul_linear(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if w.ndim != 2:
        raise ValueError("`w` must be (K, N)")
    vec = a.ndim == 1
    a_ = a.unsqueeze(0) if vec else a
    k, n = w.shape
    out = F.linear(a_.reshape(-1, k), w.t()).reshape(*a_.shape[:-1], n)
    return out.squeeze(0) if vec else out

# ----------------------------------------------------------------------

def tensordot_linear(a: torch.Tensor, b: torch.Tensor,
                     dims: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> torch.Tensor:
    # flatten-matmul-reshape workaround
    ...
```

4. **Project cleanup**
   * Removed bloated conda exports – now Python **3.10** + `pip install -r requirements.txt` is enough.
   * Added Bash helpers (`train.sh`, `inference.sh`, `export_to_onnx.sh`).
   * Wrote a fresh minimal README.

## How to test
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./scripts/train.sh               # training from scratch
./scripts/inference.sh -m <exp>  # sanity inference
./scripts/export_to_onnx.sh -m <exp> -c last.ckpt -o model.onnx -s
```

## Acknowledgements
Huge thanks to @dohoon19-kim for the hand-mask pipeline and to the original LaMa authors. Verisilicon support provided test hardware for the VIPNano-Ql.7120 NPU.