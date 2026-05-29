def is_tensor_like(x):
    return hasattr(x, "shape") and hasattr(x, "dtype")


def safe_shape(x):
    try:
        return list(x.shape)
    except Exception:
        return None


def safe_dtype(x):
    try:
        return str(x.dtype)
    except Exception:
        return None


def safe_device(x):
    try:
        return str(x.device)
    except Exception:
        return None


def safe_tensor_stats(x):
    """Return compact stats for tensor-like values without storing raw tensors."""
    stats = {
        "python_type": type(x).__name__,
        "shape": safe_shape(x),
        "dtype": safe_dtype(x),
        "device": safe_device(x),
    }

    try:
        import torch
        if isinstance(x, torch.Tensor):
            y = x.detach()
            stats["numel"] = int(y.numel())
            if y.numel() == 0:
                return stats

            yf = y.float()
            stats.update({
                "mean": float(yf.mean().item()),
                "std": float(yf.std().item()) if y.numel() > 1 else 0.0,
                "min": float(yf.min().item()),
                "max": float(yf.max().item()),
                "norm": float(torch.linalg.vector_norm(yf).item()),
                "nan_count": int(torch.isnan(yf).sum().item()),
                "inf_count": int(torch.isinf(yf).sum().item()),
            })
    except Exception as e:
        stats["stats_error"] = str(e)

    return stats


def extract_latent_samples(latent):
    if isinstance(latent, dict) and "samples" in latent:
        return latent["samples"]
    return latent


def summarize_latent(latent):
    if isinstance(latent, dict) and "samples" in latent:
        stats = safe_tensor_stats(latent["samples"])
        stats["latent_format"] = "dict_samples"
        stats["dict_keys"] = list(latent.keys())
        return stats

    stats = safe_tensor_stats(latent)
    stats["latent_format"] = "direct_or_unknown"
    return stats


def summarize_image(image):
    stats = safe_tensor_stats(image)
    stats["image_format"] = "tensor_like_or_unknown"
    return stats


def summarize_noise(noise):
    stats = safe_tensor_stats(noise)
    stats["noise_format"] = "tensor_like_or_unknown"
    return stats


def summarize_delta(delta, before=None):
    stats = safe_tensor_stats(delta)
    stats["delta_format"] = "tensor_like_or_unknown"
    if stats.get("norm") is not None:
        stats["delta_norm"] = stats.get("norm")

    try:
        before_stats = safe_tensor_stats(before) if before is not None else {}
        before_norm = before_stats.get("norm")
        if before_norm is not None:
            eps = 1e-12
            stats["before_norm"] = before_norm
            stats["relative_delta"] = float(stats.get("delta_norm", 0.0) / max(abs(before_norm), eps))
    except Exception as e:
        stats["relative_delta_error"] = str(e)

    return stats


def summarize_conditioning(cond):
    summary = {
        "python_type": type(cond).__name__,
        "shape": safe_shape(cond),
        "dtype": safe_dtype(cond),
        "device": safe_device(cond),
    }
    try:
        summary["length"] = len(cond)
    except Exception:
        summary["length"] = None

    nested = []
    try:
        if isinstance(cond, (list, tuple)):
            for i, item in enumerate(cond[:8]):
                if isinstance(item, (list, tuple)):
                    nested.append({"index": i, "type": type(item).__name__, "length": len(item)})
                elif isinstance(item, dict):
                    nested.append({"index": i, "type": "dict", "keys": list(item.keys())})
                else:
                    nested.append({"index": i, "type": type(item).__name__, "shape": safe_shape(item)})
    except Exception as e:
        summary["nested_error"] = str(e)

    if nested:
        summary["nested_summary"] = nested
    return summary


def compute_tensor_delta(before, after):
    """Try to compute after - before for tensor-like values.

    Returns (delta, error_string).
    """
    b = extract_latent_samples(before)
    a = extract_latent_samples(after)

    try:
        if safe_shape(b) != safe_shape(a):
            return None, f"shape mismatch: before={safe_shape(b)} after={safe_shape(a)}"
        delta = a - b
        return delta, ""
    except Exception as e:
        return None, str(e)
