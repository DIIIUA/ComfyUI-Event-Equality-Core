from datetime import datetime
from .tensor_stats import summarize_latent, summarize_image, summarize_noise, summarize_conditioning


def now_run_id(prefix="run"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}"


def safe_text_signature(text):
    text = "" if text is None else str(text)
    return {
        "present": bool(text.strip()),
        "char_count": len(text),
        "word_count": len(text.split()),
        "line_count": len(text.splitlines()),
    }


def compact_stats(stats):
    if not isinstance(stats, dict):
        return {"present": False}
    keys = [
        "python_type", "shape", "dtype", "device",
        "mean", "std", "min", "max", "norm",
        "latent_format", "image_format", "noise_format",
        "length",
    ]
    return {k: stats.get(k) for k in keys if k in stats and stats.get(k) is not None}


def build_input_signatures(
    text=None,
    structured_strategy_text=None,
    latent_before=None,
    latent_after=None,
    image=None,
    noise=None,
    conditioning=None,
    decoded_image=None,
):
    signatures = {
        "text": safe_text_signature(text),
        "structured_strategy_text": safe_text_signature(structured_strategy_text),
    }

    signatures["latent_before"] = (
        {"present": True, **compact_stats(summarize_latent(latent_before))}
        if latent_before is not None else {"present": False}
    )
    signatures["latent_after"] = (
        {"present": True, **compact_stats(summarize_latent(latent_after))}
        if latent_after is not None else {"present": False}
    )
    signatures["image"] = (
        {"present": True, **compact_stats(summarize_image(image))}
        if image is not None else {"present": False}
    )
    raw_noise = noise.get("samples") if isinstance(noise, dict) and "samples" in noise else noise
    signatures["noise"] = (
        {"present": True, **compact_stats(summarize_noise(raw_noise))}
        if noise is not None else {"present": False}
    )
    signatures["conditioning"] = (
        {"present": True, **compact_stats(summarize_conditioning(conditioning))}
        if conditioning is not None else {"present": False}
    )
    signatures["decoded_image"] = (
        {"present": True, **compact_stats(summarize_image(decoded_image))}
        if decoded_image is not None else {"present": False}
    )

    return signatures


def build_passthrough_status(**kwargs):
    status = {}
    for name, value in kwargs.items():
        status[name] = {
            "present": value is not None,
            "status": "passthrough_exact" if value is not None else "missing",
            "mutation_policy": "do_not_mutate_input_object",
        }
    return status


def score_observability(signatures, has_structured=False, has_noise=False, has_sampler_boundary=False, has_decode=False, has_wan=False):
    text = signatures.get("text", {}).get("present") or signatures.get("structured_strategy_text", {}).get("present")
    technical_count = sum(
        1 for k in ["latent_before", "latent_after", "image", "noise", "conditioning", "decoded_image"]
        if signatures.get(k, {}).get("present")
    )

    score = 0.0
    level = "EMPTY"

    if text:
        score = max(score, 0.10)
        level = "TEXT_ONLY"
    if text and technical_count >= 1:
        score = max(score, 0.25)
        level = "PARTIAL"
    if technical_count >= 2:
        score = max(score, 0.40)
        level = "PARTIAL"
    if has_sampler_boundary or (signatures.get("latent_before", {}).get("present") and signatures.get("latent_after", {}).get("present")):
        score = max(score, 0.55)
        level = "BOUNDARY"
    if has_sampler_boundary and text:
        score = max(score, 0.70)
        level = "FULL_SUMMARY"
    if has_sampler_boundary and has_decode:
        score = max(score, 0.80)
        level = "FULL_SUMMARY"
    if has_structured and has_noise and has_sampler_boundary and has_decode:
        score = max(score, 0.90)
        level = "TRACE_READY"
    if has_wan and score >= 0.55:
        score = min(0.95, score + 0.03)

    return {
        "level": level,
        "score": round(float(score), 3),
        "text_present": bool(text),
        "technical_signal_count": technical_count,
        "has_structured_strategy": bool(has_structured),
        "has_noise": bool(has_noise),
        "has_sampler_boundary": bool(has_sampler_boundary),
        "has_decode": bool(has_decode),
        "has_wan_adapter": bool(has_wan),
    }


def collect_shared_targets(packet):
    target_map = {}
    for rel_id, rel in packet.get("relations", {}).items():
        for target_id in rel.get("target_signal_ids", []):
            target_map.setdefault(target_id, []).append({
                "relation_id": rel_id,
                "relation_type": rel.get("relation_type"),
                "local_strategy_id": rel.get("local_strategy_id"),
                "formula_meaning": rel.get("formula_meaning", ""),
            })
    return {k: v for k, v in target_map.items() if len(v) > 1}
