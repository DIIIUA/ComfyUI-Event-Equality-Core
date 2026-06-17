import os
import sys

# This must be at the very top before any other imports.
# ComfyUI's custom node loader often breaks relative imports from subpackages.
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

_EVENT_HORIZON_CASCADE_CACHE = {
    "latent": None,
    "frames": None,
    "segment_index": -1
}
import copy
import csv
import json
import math
import os
import re
import inspect
import importlib
import time
import hashlib
from pathlib import Path
from datetime import datetime
from .enums import (
    DEBUG_MODES,
    DEBUG_BASIC,
    TECH_TEXT,
    TECH_LATENT,
    TECH_IMAGE,
    TECH_NOISE,
    TECH_CONDITIONING,
    TECH_DELTA,
    FORMULA_ROLES_CODE_PASS_6,
    RELATION_TYPES_CODE_PASS_6,
    EQ_UNKNOWN,
    SPACE_TEXT,
    SPACE_LATENT,
    SPACE_IMAGE,
    SPACE_NOISE,
    SPACE_CONDITIONING,
    SPACE_DELTA,
    ROLE_UNKNOWN,
    ROLE_OUTCOME_PREVIOUS,
    ROLE_OUTCOME_NEXT,
    ROLE_OBSERVED_BEHAVIOR,
    SEV_LOW,
    SEV_INFO,
    SEV_MEDIUM,
    CONFLICT_ROLE_UNKNOWN,
    CONFLICT_NO_READER_FOUND,
    CONFLICT_EMPTY_RELATION,
    CONFLICT_INVALID_SIGNAL_ID,
    CONFLICT_DELTA_UNAVAILABLE,
    CONFLICT_BOUNDARY_SHAPE_MISMATCH,
    CONFLICT_SAMPLER_BOUNDARY_SHAPE_MISMATCH,
    CONFLICT_SAMPLER_DELTA_UNAVAILABLE,
    CONFLICT_SAMPLER_METADATA_MISSING,
    CONFLICT_NO_STABLE_ANCHORS,
    CONFLICT_NO_ACTIVE_CHANGES,
    CONFLICT_NO_CONTACT_RULE,
    CONFLICT_NO_ENDPOINT,
    CONFLICT_NO_FORBIDDEN_DRIFT,
    CONFLICT_NOISE_MISSING,
    CONFLICT_NOISE_SHAPE_UNKNOWN,
    CONFLICT_NOISE_READER_FAILED,
    CONFLICT_NOISE_LATENT_SHAPE_MISMATCH,
    CONFLICT_ALPHA_ROUTE_MISSING,
    CONFLICT_ALPHA_PARTIAL_INPUT,
    CONFLICT_FROZEN_ROUTE_MISSING,
    CONFLICT_FROZEN_PARTIAL_OBSERVABILITY,
    CONFLICT_PASSTHROUGH_OUTPUT_MISSING,
    CONFLICT_PACKET_BRANCH_CREATED,
    CONFLICT_REPORT_SAVE_FAILED,
    REL_GUIDES,
    REL_CONSTRAINS,
    REL_EXPANDS_TO,
    REL_TRANSFORMS_INTO,
)
from .packet import (
    make_event_packet,
    ensure_packet,
    record_stage,
    add_signal,
    add_projection,
    add_relation,
    add_conflict,
    get_latest_signals,
    get_projection_ids_for_signal,
    signal_exists,
)
from .signal import make_event_signal
from .relation import make_event_relation
from .conflict import make_conflict
from .event_sampler import EventSamplerCore, EventSamplerWindow, EventSamplerResult
from ..reports.markdown_report import build_markdown_report
from ..resolvers.role_resolver import resolve_role
from ..resolvers.operator_registry import OperatorRegistry
from ..resolvers.s_resolver import build_sstate_from_packet
from ..utils.tensor_stats import compute_tensor_delta, extract_latent_samples, safe_shape
from ..utils.frozen_helpers import build_input_signatures, build_passthrough_status, score_observability, collect_shared_targets, now_run_id
from ..adapters.wan.wan_adapter import apply_wan_adapter

EVENT_HORIZON_RUNTIME_VERSION = "0.1.1-r113"
EVENT_HORIZON_RUNTIME_NAME = "Singularity R113 Widget Order Hotfix"
EVENT_HORIZON_BODY_VERSION = "0.1-r113"


def _event_json_safe(value, depth=0):
    if depth > 8:
        return str(type(value).__name__)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _event_json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_event_json_safe(v, depth + 1) for v in value]
    return str(value)



def _read_signal(packet, technical_type, representation_space, raw_ref, source_stage, manual_role, route_id, reader_operator, metadata=None, route_position=""):
    packet = ensure_packet(packet)
    role = resolve_role(
        technical_type,
        position_context={"stage_name": source_stage, "route_position": route_position},
        manual_role=manual_role,
    )

    conflict_ids = []
    signal = make_event_signal(
        technical_type=technical_type,
        formula_role=role,
        representation_space=representation_space,
        raw_ref=raw_ref,
        source_stage=source_stage,
        created_by=f"EventRead{technical_type.title()}",
        reader_operator=reader_operator,
        route_id=route_id,
        position_context={"route_position": route_position},
        metadata=metadata or {},
    )

    projection = OperatorRegistry().read(signal)

    if role == ROLE_UNKNOWN:
        conflict = make_conflict(
            CONFLICT_ROLE_UNKNOWN,
            severity=SEV_LOW,
            involved_signal_ids=[signal["id"]],
            stage_position=source_stage,
            suspected_cause="RoleResolver could not assign a formula role.",
            observed_symptom=f"{technical_type} resolved to UnknownRole.",
            suggested_response="Provide a manual role or improve position_context.",
        )
        packet = add_conflict(packet, conflict)
        conflict_ids.append(conflict["id"])

    if projection.get("metadata", {}).get("warning") == CONFLICT_NO_READER_FOUND or projection.get("operator_name") == "UnknownReader":
        conflict = make_conflict(
            CONFLICT_NO_READER_FOUND,
            severity=SEV_MEDIUM,
            involved_signal_ids=[signal["id"]],
            involved_projection_ids=[projection["id"]],
            stage_position=source_stage,
            suspected_cause="OperatorRegistry found no reader for this signal.",
            observed_symptom=f"No reader for technical type {technical_type}.",
            suggested_response="Add/register a ReaderOperator for this technical type.",
        )
        packet = add_conflict(packet, conflict)
        conflict_ids.append(conflict["id"])

    packet = add_signal(packet, signal)
    packet = add_projection(packet, projection)
    packet = record_stage(
        packet,
        stage_name=f"EventRead{technical_type.title()}",
        action="READ_SIGNAL",
        observed_behavior=f"{technical_type} read as {role}",
        output_signal_ids=[signal["id"]],
        projection_ids=[projection["id"]],
        conflict_ids=conflict_ids,
        formula_note=f"{technical_type} projected into EVENT_SPACE",
    )
    return packet, signal, projection, conflict_ids


def _parse_id_list(value):
    text = str(value or "").strip()
    if not text:
        return []
    if text.lower() in ("latest", "auto"):
        return []
    return [x.strip() for x in text.split(",") if x.strip()]



def _event_core_list_input_images():
    try:
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        if not os.path.isdir(input_dir):
            return ["none"]
        valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
        files = []
        for name in os.listdir(input_dir):
            path = os.path.join(input_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in valid_ext:
                files.append(name)
        files = sorted(files)
        return ["none"] + files if files else ["none"]
    except Exception:
        return ["none"]


def _event_core_list_output_folders():
    try:
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        entries = ["default"]
        if os.path.isdir(output_dir):
            for root, dirs, files in os.walk(output_dir):
                rel = os.path.relpath(root, output_dir)
                if rel != ".":
                    entries.append(rel.replace("\\", "/"))
                if len(entries) >= 100:
                    break
        return sorted(set(entries))
    except Exception:
        return ["default"]



def _event_core_preferred_media_dirs(media_type="video"):
    media_type = str(media_type or "video").lower()
    if media_type == "video":
        folder_names = ["VID"]
    elif media_type == "image":
        folder_names = ["PIC"]
    elif media_type == "report":
        folder_names = ["reports"]
    else:
        folder_names = [media_type]

    # Old D:\AI NSFW paths removed (updated 2026-06)
    roots = []
    candidates = []
    for root in roots:
        for name in folder_names:
            candidates.append(root / name)
    return candidates

from .telemetry import SingularityTelemetryMixin
from .cascade import SingularityCascadeMixin
from .execution import SingularityExecutionMixin
class WanEventWorkflowCore(SingularityExecutionMixin, SingularityTelemetryMixin, SingularityCascadeMixin):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "primary_model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "source_image_file": (_event_core_list_input_images(), {"image_upload": True}),

                "positive_prompt": ("STRING", {"default": "", "multiline": True, "height": 180, "dynamicPrompts": False}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "height": 180, "dynamicPrompts": False}),
                "event_strategy": ("STRING", {"default": "", "multiline": True, "height": 90}),

                "generation_target": (["VIDEO", "IMAGE", "AUTO"], {"default": "VIDEO"}),
                "terminal_mode": ("BOOLEAN", {"default": True}),
                "enable_continuation_outputs": ("BOOLEAN", {"default": False}),
                "execution_mode": (["RUN"], {"default": "RUN"}),
                "branch_mode": (["AUTO", "SINGLE", "DUAL_HIGH_LOW"], {"default": "AUTO"}),

                                "cascade_count": ("INT", {"default": 1, "min": 1, "max": 5}),
                "pause_after_cascade_1": ("BOOLEAN", {"default": False}),
                "pause_after_cascade_2": ("BOOLEAN", {"default": False}),
                "pause_after_cascade_3": ("BOOLEAN", {"default": False}),
                "pause_after_cascade_4": ("BOOLEAN", {"default": False}),

                "cascade_mode": (["SOLO_1", "CASCADE_2", "CASCADE_3", "CASCADE_4", "CASCADE_5"], {"default": "SOLO_1"}),
                "frames_per_cascade": ("INT", {"default": 49, "min": 1, "max": 4096}),

                "width": ("INT", {"default": 608, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 416, "min": 16, "max": 8192, "step": 8}),
                "frames": ("INT", {"default": 49, "min": 1, "max": 4096}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 240}),
                "seed": ("INT", {"default": 359, "min": 0, "max": 0xffffffffffffffff}),

                "sampler_name": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "simple"}),
                "global_steps": ("INT", {"default": 4, "min": 0, "max": 10000}),
                "primary_cfg": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "secondary_cfg": ("FLOAT", {"default": 1.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "primary_start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "primary_end_step": ("INT", {"default": 3, "min": 0, "max": 10000}),
                "secondary_start_step": ("INT", {"default": 3, "min": 0, "max": 10000}),
                "secondary_end_step": ("INT", {"default": 4, "min": 0, "max": 10000}),
                "primary_sd3_shift": ("FLOAT", {"default": 8.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "secondary_sd3_shift": ("FLOAT", {"default": 8.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),

                "decode_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "decode_overlap": ("INT", {"default": 64, "min": 0, "max": 8192, "step": 8}),
                "decode_temporal_size": ("INT", {"default": 32, "min": 1, "max": 4096}),
                "decode_temporal_overlap": ("INT", {"default": 12, "min": 0, "max": 4096}),

                "image_upscale_method": (["nearest-exact", "nearest", "bilinear", "area", "bicubic", "lanczos"], {"default": "nearest-exact"}),
                "image_crop": (["disabled", "center"], {"default": "disabled"}),

                "stage_delay_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}),

                "save_video": ("BOOLEAN", {"default": True}),
                "video_format": (["video/h264-mp4", "video/h265-mp4", "image/webp", "image/gif"], {"default": "video/h264-mp4"}),
                "force_vhs_video_combine": ("BOOLEAN", {"default": True}),
                "save_frames": ("BOOLEAN", {"default": False}),
                "save_report": ("BOOLEAN", {"default": True}),
                "output_target": (["COMFY_OUTPUT"], {"default": "COMFY_OUTPUT"}),
                "save_output_image": ("BOOLEAN", {"default": False}),
                "save_prefix": ("STRING", {"default": "Singularity"}),
                "output_folder_mode": (["DEFAULT", "PICKER", "CUSTOM"], {"default": "DEFAULT"}),
                "output_folder": (["default"],),
                "custom_output_folder": ("STRING", {"default": ""}),
                "report_detail": (["COMPACT", "STANDARD", "FULL"], {"default": "STANDARD"}),
            },
            "optional": {
                "secondary_model": ("MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    # Terminal output node.
    # IMPORTANT r15:
    # No IMAGE/LATENT outputs here. IMAGE outputs caused ComfyUI/PreviewImage-style
    # behavior and made the node look like an image-producing endpoint.
    # Video/image files are saved internally; strings are the final visible outputs.
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "status",
        "saved_video_path",
        "saved_report_path",
        "report",
        "tail_frame_0",
        "tail_frame_1",
        "tail_frame_2",
        "tail_frame_3",
        "tail_frame_4",
    )
    FUNCTION = "run"
    CATEGORY = "Singularity/Singularity"
    OUTPUT_NODE = True

    def _placeholder_image(self, width=64, height=64):
        try:
            import torch
            return torch.zeros((1, int(height), int(width), 3), dtype=torch.float32)
        except Exception:
            return None

    def _ensure_preview_image(self, image, fallback_width=64, fallback_height=64):
        if image is None:
            return self._placeholder_image(fallback_width, fallback_height)
        try:
            import torch
            if hasattr(image, "dim"):
                if image.dim() == 3:
                    return image.unsqueeze(0)
                return image
        except Exception:
            pass
        return image

    def _representative_preview_frame(self, image, fallback_width=64, fallback_height=64, mode="last"):
        """
        PreviewImage saves every image in a batch. A decoded video is usually an IMAGE batch
        shaped [frames, H, W, C]. Sending it directly to PreviewImage creates 65+ PNG files.
        This function reduces any IMAGE batch to one representative frame for UI preview.
        """
        img = self._ensure_preview_image(image, fallback_width, fallback_height)
        if img is None:
            return self._placeholder_image(fallback_width, fallback_height)
        try:
            if hasattr(img, "dim") and img.dim() == 4 and img.shape[0] > 1:
                idx = 0 if str(mode).lower() == "first" else int(img.shape[0]) - 1
                return img[idx:idx + 1]
        except Exception:
            pass
        return img

    def _load_image_from_upload(self, source_image_file, records):
        if not source_image_file or str(source_image_file).lower() in ("none", "null", ""):
            records.append({"stage": "source_image_upload", "status": "skipped", "reason": "no source_image_file"})
            return None
        try:
            import folder_paths
            import numpy as np
            import torch
            from PIL import Image, ImageOps, ImageSequence

            image_path = folder_paths.get_annotated_filepath(source_image_file)
            img = Image.open(image_path)
            frames = []
            for frame in ImageSequence.Iterator(img):
                frame = ImageOps.exif_transpose(frame)
                if frame.mode == "I":
                    frame = frame.point(lambda i: i * (1 / 255))
                frame = frame.convert("RGB")
                arr = np.array(frame).astype(np.float32) / 255.0
                frames.append(torch.from_numpy(arr)[None,])
                break

            if not frames:
                raise RuntimeError("No frame loaded from source_image_file")

            out = torch.cat(frames, dim=0)
            records.append({"stage": "source_image_upload", "status": "ok", "file": str(source_image_file), "shape": list(out.shape)})
            return out
        except Exception as e:
            records.append({"stage": "source_image_upload", "status": "failed", "file": str(source_image_file), "error": str(e)})
            return None

    def _make_ui_previews(self, source_preview, result_preview, save_prefix,
        records, include_result_preview=False):
        """
        r15: no internal PreviewImage calls.

        Reason:
        PreviewImage writes PNG files. The main terminal node must not create image
        outputs or preview dumps during video generation. Source selection is handled
        by the image picker widget; final result is saved by VHS/video export.
        """
        records.append({
            "stage": "preview_ui",
            "status": "disabled",
            "preview_policy": "no_internal_previewimage_calls_in_terminal_node",
            "reason": "avoid PNG dumps and avoid making video workflow behave like image output",
        })
        return []



    def _import_comfy_nodes(self):
        return importlib.import_module("nodes")

    def _get_node_class(self, class_name):
        comfy_nodes = self._import_comfy_nodes()
        cls = getattr(comfy_nodes, class_name, None)
        if cls is None and hasattr(comfy_nodes, "NODE_CLASS_MAPPINGS"):
            cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_name)
        if cls is None:
            raise RuntimeError(f"ComfyUI node class {class_name} not found")
        return cls

    def _call_node_method(self, class_name, method_names, *args, **kwargs):
        cls = self._get_node_class(class_name)
        inst = cls()
        if isinstance(method_names, str):
            method_names = [method_names]

        fn_name = getattr(cls, "FUNCTION", None) or getattr(inst, "FUNCTION", None)
        candidate_names = []
        for name in method_names:
            if name not in candidate_names:
                candidate_names.append(name)
        if fn_name and fn_name not in candidate_names:
            candidate_names.append(fn_name)
        for fallback in ("run", "execute", "process", "generate", "encode", "decode", "sample", "patch", "upscale", "save_images", "combine_video", "combine"):
            if fallback not in candidate_names:
                candidate_names.append(fallback)

        last_error = None
        tried = []
        for name in candidate_names:
            if hasattr(inst, name):
                tried.append(name)
                method = getattr(inst, name)
                try:
                    return method(*args, **kwargs)
                except TypeError as e:
                    last_error = e
                    try:
                        sig = inspect.signature(method)
                        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
                        return method(*args, **filtered)
                    except Exception as e2:
                        last_error = e2
                except Exception as e:
                    last_error = e

        raise RuntimeError(
            f"Could not call {class_name}. tried={tried}, requested={method_names}, "
            f"class_FUNCTION={fn_name}, last_error={last_error}"
        )


    def _describe_raw_output(self, obj, max_items=80):
        try:
            attrs = []
            for name in dir(obj):
                if name.startswith("__"):
                    continue
                attrs.append(name)
                if len(attrs) >= max_items:
                    break
            return {
                "type": f"{type(obj).__module__}.{type(obj).__name__}",
                "repr": str(obj)[:500],
                "dir_sample": attrs,
                "has_dict": hasattr(obj, "__dict__"),
                "dict_keys": list(getattr(obj, "__dict__", {}).keys())[:max_items] if hasattr(obj, "__dict__") else [],
            }
        except Exception as e:
            return {"type": str(type(obj)), "describe_error": str(e)}

    def _node_output_to_sequence(self, result, preferred_names=None, stage="node_output", records=None):
        preferred_names = preferred_names or []
        if isinstance(result, tuple) or isinstance(result, list):
            return list(result)
        if isinstance(result, dict):
            if "result" in result:
                return self._node_output_to_sequence(result["result"], preferred_names, stage, records)
            values = []
            for name in preferred_names:
                if name in result:
                    values.append(result[name])
            return values if values else list(result.values())

        values = []
        for name in preferred_names:
            if hasattr(result, name):
                try:
                    values.append(getattr(result, name))
                except Exception:
                    pass
        if values:
            if records is not None:
                records.append({"stage": f"{stage}_node_output_unwrapped", "status": "ok", "mode": "preferred_attrs", "names": preferred_names})
            return values

        for attr in ("result", "results", "value", "values", "output", "outputs", "data"):
            if hasattr(result, attr):
                try:
                    obj = getattr(result, attr)
                    seq = self._node_output_to_sequence(obj, preferred_names, stage, records)
                    if seq:
                        if records is not None:
                            records.append({"stage": f"{stage}_node_output_unwrapped", "status": "ok", "mode": attr})
                        return seq
                except Exception:
                    pass

        try:
            d = getattr(result, "__dict__", None)
            if isinstance(d, dict) and d:
                values = []
                for name in preferred_names:
                    if name in d:
                        values.append(d[name])
                if values:
                    return values
                # common private storages in API output wrappers
                for storage_name in ("_values", "_outputs", "_data", "_result", "_results"):
                    if storage_name in d:
                        try:
                            return self._node_output_to_sequence(d[storage_name], preferred_names, stage, records)
                        except Exception:
                            pass
                plain = [v for k, v in d.items() if not k.startswith("_") and not callable(v)]
                if plain:
                    return plain
                private_plain = [v for k, v in d.items() if not callable(v)]
                if private_plain:
                    return private_plain
        except Exception:
            pass

        indexed = []
        try:
            for i in range(8):
                try:
                    indexed.append(result[i])
                except Exception:
                    break
            if indexed:
                if records is not None:
                    records.append({"stage": f"{stage}_node_output_unwrapped", "status": "ok", "mode": "getitem"})
                return indexed
        except Exception:
            pass

        try:
            seq = list(result)
            if seq:
                if records is not None:
                    records.append({"stage": f"{stage}_node_output_unwrapped", "status": "ok", "mode": "iter"})
                return seq
        except Exception:
            pass

        desc = self._describe_raw_output(result)
        if records is not None:
            records.append({"stage": f"{stage}_node_output_unwrap_failed", "status": "failed", "raw_output": desc})
        raise RuntimeError(f"{stage} returned unrecognized NodeOutput/container: {desc}")

    def _first_output(self, result, preferred_names=None, stage="node_output", records=None):
        seq = self._node_output_to_sequence(result, preferred_names=preferred_names, stage=stage, records=records)
        if not seq:
            raise RuntimeError(f"{stage} returned no outputs")
        return seq[0]

    def _extract_tensors_from_obj(self, obj, prefix="root", max_items=16):
        tensors = []
        try:
            if hasattr(obj, "detach") and hasattr(obj, "shape"):
                tensors.append((prefix, obj))
                return tensors
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if len(tensors) >= max_items:
                        break
                    tensors.extend(self._extract_tensors_from_obj(v, f"{prefix}.{k}", max_items=max_items-len(tensors)))
                return tensors
            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    if len(tensors) >= max_items:
                        break
                    tensors.extend(self._extract_tensors_from_obj(v, f"{prefix}[{i}]", max_items=max_items-len(tensors)))
                return tensors
        except Exception:
            pass
        return tensors

    def _event_tensor_probe(self, obj, *, label="", max_tensors=4, sample_size=128):
        """
        Report-only tensor fingerprint used for raw-vs-Singularity route parity.
        It samples a tiny, deterministic subset of tensor values and records shape
        and statistics without mutating the object or moving the full tensor.
        """
        probe = {
            "label": str(label or ""),
            "object_present": obj is not None,
            "object_type": type(obj).__name__ if obj is not None else "NoneType",
            "object_module": type(obj).__module__ if obj is not None else "",
            "tensor_count": 0,
            "tensors": [],
            "probe_policy": "metadata_and_small_sample_only_no_tensor_mutation",
        }
        try:
            import torch
            tensors = self._extract_tensors_from_obj(obj, max_items=int(max_tensors))
            probe["tensor_count"] = len(tensors)
            for tensor_path, tensor in tensors[: int(max_tensors)]:
                try:
                    td = tensor.detach()
                    tf = td.float()
                    safe = torch.nan_to_num(tf, nan=0.0, posinf=0.0, neginf=0.0)
                    finite = torch.isfinite(tf)
                    total = int(tf.numel())
                    finite_count = int(finite.sum().item()) if total else 0
                    flat = safe.reshape(-1)
                    sample_hash = ""
                    sample_count = 0
                    if total:
                        sample_count = min(int(sample_size), total)
                        if sample_count == 1:
                            sample = flat[:1]
                        else:
                            idx = torch.linspace(
                                0,
                                total - 1,
                                steps=sample_count,
                                device=flat.device,
                            ).round().long().clamp_(0, total - 1)
                            sample = flat[idx]
                        sample_np = sample.detach().cpu().numpy().astype("float32", copy=False)
                        sample_hash = hashlib.sha256(sample_np.tobytes()).hexdigest()[:16]

                    tensor_probe = {
                        "path": str(tensor_path),
                        "shape": [int(x) for x in list(td.shape)],
                        "dtype": str(getattr(td, "dtype", "")),
                        "device": str(getattr(td, "device", "")),
                        "total": total,
                        "finite_ratio": float(finite_count / total) if total else 1.0,
                        "mean": float(safe.mean().item()) if total else 0.0,
                        "std": float(safe.std().item()) if total > 1 else 0.0,
                        "min": float(safe.min().item()) if total else 0.0,
                        "max": float(safe.max().item()) if total else 0.0,
                        "norm": float(torch.linalg.vector_norm(safe).item()) if total else 0.0,
                        "sample_count": sample_count,
                        "sample_hash": sample_hash,
                    }
                    stable = {
                        "path": tensor_probe["path"],
                        "shape": tensor_probe["shape"],
                        "dtype": tensor_probe["dtype"],
                        "total": tensor_probe["total"],
                        "finite_ratio": round(tensor_probe["finite_ratio"], 8),
                        "mean": round(tensor_probe["mean"], 8),
                        "std": round(tensor_probe["std"], 8),
                        "min": round(tensor_probe["min"], 8),
                        "max": round(tensor_probe["max"], 8),
                        "norm": round(tensor_probe["norm"], 8),
                        "sample_hash": sample_hash,
                    }
                    tensor_probe["route_signature"] = hashlib.sha256(
                        json.dumps(stable, sort_keys=True, default=str).encode("utf-8", errors="ignore")
                    ).hexdigest()[:16]
                    probe["tensors"].append(tensor_probe)
                except Exception as tensor_exc:
                    probe["tensors"].append({
                        "path": str(tensor_path),
                        "status": "inspect_failed",
                        "error": str(tensor_exc)[:240],
                    })
        except Exception as exc:
            probe["status"] = "failed"
            probe["error"] = str(exc)[:240]

        stable_probe = {
            "object_type": probe.get("object_type"),
            "object_module": probe.get("object_module"),
            "tensor_count": probe.get("tensor_count"),
            "tensor_signatures": [
                t.get("route_signature", "")
                for t in probe.get("tensors", [])
                if isinstance(t, dict)
            ],
        }
        probe["route_signature"] = hashlib.sha256(
            json.dumps(stable_probe, sort_keys=True, default=str).encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        probe.setdefault("status", "recorded")
        return probe

    def _record_raw_vs_singularity_parity_probe(
        self,
        records,
        stage,
        *,
        route_kind,
        input_state=None,
        output_state=None,
        reference_state=None,
        metadata=None,
    ):
        record = {
            "stage": str(stage),
            "status": "recorded",
            "probe_version": "raw_vs_singularity_parity_probe_v1",
            "route_kind": str(route_kind),
            "input_probe": self._event_tensor_probe(input_state, label=f"{stage}.input") if input_state is not None else None,
            "output_probe": self._event_tensor_probe(output_state, label=f"{stage}.output") if output_state is not None else None,
            "reference_probe": self._event_tensor_probe(reference_state, label=f"{stage}.reference") if reference_state is not None else None,
            "metadata": _event_json_safe(metadata or {}),
            "control_mode": "REPORT_ONLY",
            "formula": (
                "Raw-vs-Singularity parity reads the event boundary as "
                "OutcomePrevious + ObservedBehavior -> StrategyCarrier -> OutcomeNext, "
                "without mutating tensors."
            ),
        }
        records.append(record)
        return record

    def _record_sampler_route_parity_probe(
        self,
        records,
        stage,
        *,
        branch_name,
        route_variant,
        latent_before=None,
        latent_after=None,
        model=None,
        seed=None,
        steps=None,
        cfg=None,
        sampler_name=None,
        scheduler=None,
        start_at_step=None,
        end_at_step=None,
        add_noise=None,
        return_leftover_noise=None,
        sd3_shift=None,
        segment_index=None,
        extra=None,
    ):
        try:
            model_probe = self._probe_operator_route_object(model, f"{branch_name}_model") if model is not None else None
        except Exception as exc:
            model_probe = {"status": "failed", "error": str(exc)[:240]}
        return self._record_raw_vs_singularity_parity_probe(
            records,
            stage,
            route_kind="sampler_window",
            input_state=latent_before,
            output_state=latent_after,
            metadata={
                "branch_name": str(branch_name),
                "route_variant": str(route_variant),
                "segment_index": int(segment_index) if segment_index is not None else None,
                "seed": int(seed) if seed is not None else None,
                "steps": int(steps) if steps is not None else None,
                "cfg": float(cfg) if cfg is not None else None,
                "sampler_name": str(sampler_name) if sampler_name is not None else "",
                "scheduler": str(scheduler) if scheduler is not None else "",
                "start_at_step": int(start_at_step) if start_at_step is not None else None,
                "end_at_step": int(end_at_step) if end_at_step is not None else None,
                "add_noise": str(add_noise) if add_noise is not None else "",
                "return_leftover_noise": str(return_leftover_noise) if return_leftover_noise is not None else "",
                "sd3_shift": float(sd3_shift) if sd3_shift is not None else None,
                "model_probe": model_probe,
                "extra": extra or {},
            },
        )

    def _finite_guard(self, obj, records, stage, strict=True):
        """
        Detect NaN/Inf tensors before black-video save.
        Returns True if all discovered tensors are finite.
        """
        try:
            import torch
            tensors = self._extract_tensors_from_obj(obj)
            if not tensors:
                records.append({"stage": stage, "status": "no_tensor_found"})
                return True
            all_ok = True
            reports = []
            for name, t in tensors:
                try:
                    td = t.detach()
                    finite = torch.isfinite(td)
                    total = int(td.numel())
                    finite_count = int(finite.sum().item()) if total else 0
                    nonfinite_count = total - finite_count
                    finite_ratio = float(finite_count / total) if total else 1.0
                    reports.append({
                        "tensor": name,
                        "shape": list(td.shape),
                        "dtype": str(td.dtype),
                        "device": str(td.device),
                        "total": total,
                        "nonfinite_count": nonfinite_count,
                        "finite_ratio": finite_ratio,
                    })
                    if nonfinite_count > 0:
                        all_ok = False
                except Exception as e:
                    reports.append({"tensor": name, "status": "inspect_failed", "error": str(e)})
            records.append({"stage": stage, "status": "ok" if all_ok else "nonfinite_detected", "reports": reports})
            if strict and not all_ok:
                raise RuntimeError(f"{stage}: non-finite tensor values detected; refusing to save black/corrupt video")
            return all_ok
        except Exception as e:
            records.append({"stage": stage, "status": "failed", "error": str(e)})
            if strict:
                raise
            return False

    def _encode_text(self, clip, text, records=None, label="TextEncode"):
        result = self._call_node_method("CLIPTextEncode", ["encode"], clip=clip, text=str(text or ""))
        conditioning = self._first_output(result, preferred_names=["conditioning", "CONDITIONING"], stage=f"Event{label}")
        if records is not None:
            self._event_universal_stage_math(
                records,
                f"Event{label}",
                input_state=text,
                output_state=conditioning,
                observed_behavior="text prompt encoded into conditioning strategy",
                formula_role="TEXT/CLIP -> CONDITIONING as NumericStrategy",
                route_id=f"route_{str(label).lower()}",
                next_requirement="WanImageToVideo and sampler require conditioning compatible with model route",
                control_mode="REPORT_ONLY",
                metadata={"text_length": len(str(text or ""))},
            )
        return conditioning

    def _scale_image(self, image, width, height, method, crop, records, segment_index=None, route_label=""):
        if image is None:
            records.append({"stage": "EventImageScaleStart", "status": "skipped", "reason": "no image"})
            return None
        crop_mode = str(crop or "wan_native")
        crop_mode_normalized = crop_mode.lower()
        if crop_mode_normalized in ("wan_native", "wan-native", "native", "passthrough", "pass_through"):
            records.append({
                "stage": "EventImageScaleStart",
                "status": "passthrough_wan_native",
                "width": width,
                "height": height,
                "method": str(method or "nearest-exact"),
                "crop": crop_mode,
                "formula": "SourceAnchor is preserved; WanImageToVideo owns target-grid center normalization.",
            })
            self._event_universal_stage_math(
                records,
                "EventImageScaleStart",
                input_state=image,
                output_state=image,
                observed_behavior="source image passed through unchanged before WanImageToVideo native resize/crop",
                formula_role="IMAGE SourceAnchor -> IMAGE preserved OutcomePrevious",
                route_id="route_source_image",
                next_requirement="WanImageToVideo will normalize SourceAnchor to the requested width/height latent grid",
                control_mode="REPORT_ONLY",
                metadata={
                    "wan_width": int(width),
                    "wan_height": int(height),
                    "method": str(method or "nearest-exact"),
                    "crop": crop_mode,
                    "external_image_scale": False,
                },
            )
            self._record_raw_vs_singularity_parity_probe(
                records,
                "EventRawVsSingularityParity_SourcePreprocess",
                route_kind="source_image_preprocess",
                input_state=image,
                output_state=image,
                metadata={
                    "segment_index": int(segment_index) if segment_index is not None else None,
                    "route_label": str(route_label or ""),
                    "wan_width": int(width),
                    "wan_height": int(height),
                    "method": str(method or "nearest-exact"),
                    "crop": crop_mode,
                    "external_image_scale": False,
                    "interpretation": "ImageScale is intentionally skipped; WanImageToVideo receives the original source tensor.",
                },
            )
            return image
        try:
            result = self._call_node_method(
                "ImageScale",
                ["upscale"],
                image=image,
                upscale_method=str(method or "nearest-exact"),
                width=int(width),
                height=int(height),
                crop=crop_mode_normalized if crop_mode_normalized in ("disabled", "center") else "disabled",
            )
            scaled = self._first_output(result, preferred_names=["image", "images", "IMAGE"], stage="EventImageScaleStart", records=records)
            records.append({"stage": "EventImageScaleStart", "status": "ok", "width": width, "height": height})
            self._event_universal_stage_math(
                records,
                "EventImageScaleStart",
                input_state=image,
                output_state=scaled,
                observed_behavior="source image resized/cropped into model-compatible source outcome",
                formula_role="IMAGE SourceAnchor -> IMAGE scaled OutcomePrevious",
                route_id="route_source_image",
                next_requirement="WanImageToVideo requires scaled image, width, height, frame count and conditioning",
                control_mode="REPORT_ONLY",
                metadata={"width": int(width), "height": int(height), "method": str(method or "nearest-exact"), "crop": crop_mode, "external_image_scale": True},
            )
            self._record_raw_vs_singularity_parity_probe(
                records,
                "EventRawVsSingularityParity_SourcePreprocess",
                route_kind="source_image_preprocess",
                input_state=image,
                output_state=scaled,
                metadata={
                    "segment_index": int(segment_index) if segment_index is not None else None,
                    "route_label": str(route_label or ""),
                    "wan_width": int(width),
                    "wan_height": int(height),
                    "method": str(method or "nearest-exact"),
                    "crop": crop_mode,
                    "external_image_scale": True,
                    "interpretation": "ImageScale produced the source tensor handed to WanImageToVideo.",
                },
            )
            return scaled
        except Exception as e:
            records.append({"stage": "EventImageScaleStart", "status": "failed_no_fake_success", "error": str(e)})
            raise

    def _apply_sd3_shift(self, model, shift, label, records):
        try:
            shift_value = float(shift or 0.0)
        except Exception:
            shift_value = 0.0
        if abs(shift_value) < 1e-9:
            records.append({
                "stage": f"EventModelShift_{label}",
                "status": "skipped_zero_shift",
                "shift": shift_value,
                "formula": "shift=0 -> original model passthrough; do not apply ModelSamplingSD3 patch",
            })
            self._event_universal_stage_math(
                records,
                f"EventModelShift_{label}",
                input_state=model,
                output_state=model,
                observed_behavior="shift value is zero; model route passes through unchanged",
                formula_role="MODEL Operator -> MODEL Operator passthrough",
                route_id=f"route_model_shift_{label}",
                next_requirement="sampler requires model route compatible with current branch",
                control_mode="REPORT_ONLY",
                metadata={"shift": shift_value, "applied": False},
            )
            return model
        try:
            result = self._call_node_method("ModelSamplingSD3", ["patch"], model=model, shift=shift_value)
            patched = self._first_output(result, preferred_names=["model", "MODEL"], stage=f"EventModelShift_{label}", records=records)
            records.append({"stage": f"EventModelShift_{label}", "status": "ok", "shift": shift_value, "formula": "model + shift behavior = EventSingularity_model_shift = shifted model"})
            self._event_universal_stage_math(
                records,
                f"EventModelShift_{label}",
                input_state=model,
                output_state=patched,
                observed_behavior="ModelSamplingSD3 shift patches model route for sampler branch",
                formula_role="MODEL Operator + shift behavior -> MODEL shifted Operator",
                route_id=f"route_model_shift_{label}",
                next_requirement="sampler requires shifted model route compatible with branch window",
                control_mode="REPORT_ONLY",
                metadata={"shift": shift_value, "applied": True},
            )
            return patched
        except Exception as e:
            records.append({"stage": f"event_model_shift_{label}", "status": "failed", "shift": shift_value, "error": str(e)})
            raise

    def _wan_image_to_video(self, positive, negative, vae, start_image, width, height, frames, batch_size, records, segment_index=None, route_label=""):
        call_signature = "positive_negative_vae_start_image_width_height_length_batch_size"
        first_error_text = ""
        try:
            result = self._call_node_method(
                "WanImageToVideo",
                ["encode", "generate", "process", "run"],
                positive=positive,
                negative=negative,
                vae=vae,
                clip_vision_output=None,
                start_image=start_image,
                width=int(width),
                height=int(height),
                length=int(frames),
                batch_size=int(batch_size),
            )
        except Exception as first_error:
            call_signature = "retry_with_image_alias_and_frames_length"
            first_error_text = str(first_error)[:300]
            records.append({"stage": "EventWanImageToVideoSeed_first_signature_failed", "status": "retrying", "error": str(first_error)})
            result = self._call_node_method(
                "WanImageToVideo",
                ["generate", "process", "run", "encode"],
                positive=positive,
                negative=negative,
                vae=vae,
                clip_vision_output=None,
                image=start_image,
                start_image=start_image,
                width=int(width),
                height=int(height),
                frames=int(frames),
                length=int(frames),
                batch_size=int(batch_size),
            )

        seq = self._node_output_to_sequence(
            result,
            preferred_names=["positive", "negative", "latent", "conditioning", "conditioning_positive", "conditioning_negative", "samples"],
            stage="EventWanImageToVideoSeed",
            records=records,
        )
        if len(seq) < 3:
            raise RuntimeError(
                f"WanImageToVideo returned fewer than 3 usable outputs after unwrapping: "
                f"count={len(seq)}, raw={self._describe_raw_output(result)}"
            )

        records.append({
            "stage": "EventWanImageToVideoSeed",
            "status": "ok",
            "width": width,
            "height": height,
            "frames": frames,
            "formula": "source image + conditioning + VAE behavior = EventSingularity_i2v = video latent seed",
            "event_node": "EventWanImageToVideo",
            "unwrapped_output_count": len(seq),
        })
        self._event_universal_stage_math(
            records,
            "EventWanImageToVideoSeed",
            input_state=start_image,
            output_state=seq[2],
            observed_behavior="scaled source image, positive conditioning, VAE and dimensions created initial video latent seed",
            formula_role="IMAGE + CONDITIONING + VAE -> LATENT seed OutcomePrevious",
            route_id="route_wan_i2v_seed",
            next_requirement="High sampler requires latent seed as previous outcome and noise-enabled first denoise window",
            control_mode="REPORT_ONLY",
            metadata={"width": int(width), "height": int(height), "frames": int(frames), "batch_size": int(batch_size)},
        )
        self._record_raw_vs_singularity_parity_probe(
            records,
            "EventRawVsSingularityParity_WanLatentInit",
            route_kind="wan_image_to_video_latent_init",
            input_state=start_image,
            output_state=seq[2],
            metadata={
                "segment_index": int(segment_index) if segment_index is not None else None,
                "route_label": str(route_label or ""),
                "width": int(width),
                "height": int(height),
                "frames": int(frames),
                "batch_size": int(batch_size),
                "call_signature": call_signature,
                "first_signature_error": first_error_text,
                "unwrapped_output_count": len(seq),
                "positive_conditioning_probe": self._event_tensor_probe(seq[0], label="wan_positive_conditioning"),
                "negative_conditioning_probe": self._event_tensor_probe(seq[1], label="wan_negative_conditioning"),
                "vae_route_signature": self._object_route_cache_signature(vae),
            },
        )
        return seq[0], seq[1], seq[2]


    def _low_level_sampler_operation(self, *, model, positive, negative, latent, seed, steps, cfg, sampler_name, scheduler, start_at_step, end_at_step, add_noise, return_leftover_noise):
        result = self._call_node_method(
            "KSamplerAdvanced",
            ["sample"],
            model=model,
            add_noise=add_noise,
            noise_seed=int(seed),
            steps=int(steps),
            cfg=float(cfg),
            sampler_name=str(sampler_name or "euler"),
            scheduler=str(scheduler or "simple"),
            positive=positive,
            negative=negative,
            latent_image=latent,
            start_at_step=int(start_at_step),
            end_at_step=int(end_at_step),
            return_with_leftover_noise=return_leftover_noise,
        )
        return self._first_output(result, preferred_names=["latent", "samples", "LATENT"], stage="EventSamplerLowLevel")

    def _tensor_from_latent_like(self, obj):
        try:
            if isinstance(obj, dict):
                if "samples" in obj:
                    return obj["samples"]
                for v in obj.values():
                    t = self._tensor_from_latent_like(v)
                    if t is not None:
                        return t
            if hasattr(obj, "detach") and hasattr(obj, "shape"):
                return obj
            if isinstance(obj, (list, tuple)) and obj:
                return self._tensor_from_latent_like(obj[0])
        except Exception:
            return None
        return None







