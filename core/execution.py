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
from .prompt_strategy import build_prompt_strategy_packet
import os
import sys

# Ensure parent directory is in sys.path so relative imports work
# when loaded by ComfyUI's custom node loader
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from ..reports.markdown_report import build_markdown_report
from ..resolvers.role_resolver import resolve_role
from ..resolvers.operator_registry import OperatorRegistry
from ..resolvers.s_resolver import build_sstate_from_packet
from ..utils.tensor_stats import compute_tensor_delta, extract_latent_samples, safe_shape
from ..utils.frozen_helpers import build_input_signatures, build_passthrough_status, score_observability, collect_shared_targets, now_run_id
from ..adapters.wan.wan_adapter import apply_wan_adapter

try:
    from server import PromptServer
    from aiohttp import web
    from comfy.model_management import InterruptProcessingException, throw_exception_if_processing_interrupted
except Exception:
    PromptServer = None
    web = None
    InterruptProcessingException = RuntimeError

    def throw_exception_if_processing_interrupted():
        return None

EVENT_HORIZON_RUNTIME_VERSION = "0.1.1-r113"
EVENT_HORIZON_RUNTIME_NAME = "Singularity R113 Widget Order Hotfix"
EVENT_HORIZON_BODY_VERSION = "0.1-r113"
TAIL_CANDIDATE_COUNT = 5
_SINGULARITY_PAUSE_STATES = {}


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


def _singularity_pause_key(node_id):
    text = str(node_id or "global").strip()
    return text or "global"


def _singularity_parse_resume_index(value, default=-1):
    try:
        return int(value)
    except Exception:
        return int(default)


def _singularity_prompt_payload(payload):
    if not isinstance(payload, dict):
        return {}
    has_prompt_payload = (
        payload.get("prompt_payload_version") == "cascade_continue_prompt_v1"
        or "positive_prompt" in payload
        or "negative_prompt" in payload
        or "prompt_transcode_mode" in payload
    )
    if not has_prompt_payload:
        return {}
    update = {
        "payload_version": str(payload.get("prompt_payload_version") or "cascade_continue_prompt_v1"),
        "source": str(payload.get("prompt_source") or "node_widgets_at_continue_click"),
        "positive_prompt_present": "positive_prompt" in payload,
        "negative_prompt_present": "negative_prompt" in payload,
        "positive_prompt": str(payload.get("positive_prompt", "")),
        "negative_prompt": str(payload.get("negative_prompt", "")),
    }
    if "prompt_transcode_mode" in payload:
        update["prompt_transcode_mode"] = str(payload.get("prompt_transcode_mode") or "")
    return update


if PromptServer is not None and web is not None:
    @PromptServer.instance.routes.post("/singularity/cascade/continue/{node_id}")
    async def _singularity_handle_cascade_continue(request):
        node_id = _singularity_pause_key(request.match_info.get("node_id"))
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        resume_frame_index = _singularity_parse_resume_index(
            payload.get("resume_frame_index", -1),
            default=-1,
        )
        prompt_update = _singularity_prompt_payload(payload)
        _SINGULARITY_PAUSE_STATES[node_id] = {
            "status": "continue",
            "resume_frame_index": resume_frame_index,
            "prompt_update": prompt_update,
            "prompt_update_present": bool(prompt_update),
            "updated_at": time.time(),
        }
        return web.json_response({
            "status": "ok",
            "node_id": node_id,
            "resume_frame_index": resume_frame_index,
            "prompt_update_present": bool(prompt_update),
        })

    @PromptServer.instance.routes.post("/singularity/cascade/cancel/{node_id}")
    async def _singularity_handle_cascade_cancel(request):
        node_id = _singularity_pause_key(request.match_info.get("node_id"))
        _SINGULARITY_PAUSE_STATES[node_id] = {
            "status": "cancelled",
            "updated_at": time.time(),
        }
        return web.json_response({"status": "ok", "node_id": node_id})

    @PromptServer.instance.routes.post("/singularity/cascade/cancel")
    async def _singularity_handle_cascade_cancel_all(request):
        touched = []
        if _SINGULARITY_PAUSE_STATES:
            for node_id in list(_SINGULARITY_PAUSE_STATES.keys()):
                _SINGULARITY_PAUSE_STATES[node_id] = {
                    "status": "cancelled",
                    "updated_at": time.time(),
                }
                touched.append(node_id)
        return web.json_response({"status": "ok", "node_ids": touched})

    @PromptServer.instance.routes.get("/singularity/cascade/status/{node_id}")
    async def _singularity_handle_cascade_status(request):
        node_id = _singularity_pause_key(request.match_info.get("node_id"))
        state = _SINGULARITY_PAUSE_STATES.get(node_id, {})
        return web.json_response({
            "node_id": node_id,
            "state": _event_json_safe(state),
        })



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


class SingularityExecutionMixin:
    def _save_image_attempt(self, image, save_prefix, records):
        result = self._call_node_method("SaveImage", ["save_images"], images=image, filename_prefix=str(save_prefix or "Singularity"))
        records.append({"stage": "EventSaveFrames", "status": "ok", "result": str(result)[:300]})
        self._event_universal_stage_math(
            records,
            "EventSaveFrames",
            input_state=image,
            output_state=str(result)[:300],
            observed_behavior="decoded frame/image tensor saved as image assets",
            formula_role="IMAGE/FRAMES VisibleOutcome -> FILE output reference",
            route_id="route_save_frames",
            next_requirement="saved image assets should match decoded visible outcome",
            control_mode="REPORT_ONLY",
            metadata={"save_prefix": str(save_prefix or "Singularity")},
        )
        return str(result)

    def _normalize_video_path_candidate(self, candidate):
        if candidate is None:
            return ""
        c = str(candidate)
        if not c:
            return ""
        exts = (".mp4", ".webp", ".gif", ".mov", ".mkv")
        if not c.lower().endswith(exts):
            return ""

        # If absolute, return it even if the file existence check fails during Comfy async timing.
        if os.path.isabs(c):
            return c

        try:
            import folder_paths
            output_dir = Path(folder_paths.get_output_directory())
        except Exception:
            output_dir = Path.cwd() / "output"

        direct = output_dir / c
        if direct.exists():
            return str(direct)

        # VHS sometimes gives only filename and subfolder separately; search by basename.
        if output_dir.exists():
            matches = list(output_dir.rglob(Path(c).name))
            if matches:
                return str(max(matches, key=lambda x: x.stat().st_mtime))

        # As a last resort return the candidate, but report will show it was unresolved.
        return c

    def _extract_path_from_video_result(self, result, save_prefix, records, *args, **kwargs):
        """
        r19: extract path only from standard VHS return / standard ComfyUI output.
        """
        raw_description = str(result)[:1000]

        try:
            import folder_paths
            output_dir = Path(folder_paths.get_output_directory())
        except Exception:
            output_dir = Path.cwd() / "output"

        try:
            if isinstance(result, dict):
                gifs = result.get("ui", {}).get("gifs", []) or []
                for item in gifs:
                    if isinstance(item, dict):
                        fullpath = item.get("fullpath")
                        path = self._normalize_video_path_candidate(fullpath)
                        if path:
                            records.append({"stage": "EventVideoCombine_extract_path", "status": "ok", "mode": "ui.gifs.fullpath", "path": path})
                            return path
                        filename = item.get("filename")
                        subfolder = item.get("subfolder", "")
                        if filename:
                            candidate = output_dir / str(subfolder) / str(filename)
                            path = self._normalize_video_path_candidate(str(candidate))
                            if path:
                                records.append({"stage": "EventVideoCombine_extract_path", "status": "ok", "mode": "ui.gifs.filename", "path": path})
                                return path
        except Exception as e:
            records.append({"stage": "EventVideoCombine_extract_path", "status": "ui_parse_failed", "error": str(e)})

        candidates = []
        def visit(obj, depth=0):
            if obj is None or depth > 8:
                return
            if isinstance(obj, str):
                candidates.append(obj)
                return
            if isinstance(obj, (list, tuple, set)):
                for x in obj:
                    visit(x, depth + 1)
                return
            if isinstance(obj, dict):
                for key in ("fullpath", "path", "filename", "file", "files", "result", "results", "ui", "gifs", "images"):
                    if key in obj:
                        visit(obj[key], depth + 1)
                return
            for attr in ("fullpath", "path", "filename", "file", "files", "result", "results", "value", "values", "output", "outputs", "data"):
                if hasattr(obj, attr):
                    try:
                        visit(getattr(obj, attr), depth + 1)
                    except Exception:
                        pass
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                visit(d, depth + 1)

        visit(result)
        for ext in (".mp4", ".mov", ".mkv", ".webp", ".gif"):
            for c in candidates:
                if isinstance(c, str) and c.lower().endswith(ext):
                    path = self._normalize_video_path_candidate(c)
                    if path:
                        records.append({"stage": "EventVideoCombine_extract_path", "status": "ok", "mode": "recursive_standard_vhs", "path": path})
                        return path

        newest = self._find_latest_standard_output_video(save_prefix, records)
        if newest:
            return newest

        records.append({
            "stage": "EventVideoCombine_extract_path",
            "status": "not_found",
            "raw_result_type": f"{type(result).__module__}.{type(result).__name__}",
            "raw_result": raw_description,
            "candidates": [str(x)[:200] for x in candidates[:20]],
        })
        return ""


    def _get_first_available_node_class(self, class_names, records=None, stage="node_class_lookup"):
        errors = {}
        for class_name in class_names:
            try:
                cls = self._get_node_class(class_name)
                if records is not None:
                    records.append({
                        "stage": stage,
                        "status": "found",
                        "class_name": class_name,
                        "class": str(cls),
                    })
                return class_name, cls
            except Exception as e:
                errors[class_name] = str(e)
        if records is not None:
            records.append({
                "stage": stage,
                "status": "not_found",
                "class_names": list(class_names),
                "errors": errors,
            })
        raise RuntimeError(f"No candidate node class found for {class_names}; errors={errors}")

    def _call_candidate_node_method(self, class_names, method_names, records=None, stage="candidate_node_call", *args, **kwargs):
        last_error = None
        tried = []
        for class_name in class_names:
            try:
                # Check class exists first so report shows whether VHS is installed / mapped.
                self._get_first_available_node_class([class_name], records=records, stage=f"{stage}_class_check")
                tried.append(class_name)
                return self._call_node_method(class_name, method_names, *args, **kwargs)
            except Exception as e:
                last_error = e
                if records is not None:
                    records.append({
                        "stage": stage,
                        "status": "candidate_failed",
                        "class_name": class_name,
                        "error": str(e),
                    })
        raise RuntimeError(f"All candidate node calls failed. tried={tried}; last_error={last_error}")

    def _frames_to_uint8_numpy(self, image, records=None, stage="frames_to_uint8"):
        try:
            import numpy as np
            frames = image
            if not hasattr(frames, "detach"):
                raise RuntimeError(f"{stage} expected torch-like IMAGE batch, got {type(frames)}")
            arr = frames.detach().cpu().numpy()
            if arr.ndim != 4:
                raise RuntimeError(f"{stage} expected IMAGE batch [N,H,W,C], got shape={getattr(arr, 'shape', None)}")
            finite = np.isfinite(arr)
            nonfinite_count = int(arr.size - finite.sum())
            if nonfinite_count > 0:
                raise RuntimeError(f"{stage} refusing to encode non-finite frames: nonfinite_count={nonfinite_count}, shape={arr.shape}")
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype("uint8")
            if records is not None:
                records.append({"stage": stage, "status": "ok", "shape": list(arr.shape), "nonfinite_count": 0})
            return arr
        except Exception as e:
            if records is not None:
                records.append({"stage": stage, "status": "failed", "error": str(e)})
            raise

    def _fallback_animated_export(self, image, fps, save_prefix, video_format, records, output_target="USER_D_AI_NSFW", output_folder_mode="DEFAULT", output_folder="default", custom_output_folder=""):
        """
        Emergency media writer if VHS is absent or incompatible.
        Tries mp4 first, then animated WebP. It must physically write a file or return "".
        """
        try:
            arr = self._frames_to_uint8_numpy(image, records=records, stage="EventVideoFallback_frames_to_numpy")
            output_dir = self._resolve_output_dir(output_folder_mode, output_folder, custom_output_folder, subdir="", output_target=output_target, media_type="video")
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_prefix = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", str(save_prefix or "Singularity")).strip("_") or "Singularity"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # 1) imageio v3 / ffmpeg if available.
            if "mp4" in str(video_format).lower():
                try:
                    import imageio.v3 as iio
                    path = output_dir / f"{safe_prefix}_{ts}_fallback_imageio.mp4"
                    iio.imwrite(str(path), arr, fps=float(fps), codec="libx264")
                    if path.exists() and path.stat().st_size > 0:
                        records.append({"stage": "EventVideoFallbackImageIO", "status": "ok", "path": str(path), "size": path.stat().st_size})
                        return str(path)
                    records.append({"stage": "EventVideoFallbackImageIO", "status": "wrote_no_file_or_empty", "path": str(path)})
                except Exception as e:
                    records.append({"stage": "EventVideoFallbackImageIO", "status": "failed", "error": str(e)})

                # 2) OpenCV mp4 if available.
                try:
                    import cv2
                    path = output_dir / f"{safe_prefix}_{ts}_fallback_cv2.mp4"
                    h, w = int(arr.shape[1]), int(arr.shape[2])
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("cv2.VideoWriter did not open")
                    for frame in arr:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    writer.release()
                    if path.exists() and path.stat().st_size > 0:
                        records.append({"stage": "EventVideoFallbackCV2", "status": "ok", "path": str(path), "size": path.stat().st_size})
                        return str(path)
                    records.append({"stage": "EventVideoFallbackCV2", "status": "wrote_no_file_or_empty", "path": str(path)})
                except Exception as e:
                    records.append({"stage": "EventVideoFallbackCV2", "status": "failed", "error": str(e)})

            # 3) Animated WebP fallback.
            try:
                from PIL import Image
                path = output_dir / f"{safe_prefix}_{ts}_fallback.webp"
                pil_frames = [Image.fromarray(frame) for frame in arr]
                duration = int(1000 / max(1, float(fps)))
                pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=duration, loop=0, lossless=True)
                if path.exists() and path.stat().st_size > 0:
                    records.append({"stage": "EventVideoFallbackWebP", "status": "ok", "path": str(path), "size": path.stat().st_size})
                    return str(path)
                records.append({"stage": "EventVideoFallbackWebP", "status": "wrote_no_file_or_empty", "path": str(path)})
            except Exception as e:
                records.append({"stage": "EventVideoFallbackWebP", "status": "failed", "error": str(e)})

            return ""
        except Exception as e:
            records.append({"stage": "EventVideoFallback", "status": "failed", "error": str(e)})
            return ""


    def _extract_vhs_ui_payload_from_records(self, records):
        """
        r21: recover the VHS UI payload so ComfyUI can display the generated video
        in the node UI without adding IMAGE outputs.
        """
        try:
            import ast
            for rec in reversed(records or []):
                if rec.get("stage") == "EventVideoCombine" and rec.get("raw_vhs_result"):
                    raw = rec.get("raw_vhs_result")
                    try:
                        payload = ast.literal_eval(raw)
                    except Exception:
                        payload = None
                    if isinstance(payload, tuple) and len(payload) > 0:
                        payload = payload[0]
                    if isinstance(payload, dict) and isinstance(payload.get("ui"), dict):
                        ui = payload.get("ui")
                        if "gifs" in ui:
                            return {"gifs": ui.get("gifs")}
                        if "images" in ui:
                            return {"images": ui.get("images")}
        except Exception:
            pass
        return None

    def _video_combine_attempt(self, image, fps, save_prefix, records, video_format="video/h264-mp4", force_vhs=True, output_target="COMFY_OUTPUT", output_folder_mode="DEFAULT", output_folder="default", custom_output_folder=""):
        """
        r19: Standard VHS-only save.

        This intentionally matches the working JSON:
        VHS_VideoCombine receives filename_prefix, format, frame_rate, save_output=True.
        It does NOT receive output_dir or custom path arguments.
        VHS saves into the standard ComfyUI output folder.
        """
        records.append({
            "stage": "EventVideoSaveBegin",
            "status": "start_standard_vhs",
            "video_format": str(video_format),
            "fps": fps,
            "filename_prefix": str(save_prefix or "wansolo"),
            "path_policy": "standard_comfyui_vhs_output_no_custom_output_dir",
            "matches_working_json": True,
            "loop_count": 0,
            "pingpong": False,
        })
        self._event_universal_stage_math(
            records,
            "EventVideoSaveBegin",
            input_state=image,
            output_state=image,
            observed_behavior="decoded frames prepared for VHS video combine without loop/pingpong modification",
            formula_role="FRAMES VisibleOutcome -> FRAMES video input state",
            route_id="route_video_save_begin",
            next_requirement="VHS_VideoCombine receives ordered frames with frame_rate and save_output",
            control_mode="REPORT_ONLY",
            metadata={"video_format": str(video_format), "fps": float(fps), "loop_count": 0, "pingpong": False},
        )
        self._record_raw_vs_singularity_parity_probe(
            records,
            "EventRawVsSingularityParity_VideoSaveBegin",
            route_kind="vhs_video_save",
            input_state=image,
            output_state=None,
            metadata={
                "fps": float(fps),
                "video_format": str(video_format),
                "filename_prefix": str(save_prefix or "wansolo"),
                "force_vhs": bool(force_vhs),
                "output_target": str(output_target),
                "output_folder_mode": str(output_folder_mode),
                "loop_count": 0,
                "pingpong": False,
                "probe_scope": "frames_before_vhs_encode",
            },
        )

        try:
            self._frames_to_uint8_numpy(image, records=records, stage="EventVideoSave_validate_frames")
        except Exception as e:
            raise RuntimeError(f"Cannot save video because decoded frames are invalid: {e}")

        try:
            result = self._call_candidate_node_method(
                ["VHS_VideoCombine"],
                ["combine_video", "combine", "run", "execute"],
                records=records,
                stage="EventVideoCombineVHS",
                images=image,
                audio=None,
                meta_batch=None,
                vae=None,
                frame_rate=float(fps),
                loop_count=0,
                filename_prefix=str(save_prefix or "wansolo"),
                format=str(video_format or "video/h264-mp4"),
                pingpong=False,
                save_output=True,
                pix_fmt="yuv420p",
                crf=19,
                save_metadata=True,
                trim_to_audio=False,
            )
            path = self._extract_path_from_video_result(result, save_prefix, records)
            if not path:
                path = self._find_latest_standard_output_video(save_prefix, records)

            records.append({
                "stage": "EventVideoCombine",
                "status": "ok" if path else "ok_no_path_returned",
                "saved_video_path": path,
                "raw_vhs_result": str(result)[:5000],
                "formula": "decoded frames + VHS_VideoCombine(save_output=True) = saved video in standard ComfyUI output",
            })
            self._event_universal_stage_math(
                records,
                "EventVideoCombine",
                input_state=image,
                output_state=path or str(result)[:500],
                observed_behavior="VHS_VideoCombine packaged decoded frames into video file/reference",
                formula_role="FRAMES VisibleOutcome -> VIDEO file OutcomeFinal",
                route_id="route_video_combine",
                next_requirement="final report and UI preview should point to saved video path",
                control_mode="REPORT_ONLY",
                metadata={"video_format": str(video_format), "fps": float(fps), "saved_video_path": path or ""},
            )
            self._record_raw_vs_singularity_parity_probe(
                records,
                "EventRawVsSingularityParity_VideoSaveResult",
                route_kind="vhs_video_save",
                input_state=image,
                output_state=path or str(result)[:500],
                metadata={
                    "fps": float(fps),
                    "video_format": str(video_format),
                    "filename_prefix": str(save_prefix or "wansolo"),
                    "saved_video_path": path or "",
                    "vhs_result_found_path": bool(path),
                    "probe_scope": "frames_after_vhs_encode_path_resolution",
                },
            )

            if path:
                return path
            raise RuntimeError(f"VHS_VideoCombine ran but no output path was found. raw={str(result)[:500]}")
        except Exception as e:
            records.append({"stage": "EventVideoCombine", "status": "failed_standard_vhs", "error": str(e)})
            raise

    def _find_latest_standard_output_video(self, save_prefix, records):
        try:
            import folder_paths
            output_dir = Path(folder_paths.get_output_directory())
        except Exception:
            output_dir = Path.cwd() / "output"

        safe_prefix = str(save_prefix or "wansolo")
        found = []
        if output_dir.exists():
            for ext in (".mp4", ".webp", ".gif", ".mov", ".mkv"):
                found.extend(output_dir.rglob(f"{safe_prefix}*{ext}"))

        if found:
            newest = max(found, key=lambda x: x.stat().st_mtime)
            records.append({
                "stage": "EventVideoCombine_standard_output_search",
                "status": "ok",
                "path": str(newest),
                "output_dir": str(output_dir),
            })
            return str(newest)

        records.append({
            "stage": "EventVideoCombine_standard_output_search",
            "status": "not_found",
            "output_dir": str(output_dir),
            "prefix": safe_prefix,
        })
        return ""


    def _make_text_signal(self, packet, text, role, route, source_stage):
        return _read_signal(
            packet,
            TECH_TEXT,
            SPACE_TEXT,
            text,
            source_stage,
            role,
            route,
            "TextStrategyReader",
            metadata={"text_length": len(str(text or ""))},
        )

    def _fail_conflict(self, packet, stage, message, metadata=None):
        conflict = make_conflict(
            CONFLICT_FROZEN_PARTIAL_OBSERVABILITY,
            severity=SEV_MEDIUM,
            stage_position=stage,
            suspected_cause="Terminal Wan workflow generation did not produce the requested target.",
            observed_symptom=str(message),
            suggested_response="Send Execution Records and ComfyUI traceback; next patch will fix exact incompatible stage.",
            metadata=metadata or {},
        )
        packet = add_conflict(packet, conflict)
        return packet, conflict["id"]

    def _stage_delay(self, seconds, records, label):
        try:
            seconds = float(seconds or 0.0)
        except Exception:
            seconds = 0.0
        if seconds <= 0:
            return
        try:
            import time
            records.append({"stage": "SingularityStageDelay", "status": "sleep", "seconds": seconds, "label": str(label)})
            time.sleep(seconds)
        except Exception as e:
            records.append({"stage": "SingularityStageDelay", "status": "failed", "seconds": seconds, "label": str(label), "error": str(e)})

    def _bounded_strategy_cfg(self, base_cfg, raw_delta_norm, strength, records, label):
        base_cfg = float(base_cfg)
        try:
            raw = max(0.0, float(raw_delta_norm or 0.0))
        except Exception:
            raw = 0.0
        try:
            strength = float(strength if strength is not None else 1.0)
        except Exception:
            strength = 1.0

        mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
        bounded_signal = math.tanh(math.log1p(raw) / 16.0) if raw > 0.0 else 0.0
        multiplier = 1.0
        adjusted_cfg = base_cfg
        status = "observer_only"
        policy = "preserve_model_native_cfg"

        if mode == "DEEP_STEP_DELTA_CONTROL" and abs(strength - 1.0) >= 1e-9:
            # Research mode only: raw latent norms are evidence, not CFG units.
            # Any active CFG pressure must be bounded and explicitly marked.
            strength_delta = strength - 1.0
            multiplier = 1.0 + (strength_delta * 0.25 * bounded_signal)
            multiplier = max(0.25, min(1.75, multiplier))
            adjusted_cfg = base_cfg * multiplier
            status = "bounded_research"
            policy = "deep_research_bounded_cfg_projection"
        elif mode == "LATENT_DELTA_SCALE":
            policy = "latent_delta_scale_keeps_cfg_native"
        elif mode == "STRATEGY_PRESSURE_WINDOW":
            policy = "strategy_pressure_window_keeps_cfg_native"
        elif mode == "LATENT_MEMORY_BRIDGE":
            policy = "latent_memory_bridge_keeps_cfg_native"
        elif mode not in ("LATENT_DELTA_SCALE", "STRATEGY_PRESSURE_WINDOW", "LATENT_MEMORY_BRIDGE", "DEEP_STEP_DELTA_CONTROL"):
            policy = "mode_keeps_cfg_native"

        records.append({
            "stage": "EventStrategyCfgCoupling",
            "status": status,
            "label": str(label),
            "math_control_mode": mode,
            "base_cfg": base_cfg,
            "raw_delta_norm": raw,
            "strength": strength,
            "bounded_signal": bounded_signal,
            "cfg_multiplier": multiplier,
            "adjusted_cfg": adjusted_cfg,
            "policy": policy,
            "formula": "CFG is preserved in LATENT_DELTA_SCALE, STRATEGY_PRESSURE_WINDOW, and LATENT_MEMORY_BRIDGE; raw delta is ObservedBehavior evidence and delta strength belongs to latent transition control.",
        })
        return adjusted_cfg

    def _save_pause_frames_to_temp(self, frames):
        import folder_paths
        import os
        import random
        from PIL import Image
        import numpy as np

        ui_images = []
        try:
            temp_dir = folder_paths.get_temp_directory()
            total_f = int(frames.shape[0])

            tail_start_idx = int(total_f * 0.8)
            valid_indices = [
                i for i in range(tail_start_idx, total_f)
                if ((i + 1) - 1) % 4 == 0
            ]
            if len(valid_indices) < TAIL_CANDIDATE_COUNT:
                valid_indices = [
                    i for i in range(total_f)
                    if ((i + 1) - 1) % 4 == 0
                ]
            selected_indices = valid_indices[-TAIL_CANDIDATE_COUNT:] if valid_indices else [max(0, total_f - 1)]
            prefix = "singularity_pause_" + str(random.randint(100000, 999999))

            for idx, frame_idx in enumerate(selected_indices):
                frame = frames[frame_idx]
                img_array = (255.0 * frame.detach().cpu().numpy()).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                filename = f"{prefix}_{idx}.png"
                filepath = os.path.join(temp_dir, filename)
                img.save(filepath)
                ui_images.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "temp",
                    "resume_index": int(frame_idx) + 1,
                })
        except Exception as e:
            print(f"[Singularity] Failed to save pause frames: {e}")
        return ui_images

    def _save_pause_preview_video_to_temp(self, frames, fps, execution_records, segment_index):
        if frames is None:
            return None
        try:
            import folder_paths
            import random
            import numpy as np
            from PIL import Image

            arr = self._frames_to_uint8_numpy(
                frames,
                records=execution_records,
                stage=f"SingularityCascadePausePreview_{segment_index}_frames_to_numpy",
            )
            if arr.ndim != 4 or arr.shape[0] < 1:
                raise RuntimeError(f"pause preview expected [N,H,W,C], got {getattr(arr, 'shape', None)}")

            original_shape = list(arr.shape)
            max_edge = 420
            h, w = int(arr.shape[1]), int(arr.shape[2])
            longest = max(h, w)
            if longest > max_edge:
                scale = float(max_edge) / float(longest)
                target_w = max(2, int(round(w * scale)))
                target_h = max(2, int(round(h * scale)))
                target_w = max(2, target_w - (target_w % 2))
                target_h = max(2, target_h - (target_h % 2))
                resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
                arr = np.stack([
                    np.asarray(Image.fromarray(frame).resize((target_w, target_h), resample=resample))
                    for frame in arr
                ], axis=0).astype("uint8")

            temp_dir = Path(folder_paths.get_temp_directory())
            temp_dir.mkdir(parents=True, exist_ok=True)
            prefix = "singularity_pause_preview_" + str(segment_index) + "_" + str(random.randint(100000, 999999))

            try:
                import imageio.v3 as iio
                filename = f"{prefix}.mp4"
                path = temp_dir / filename
                iio.imwrite(
                    str(path),
                    arr,
                    fps=float(fps),
                    codec="libx264",
                    macro_block_size=1,
                    output_params=["-pix_fmt", "yuv420p", "-crf", "24"],
                )
                if path.exists() and path.stat().st_size > 0:
                    payload = {
                        "filename": filename,
                        "subfolder": "",
                        "type": "temp",
                        "format": "video/h264-mp4",
                        "frame_count": int(arr.shape[0]),
                        "preview_kind": "stitched_so_far",
                    }
                    execution_records.append({
                        "stage": f"SingularityCascadePausePreview_{segment_index}",
                        "status": "ok",
                        "path": str(path),
                        "size": int(path.stat().st_size),
                        "original_shape": original_shape,
                        "preview_shape": list(arr.shape),
                        "storage": "temp",
                        "formula": "stitched cascade frames so far are exposed as preview-only media; final Outcome is still produced by the final concat/video combine",
                    })
                    return payload
                execution_records.append({
                    "stage": f"SingularityCascadePausePreview_{segment_index}",
                    "status": "empty_mp4",
                    "path": str(path),
                })
            except Exception as e:
                execution_records.append({
                    "stage": f"SingularityCascadePausePreview_{segment_index}_mp4",
                    "status": "failed_nonfatal",
                    "error": str(e),
                })

            try:
                filename = f"{prefix}.webp"
                path = temp_dir / filename
                pil_frames = [Image.fromarray(frame) for frame in arr]
                duration = int(1000 / max(1, float(fps)))
                pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:], duration=duration, loop=0, lossless=False, quality=82)
                if path.exists() and path.stat().st_size > 0:
                    payload = {
                        "filename": filename,
                        "subfolder": "",
                        "type": "temp",
                        "format": "image/webp",
                        "frame_count": int(arr.shape[0]),
                        "preview_kind": "stitched_so_far",
                    }
                    execution_records.append({
                        "stage": f"SingularityCascadePausePreview_{segment_index}_webp",
                        "status": "ok",
                        "path": str(path),
                        "size": int(path.stat().st_size),
                        "original_shape": original_shape,
                        "preview_shape": list(arr.shape),
                        "storage": "temp",
                    })
                    return payload
            except Exception as e:
                execution_records.append({
                    "stage": f"SingularityCascadePausePreview_{segment_index}_webp",
                    "status": "failed_nonfatal",
                    "error": str(e),
                })
        except Exception as e:
            execution_records.append({
                "stage": f"SingularityCascadePausePreview_{segment_index}",
                "status": "failed_nonfatal",
                "error": str(e),
            })
        return None

    def _wait_for_cascade_continue(self, node_id, generated_frames, segment_index, execution_records, stitched_preview_frames=None, fps=16):
        node_key = _singularity_pause_key(node_id)
        pause_frames = self._save_pause_frames_to_temp(generated_frames)
        preview_video = self._save_pause_preview_video_to_temp(
            stitched_preview_frames if stitched_preview_frames is not None else generated_frames,
            fps,
            execution_records,
            segment_index,
        )
        resume_indices = [
            _singularity_parse_resume_index(item.get("resume_index"), default=-1)
            for item in pause_frames
            if isinstance(item, dict)
        ]
        resume_indices = [idx for idx in resume_indices if idx > 0]
        default_resume_index = resume_indices[-1] if resume_indices else -1

        execution_records.append({
            "stage": f"SingularityCascadePause_{segment_index}",
            "status": "waiting_for_continue",
            "node_id": node_key,
            "pause_frame_count": len(pause_frames),
            "resume_candidates": resume_indices,
            "default_resume_frame_index": default_resume_index,
            "stitched_preview": _event_json_safe(preview_video),
            "formula": "cascade boundary waits inside the current prompt until the user selects a MirrorCut frame",
        })

        if PromptServer is None:
            execution_records.append({
                "stage": f"SingularityCascadePause_{segment_index}_Fallback",
                "status": "auto_continue_no_prompt_server",
                "resume_frame_index": default_resume_index,
                "prompt_update_present": False,
            })
            return default_resume_index, pause_frames, {}

        existing_state = _SINGULARITY_PAUSE_STATES.get(node_key, {})
        if existing_state.get("status") != "continue":
            _SINGULARITY_PAUSE_STATES[node_key] = {
                "status": "paused",
                "segment_index": int(segment_index),
                "resume_frame_index": default_resume_index,
                "resume_candidates": resume_indices,
                "pause_frames": _event_json_safe(pause_frames),
                "preview_video": _event_json_safe(preview_video),
                "updated_at": time.time(),
            }

        PromptServer.instance.send_sync("singularity_cascade_paused", {
            "node_id": node_key,
            "segment_index": int(segment_index),
            "pause_frames": pause_frames,
            "resume_candidates": resume_indices,
            "default_resume_frame_index": default_resume_index,
            "preview_video": preview_video,
            "stitched_preview": preview_video,
        })

        while True:
            try:
                throw_exception_if_processing_interrupted()
            except InterruptProcessingException:
                _SINGULARITY_PAUSE_STATES.pop(node_key, None)
                execution_records.append({
                    "stage": f"SingularityCascadePause_{segment_index}_Interrupt",
                    "status": "interrupted",
                    "node_id": node_key,
                    "formula": "ComfyUI interrupt cancels the same-run pause wait instead of leaving the workflow in a blocked pause state.",
                })
                raise
            state = _SINGULARITY_PAUSE_STATES.get(node_key, {})
            status = state.get("status")
            if status == "continue":
                resume_frame_index = _singularity_parse_resume_index(
                    state.get("resume_frame_index", default_resume_index),
                    default=default_resume_index,
                )
                if resume_frame_index < 1:
                    resume_frame_index = default_resume_index
                prompt_update = state.get("prompt_update", {})
                if not isinstance(prompt_update, dict):
                    prompt_update = {}
                _SINGULARITY_PAUSE_STATES.pop(node_key, None)
                execution_records.append({
                    "stage": f"SingularityCascadePause_{segment_index}_Continue",
                    "status": "continue",
                    "node_id": node_key,
                    "resume_frame_index": resume_frame_index,
                    "prompt_update_present": bool(prompt_update),
                    "prompt_payload_version": prompt_update.get("payload_version", "") if isinstance(prompt_update, dict) else "",
                    "prompt_source": prompt_update.get("source", "") if isinstance(prompt_update, dict) else "",
                    "positive_prompt_length": len(str(prompt_update.get("positive_prompt", ""))) if isinstance(prompt_update, dict) else 0,
                    "negative_prompt_length": len(str(prompt_update.get("negative_prompt", ""))) if isinstance(prompt_update, dict) else 0,
                })
                return resume_frame_index, pause_frames, prompt_update
            if status == "cancelled":
                _SINGULARITY_PAUSE_STATES.pop(node_key, None)
                execution_records.append({
                    "stage": f"SingularityCascadePause_{segment_index}_Cancel",
                    "status": "cancelled",
                    "node_id": node_key,
                })
                raise InterruptProcessingException()
            time.sleep(0.1)

    def _trim_cascade_resume_state(self, frames, latent, resume_frame_index, execution_records, stage_prefix):
        resume_frame_index = max(1, int(resume_frame_index))
        target_t = max(1, (resume_frame_index - 1) // 4 + 1)
        frame_shape_before = list(frames.shape) if hasattr(frames, "shape") else None
        latent_shape_before = None
        try:
            samples_before = latent.get("samples") if isinstance(latent, dict) else None
            latent_shape_before = list(samples_before.shape) if hasattr(samples_before, "shape") else None
        except Exception:
            latent_shape_before = None
        self._record_raw_vs_singularity_parity_probe(
            execution_records,
            f"EventRawVsSingularityParity_{stage_prefix}TrimBefore",
            route_kind="pause_resume_trim",
            input_state={"frames": frames, "latent": latent},
            output_state=None,
            metadata={
                "resume_frame_index": int(resume_frame_index),
                "target_latent_t": int(target_t),
                "frame_shape_before": frame_shape_before,
                "latent_shape_before": latent_shape_before,
                "probe_scope": "pause_selection_before_frame_and_latent_trim",
            },
        )
        try:
            samples = latent.get("samples") if isinstance(latent, dict) else None
            if samples is not None and samples.shape[2] > target_t:
                latent["samples"] = samples[:, :, :target_t, :, :,]
        except Exception as e:
            execution_records.append({
                "stage": f"{stage_prefix}LatentTrim",
                "status": "failed_nonfatal",
                "error": str(e),
            })

        try:
            if frames is not None and frames.shape[0] > resume_frame_index:
                frames = frames[:resume_frame_index, :, :, :]
        except Exception as e:
            execution_records.append({
                "stage": f"{stage_prefix}FrameTrim",
                "status": "failed_nonfatal",
                "error": str(e),
            })
        frame_shape_after = list(frames.shape) if hasattr(frames, "shape") else None
        latent_shape_after = None
        try:
            samples_after = latent.get("samples") if isinstance(latent, dict) else None
            latent_shape_after = list(samples_after.shape) if hasattr(samples_after, "shape") else None
        except Exception:
            latent_shape_after = None
        self._record_raw_vs_singularity_parity_probe(
            execution_records,
            f"EventRawVsSingularityParity_{stage_prefix}TrimAfter",
            route_kind="pause_resume_trim",
            input_state=None,
            output_state={"frames": frames, "latent": latent},
            metadata={
                "resume_frame_index": int(resume_frame_index),
                "target_latent_t": int(target_t),
                "frame_shape_before": frame_shape_before,
                "latent_shape_before": latent_shape_before,
                "frame_shape_after": frame_shape_after,
                "latent_shape_after": latent_shape_after,
                "probe_scope": "pause_selection_after_frame_and_latent_trim",
            },
        )
        return frames, latent, target_t

    def _last_frame_image(self, frames, width=64, height=64):
        return self._representative_preview_frame(frames, width, height, mode="last")

    def _compute_continuation_fitness(self, frames, source_image=None, records=None):
        """
        Metrics that matter for using a frame as starting point for the next cascade segment.
        Focus: sharpness, color fidelity to source (if available), rough texture preservation.
        Returns dict with per-frame scores (higher = better for continuation).
        """
        if frames is None or not hasattr(frames, "dim"):
            return None

        try:
            import torch
            import torch.nn.functional as F

            t = self._tensor_from_latent_like(frames)
            if t is None or t.dim() != 4:
                return None

            n = t.shape[0]
            fitness_scores = []

            # 1. Sharpness via Laplacian variance (very standard and effective)
            for i in range(n):
                frame = t[i : i+1].permute(0, 3, 1, 2)  # to [1, C, H, W]
                # Simple Laplacian approximation
                laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], 
                                                device=frame.device, dtype=frame.dtype).repeat(3,1,1,1)
                lap = F.conv2d(frame, laplacian_kernel, padding=1, groups=3)
                sharpness = lap.var().item()
                fitness_scores.append(sharpness)

            # Normalize sharpness
            if fitness_scores:
                max_sharp = max(fitness_scores) or 1.0
                sharpness_scores = [s / max_sharp for s in fitness_scores]
            else:
                sharpness_scores = [0.5] * n

            # 2. Color fidelity to source (if source available)
            color_fidelity = [0.5] * n
            if source_image is not None:
                try:
                    src = self._tensor_from_latent_like(source_image)
                    if src is not None:
                        src_mean = src.mean(dim=[0,1,2])  # [C]
                        for i in range(n):
                            frame_mean = t[i].mean(dim=[0,1])  # [C]
                            # Cosine similarity of color means
                            cos = F.cosine_similarity(frame_mean.unsqueeze(0), src_mean.unsqueeze(0)).item()
                            color_fidelity[i] = max(0.0, min(1.0, (cos + 1.0) / 2.0))
                except Exception:
                    pass

            # Combined continuation fitness (can be tuned)
            final_fitness = []
            for i in range(n):
                score = 0.65 * sharpness_scores[i] + 0.35 * color_fidelity[i]
                final_fitness.append(float(score))

            if records is not None:
                records.append({
                    "stage": "TailFrames_ContinuationFitness",
                    "status": "ok",
                    "sharpness_scores": sharpness_scores,
                    "color_fidelity_to_source": color_fidelity if source_image is not None else None,
                    "combined_fitness": final_fitness
                })

            return {
                "fitness": final_fitness,
                "sharpness": sharpness_scores,
                "color_fidelity": color_fidelity
            }

        except Exception as e:
            if records is not None:
                records.append({"stage": "TailFrames_ContinuationFitness", "status": "failed", "error": str(e)})
            return None

    def _event_background_anchor_preservation_card(self, frames, records=None):
        """
        Report-only visible-frame anchor card.

        The center may move because the event requires motion. The background
        should stay closer to SourceAnchor / OutcomePrevious. If background ROI
        motion rises with center motion, the whole scene is drifting instead of
        only the intended event carrier changing.
        """
        def safe_float(value, default=0.0):
            try:
                out = float(value)
            except Exception:
                return default
            return out if math.isfinite(out) else default

        def clamp01(value):
            return max(0.0, min(1.0, safe_float(value, 0.0)))

        card = {
            "stage": "EventBackgroundAnchorPreservationCard",
            "status": "not_available",
            "card_version": "background_anchor_preservation_v1",
            "formula": "Background is SourceAnchor / OutcomePrevious evidence; central motion should not turn the whole visible field into ObservedBehavior.",
            "control_mode": "REPORT_ONLY",
            "active_control_allowed": False,
            "formula_role": "SourceAnchor preservation against global scene drift",
        }

        try:
            import torch

            t = self._tensor_from_latent_like(frames)
            if t is None or not hasattr(t, "dim") or t.dim() != 4:
                card.update({"status": "bad_shape", "shape": list(getattr(frames, "shape", [])) if hasattr(frames, "shape") else None})
                return card

            if t.shape[-1] not in (1, 3, 4) and t.shape[1] in (1, 3, 4):
                t = t.permute(0, 2, 3, 1)
            if t.shape[0] < 2:
                card.update({"status": "not_enough_frames", "frame_count": int(t.shape[0])})
                return card

            frame_count = int(t.shape[0])
            sample_stride = max(1, frame_count // 180)
            sampled = t[::sample_stride].detach().float().cpu()
            if sampled.shape[0] < 2:
                sampled = t.detach().float().cpu()

            # Keep measurement cheap and stable. Values are scaled for report
            # readability so historical mp4-side analysis and in-node cards
            # live in the same rough numeric range.
            if sampled.shape[-1] >= 3:
                gray = (
                    0.299 * sampled[..., 0]
                    + 0.587 * sampled[..., 1]
                    + 0.114 * sampled[..., 2]
                )
            else:
                gray = sampled[..., 0]
            diffs = (gray[1:] - gray[:-1]).abs()
            height = int(diffs.shape[1])
            width = int(diffs.shape[2])
            scale = 255.0

            def roi_mean(box):
                x0, y0, x1, y1 = box
                ix0 = max(0, min(width - 1, int(round(width * x0))))
                iy0 = max(0, min(height - 1, int(round(height * y0))))
                ix1 = max(ix0 + 1, min(width, int(round(width * x1))))
                iy1 = max(iy0 + 1, min(height, int(round(height * y1))))
                roi = diffs[:, iy0:iy1, ix0:ix1]
                per_step = roi.reshape(roi.shape[0], -1).mean(dim=1) * scale
                return {
                    "mean": float(per_step.mean().item()),
                    "max": float(per_step.max().item()),
                }

            roi_boxes = {
                "top_left_background": (0.00, 0.00, 0.28, 0.22),
                "top_right_background": (0.72, 0.00, 1.00, 0.22),
                "top_band_background": (0.20, 0.00, 0.80, 0.16),
                "left_side_floor": (0.00, 0.40, 0.18, 1.00),
                "right_side_floor": (0.82, 0.40, 1.00, 1.00),
                "lower_side_floor": (0.00, 0.78, 1.00, 1.00),
                "center_event_proxy": (0.25, 0.22, 0.75, 0.82),
                "full_frame": (0.00, 0.00, 1.00, 1.00),
            }
            roi_stats = {name: roi_mean(box) for name, box in roi_boxes.items()}
            background_names = [
                "top_left_background",
                "top_right_background",
                "top_band_background",
                "left_side_floor",
                "right_side_floor",
                "lower_side_floor",
            ]
            background_values = [roi_stats[name]["mean"] for name in background_names]
            background_max_values = [roi_stats[name]["max"] for name in background_names]
            background_mean = sum(background_values) / max(1, len(background_values))
            background_max = max(background_max_values) if background_max_values else 0.0
            center_mean = roi_stats["center_event_proxy"]["mean"]
            full_mean = roi_stats["full_frame"]["mean"]
            center_background_ratio = center_mean / max(background_mean, 1e-6)

            background_drift_score = clamp01((background_mean - 3.50) / 1.80)
            weak_separation_score = clamp01((1.80 - center_background_ratio) / 0.80)
            global_scene_drift_score = max(background_drift_score, weak_separation_score)
            if global_scene_drift_score >= 0.65:
                status = "global_scene_drift_high"
            elif global_scene_drift_score >= 0.35:
                status = "background_anchor_watch"
            else:
                status = "background_anchor_stable"

            card.update({
                "status": status,
                "frame_count": frame_count,
                "sample_stride": int(sample_stride),
                "background_temporal_mean": round(background_mean, 6),
                "background_temporal_max": round(background_max, 6),
                "center_temporal_mean": round(center_mean, 6),
                "full_frame_temporal_mean": round(full_mean, 6),
                "center_background_ratio": round(center_background_ratio, 6),
                "background_drift_score": round(background_drift_score, 6),
                "weak_center_background_separation_score": round(weak_separation_score, 6),
                "global_scene_drift_score": round(global_scene_drift_score, 6),
                "roi_temporal_means": {name: round(stats["mean"], 6) for name, stats in roi_stats.items()},
                "roi_temporal_maxes": {name: round(stats["max"], 6) for name, stats in roi_stats.items()},
                "score_policy": "background_mean > 3.5 or center/background ratio < 1.8 means visible motion may be global scene drift, not only central event motion",
                "next_action": (
                    "Pull low branch toward neutral or keep report-only if background drift is high."
                    if status == "global_scene_drift_high"
                    else "Background anchor is measurable; compare this card across fixed-seed runs."
                ),
            })
            return card
        except Exception as e:
            card.update({"status": "failed_nonfatal", "error": str(e)})
            if records is not None:
                records.append({"stage": "EventBackgroundAnchorPreservationCardError", "status": "failed_nonfatal", "error": str(e)})
            return card

    def _select_best_tail_frames(self, frames, count=TAIL_CANDIDATE_COUNT, records=None):
        """
        Selection of N frames from the tail, scored by existing motion math.
        This is part of the "1 UI + formula layer" on top of internal node flows.
        Returns dict with 'frames' tensor and 'scores' list (higher = more motion / interesting).
        """
        if frames is None:
            if records is not None:
                records.append({"stage": "TailFramesSelect", "status": "no_frames"})
            return None
        try:
            if not hasattr(frames, "dim") or frames.dim() != 4:
                if records is not None:
                    records.append({"stage": "TailFramesSelect", "status": "bad_shape"})
                return None

            total = frames.shape[0]
            n = min(int(count), total)
            tail = frames[-n:]

            # === MAXIMUM USE — SIGNALS THAT ACTUALLY INFLUENCE THE SYSTEM'S CHOICE ===
            # We prioritize information the internal formula/process already cares about:
            # - raw_delta_norm from the high branch (core of the raw trembling)
            # - Per-segment frame_motion_math (especially the last segments)
            # - Stability/reversal/jerk from the actual generation process

            # 1. Motion on the tail frames themselves (local interestingness)
            tail_motion = self._frame_motion_math(tail, records or [], "TailFrames_TailOnly")

            local_scores = []
            if tail_motion and tail_motion.get("status") == "ok":
                norms = tail_motion.get("norms", [])
                reversal = tail_motion.get("reversal_ratio", 0.0) or 0.0
                stability = tail_motion.get("stability_score", 0.5) or 0.5

                for i in range(n):
                    if i == 0:
                        local_scores.append(0.0)
                    else:
                        idx = i - 1
                        base = float(norms[idx]) if idx < len(norms) else 0.0
                        # Frames after high-reversal moments are "more alive" according to raw logic
                        reversal_boost = 1.0 + min(reversal * 0.6, 0.8)
                        local_scores.append(base * reversal_boost)
            else:
                for i in range(n):
                    local_scores.append(0.0 if i == 0 else float((tail[i] - tail[i-1]).abs().mean().item()))

            # 2. Pull the most system-relevant signals from recent execution_records
            last_raw_deltas = []      # raw_delta_norm from high samplers
            last_bounded_signals = [] # normalized seam pressure from Strategy coupling
            last_segment_motion = []  # full motion records from last segments

            if records:
                for rec in reversed(records):
                    if not isinstance(rec, dict):
                        continue
                    if "raw_delta_norm" in rec and rec["raw_delta_norm"] is not None:
                        last_raw_deltas.append(float(rec["raw_delta_norm"]))
                    if "bounded_signal" in rec and rec["bounded_signal"] is not None:
                        last_bounded_signals.append(float(rec["bounded_signal"]))
                    if rec.get("stage", "").startswith("EventMath_cascade_") and rec.get("stage", "").endswith("_frame_motion"):
                        last_segment_motion.append(rec)

            # We care most about the last 1-2 segments (they produced the tail)
            relevant_deltas = last_raw_deltas[:2]
            relevant_bounded = last_bounded_signals[:2]
            relevant_motion = last_segment_motion[:2]

            avg_relevant_delta = sum(relevant_deltas) / len(relevant_deltas) if relevant_deltas else 0.0

            # Aggregate "system interest" from last segments (high delta + certain motion profiles = more valuable frames)
            system_interest = 1.0
            if relevant_deltas:
                system_interest *= (1.0 + min(avg_relevant_delta / 6000.0, 1.5))

            if relevant_motion:
                for m in relevant_motion:
                    stab = m.get("frame_motion_stability_score", m.get("stability_score", 0.5))
                    rev = m.get("frame_delta_reversal_ratio", m.get("reversal_ratio", 0.0)) or 0.0
                    # The system (raw philosophy) tends to value moments with decent delta + some reversal
                    system_interest *= (1.0 + rev * 0.5) * (1.0 + (1.0 - min(stab, 1.0)) * 0.3)

            # 3. Final system proposal scores
            # Recency is intentionally only a tiny tie-breaker. Earlier builds
            # let the last slot win too often even when the content was weaker.
            final_scores = []
            for i, local in enumerate(local_scores):
                recency_tiebreak = (i / max(n-1, 1)) * 0.025
                score = (local * system_interest) + recency_tiebreak
                final_scores.append(float(score))

            # Normalize
            mx = max(final_scores) if final_scores else 1.0
            if mx > 0:
                final_scores = [s / mx for s in final_scores]

            # === KB-GROUNDED RAW FORMULA EXTENSION (bidirectional Mirror reading for tail) ===
            # From _knowledge_base: left side (segment past Outcome + ObservedBehavior via raw_deltas + motion)
            # forms Strategy context. Right side: each tail candidate as potential admissible causal continuation (B+ + O+).
            # MirrorBreak = semantic distance (how much choosing this tail would break the event equality for chaining).
            # We compute raw, no clamps beyond [0,1] normalization for scoring, explicit terms only.
            # [RAW5] per-candidate for full trace.
            mirror_break_scores = []
            admissible_continuation_scores = []
            past_strategy_proxy = 0.0
            if relevant_bounded:
                past_delta_signal = sum(relevant_bounded) / len(relevant_bounded)
            elif relevant_deltas:
                avg_delta = sum(relevant_deltas) / len(relevant_deltas)
                past_delta_signal = avg_delta / (avg_delta + 6000.0)
            else:
                past_delta_signal = 0.5

            motion_parts = []
            if relevant_motion:
                for m in relevant_motion:
                    rev = m.get("frame_delta_reversal_ratio", m.get("reversal_ratio", 0.0)) or 0.0
                    stab = m.get("frame_motion_stability_score", m.get("stability_score", 0.5)) or 0.5
                    spike = m.get("frame_delta_spike_ratio", 1.0) or 1.0
                    jerk = m.get("frame_delta_jerk_ratio", 0.0) or 0.0
                    motion_parts.append(
                        0.40 * min(max(float(rev), 0.0), 1.0)
                        + 0.30 * (1.0 - min(max(float(stab), 0.0), 1.0))
                        + 0.20 * min(max((float(spike) - 1.0) / 1.5, 0.0), 1.0)
                        + 0.10 * min(max(float(jerk), 0.0), 1.0)
                    )
            past_motion_signal = sum(motion_parts) / len(motion_parts) if motion_parts else 0.5
            past_strategy_proxy = min(max(0.65 * past_delta_signal + 0.35 * past_motion_signal, 0.0), 1.0)

            local_score_max = max(local_scores) if local_scores else 0.0

            for i in range(n):
                # Observed for this candidate (right side proxy): normalized tail motion
                # plus a tiny recency tie-breaker, not a dominant future trace.
                cand_local = local_scores[i] if i < len(local_scores) else 0.0
                cand_local_norm = (float(cand_local) / float(local_score_max)) if local_score_max > 0 else 0.0
                cand_recency_norm = i / max(n - 1, 1)
                observed_for_cand = min(max(0.97 * cand_local_norm + 0.03 * cand_recency_norm, 0.0), 1.0)

                # MirrorBreak compares normalized Strategy-side signals, not raw latent norm units vs pixel motion units.
                # Lower = better admissible continuation (keeps the event coherent per formula right side)
                mb = min(1.0, abs(float(past_strategy_proxy) - float(observed_for_cand)))

                mirror_break_scores.append(float(mb))
                admissible = 1.0 - mb
                admissible_continuation_scores.append(float(admissible))

                if records is not None:
                    records.append({
                        "stage": "FORMULA_TAIL_MIRROR_BREAK",
                        "candidate_index": i,
                        "mirror_break": float(mb),
                        "admissible_continuation": float(admissible),
                        "past_strategy_proxy": float(past_strategy_proxy),
                        "observed_for_candidate": float(observed_for_cand),
                        "raw_components": {
                            "relevant_raw_deltas": relevant_deltas,
                            "relevant_bounded_signals": relevant_bounded,
                            "past_delta_signal": float(past_delta_signal),
                            "past_motion_signal": float(past_motion_signal),
                            "local_motion_score": cand_local,
                            "local_motion_normalized": float(cand_local_norm),
                            "recency_normalized": float(cand_recency_norm)
                        },
                        "note": "[RAW7] Bidirectional normalized Mirror: left=normalized segment Strategy (bounded seam pressure + motion), right=normalized candidate continuation. Recency is only a tie-breaker, not the selection driver."
                    })

            # Blend existing system scores with formula admissible scores.
            # The content/continuation evidence must dominate slot order.
            blended_formula_scores = []
            for i in range(n):
                sys_s = final_scores[i] if i < len(final_scores) else 0.0
                adm = admissible_continuation_scores[i] if i < len(admissible_continuation_scores) else 0.5
                blended = 0.55 * sys_s + 0.45 * adm
                blended_formula_scores.append(float(blended))

            # Re-normalize blended for recommendation use
            mb_mx = max(blended_formula_scores) if blended_formula_scores else 1.0
            if mb_mx > 0:
                blended_formula_scores = [s / mb_mx for s in blended_formula_scores]

            if records is not None:
                records.append({
                    "stage": "TailFramesSelect",
                    "status": "ok",
                    "requested": count,
                    "returned": n,
                    "total_frames": total,
                    "mode": "system_native_signals_maximum + KB_formula_bidirectional",
                    "scores": blended_formula_scores,  # now carries stronger formula right-side admissible continuation
                    "mirror_break_scores": mirror_break_scores,
                    "admissible_continuation_scores": admissible_continuation_scores,
                    "system_signals": {
                        "last_raw_deltas_used": relevant_deltas,
                        "last_bounded_signals_used": relevant_bounded,
                        "last_segments_motion_used": len(relevant_motion),
                        "system_interest_multiplier": system_interest,
                        "past_strategy_proxy": past_strategy_proxy
                    },
                    "formula_note": "Mirror reading from KB: left (t-1 Outcome + Observed via raw deltas/motion at seam) forms Strategy context; right scores candidates as admissible continuation for next segment without breaking event equality."
                })

            return {
                "frames": tail,
                "scores": blended_formula_scores,
                "mirror_break_scores": mirror_break_scores,
                "admissible_continuation_scores": admissible_continuation_scores,
                "meta": {
                    "system_interest": system_interest,
                    "relevant_raw_deltas": relevant_deltas,
                    "past_strategy_proxy": past_strategy_proxy
                }
            }
        except Exception as e:
            if records is not None:
                records.append({"stage": "TailFramesSelect", "status": "failed", "error": str(e)})
            return None

    def _drop_first_frame_batch(self, frames, records, segment_index):
        """
        r22 temporal continuity guard:
        segment N starts from the last frame of segment N-1.
        Keeping that first generated frame duplicates the previous terminal frame and can look like
        a motion reset or vector reversal at the cascade boundary.
        """
        try:
            if hasattr(frames, "dim") and frames.dim() == 4 and frames.shape[0] > 1:
                out = frames[1:]
                self._record_raw_vs_singularity_parity_probe(
                    records,
                    "EventRawVsSingularityParity_CascadeDropFirstFrame",
                    route_kind="cascade_boundary_frame_trim",
                    input_state=frames,
                    output_state=out,
                    metadata={
                        "segment_index": int(segment_index),
                        "before_shape": list(frames.shape),
                        "after_shape": list(out.shape),
                        "dropped_first_frame": True,
                        "reason": "remove duplicated source/continuation frame at cascade boundary",
                    },
                )
                records.append({
                    "stage": "SingularityCascadeDropFirstFrame",
                    "status": "ok",
                    "segment_index": int(segment_index),
                    "before_shape": list(frames.shape),
                    "after_shape": list(out.shape),
                    "reason": "remove duplicated source/continuation frame at cascade boundary",
                })
                return out
        except Exception as e:
            records.append({
                "stage": "SingularityCascadeDropFirstFrame",
                "status": "failed",
                "segment_index": int(segment_index),
                "error": str(e),
            })
        self._record_raw_vs_singularity_parity_probe(
            records,
            "EventRawVsSingularityParity_CascadeDropFirstFrame",
            route_kind="cascade_boundary_frame_trim",
            input_state=frames,
            output_state=frames,
            metadata={
                "segment_index": int(segment_index),
                "dropped_first_frame": False,
                "reason": "frame batch was not a trim-eligible 4D tensor with more than one frame",
            },
        )
        return frames

    def _record_cascade_strategy_continuity_probe(
        self,
        records,
        *,
        segment_index,
        previous_frames,
        next_source_image,
        next_frames,
        resume_frame_index=None,
        latent_temporal_target_t=None,
        strategy_carrier_context=None,
    ):
        """
        Report-only r109 probe for the cascade Strategy return boundary.
        It checks whether the selected/terminal OutcomePrevious became the next
        source StrategyCarrier, and how hard the next segment reinterprets it.
        """
        def _slice_frame(obj, mode):
            try:
                t = self._tensor_from_latent_like(obj)
                if t is None:
                    return None
                if hasattr(t, "dim"):
                    if t.dim() == 4 and int(t.shape[0]) > 0:
                        return t[-1:] if mode == "last" else t[:1]
                    if t.dim() == 3:
                        return t.unsqueeze(0)
                return t
            except Exception:
                return None

        def _compare_frames(a, b):
            try:
                import torch
                if a is None or b is None:
                    return {"status": "unavailable"}
                af = torch.nan_to_num(a.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                bf = torch.nan_to_num(b.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                if list(af.shape) != list(bf.shape):
                    return {
                        "status": "shape_mismatch",
                        "a_shape": [int(x) for x in list(af.shape)],
                        "b_shape": [int(x) for x in list(bf.shape)],
                    }
                d = bf - af
                abs_mean = float(d.abs().mean().item())
                norm = float(torch.linalg.vector_norm(d).item())
                max_abs = float(d.abs().max().item())
                if abs_mean <= 1.0e-7 and max_abs <= 1.0e-6:
                    band = "exact"
                elif abs_mean <= 0.003:
                    band = "near_exact"
                elif abs_mean <= 0.03:
                    band = "small_motion"
                elif abs_mean <= 0.10:
                    band = "visible_reinterpretation"
                else:
                    band = "strong_reinterpretation"
                return {
                    "status": "ok",
                    "band": band,
                    "abs_mean": abs_mean,
                    "norm": norm,
                    "max_abs": max_abs,
                    "shape": [int(x) for x in list(af.shape)],
                }
            except Exception as e:
                return {"status": "failed", "error": str(e)[:240]}

        try:
            previous_last = _slice_frame(previous_frames, "last")
            source_frame = _slice_frame(next_source_image, "first")
            next_first = _slice_frame(next_frames, "first")
            previous_last_vs_source = _compare_frames(previous_last, source_frame)
            source_vs_next_first = _compare_frames(source_frame, next_first)
            previous_last_vs_next_first = _compare_frames(previous_last, next_first)

            def _band(metric):
                return metric.get("band", metric.get("status", "unknown")) if isinstance(metric, dict) else "unknown"

            tail_to_source_band = _band(previous_last_vs_source)
            source_to_first_band = _band(source_vs_next_first)
            if tail_to_source_band in ("exact", "near_exact"):
                tail_to_source_status = "strategy_carrier_preserved"
            elif tail_to_source_band in ("small_motion", "visible_reinterpretation", "strong_reinterpretation"):
                tail_to_source_status = "strategy_carrier_drift_before_sampler"
            else:
                tail_to_source_status = tail_to_source_band

            if source_to_first_band in ("exact", "near_exact", "small_motion"):
                source_to_first_status = "next_segment_continues_source_anchor"
            elif source_to_first_band in ("visible_reinterpretation", "strong_reinterpretation"):
                source_to_first_status = "next_segment_reinterprets_source_anchor"
            else:
                source_to_first_status = source_to_first_band

            ctx = strategy_carrier_context if isinstance(strategy_carrier_context, dict) else {}
            context_digest = {
                "prompt_source": ctx.get("prompt_source", ""),
                "prompt_transcode_mode": ctx.get("prompt_transcode_mode", ""),
                "prompt_continuity_reused": bool(ctx.get("prompt_continuity_reused", False)),
                "prompt_continuity_policy": ctx.get("prompt_continuity_policy", ""),
                "current_active_positive_signature": ctx.get("current_active_positive_signature", ""),
                "current_active_negative_signature": ctx.get("current_active_negative_signature", ""),
                "current_active_positive_normalized_signature": ctx.get("current_active_positive_normalized_signature", ""),
                "last_runtime_prompt_update_applies_to_segment": ctx.get("last_runtime_prompt_update_applies_to_segment", None),
            }
            record = {
                "stage": "EventCascadeStrategyContinuityProbe",
                "status": "recorded",
                "probe_version": "cascade_strategy_continuity_probe_v1",
                "segment_index": int(segment_index),
                "previous_segment_index": int(segment_index) - 1,
                "resume_frame_index": int(resume_frame_index) if resume_frame_index is not None else None,
                "latent_temporal_target_t": int(latent_temporal_target_t) if latent_temporal_target_t is not None else None,
                "previous_last_frame_probe": self._event_tensor_probe(previous_last, label=f"cascade_{segment_index}.previous_last_frame"),
                "next_source_image_probe": self._event_tensor_probe(source_frame, label=f"cascade_{segment_index}.next_source_image"),
                "next_first_frame_probe": self._event_tensor_probe(next_first, label=f"cascade_{segment_index}.next_first_frame"),
                "next_batch_probe": self._event_tensor_probe(next_frames, label=f"cascade_{segment_index}.next_frame_batch"),
                "continuity_metrics": {
                    "previous_last_vs_next_source": previous_last_vs_source,
                    "next_source_vs_next_first": source_vs_next_first,
                    "previous_last_vs_next_first": previous_last_vs_next_first,
                },
                "tail_to_source_status": tail_to_source_status,
                "source_to_first_status": source_to_first_status,
                "strategy_carrier_context": context_digest,
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
                "formula": (
                    "Cascade Strategy continuity reads the selected tail OutcomePrevious as the next source "
                    "StrategyCarrier; the first next-frame Outcome shows whether the model continued the same "
                    "event or re-solved the prompt as a local segment."
                ),
            }
            records.append(record)
            return record
        except Exception as e:
            record = {
                "stage": "EventCascadeStrategyContinuityProbe",
                "status": "failed",
                "segment_index": int(segment_index) if segment_index is not None else None,
                "error": str(e)[:240],
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
            }
            records.append(record)
            return record

    def _concat_frame_batches(self, batches, records):
        valid = [b for b in batches if b is not None]
        if not valid:
            return None
        if len(valid) == 1:
            self._record_raw_vs_singularity_parity_probe(
                records,
                "EventRawVsSingularityParity_CascadeJoin",
                route_kind="cascade_frame_join",
                input_state=valid,
                output_state=valid[0],
                metadata={
                    "segments": 1,
                    "concat_performed": False,
                    "input_shapes": [list(b.shape) if hasattr(b, "shape") else str(type(b)) for b in valid],
                    "output_shape": list(valid[0].shape) if hasattr(valid[0], "shape") else str(type(valid[0])),
                },
            )
            return valid[0]
        try:
            import torch
            out = torch.cat(valid, dim=0)
            self._record_raw_vs_singularity_parity_probe(
                records,
                "EventRawVsSingularityParity_CascadeJoin",
                route_kind="cascade_frame_join",
                input_state=valid,
                output_state=out,
                metadata={
                    "segments": len(valid),
                    "concat_performed": True,
                    "concat_dim": 0,
                    "input_shapes": [list(b.shape) if hasattr(b, "shape") else str(type(b)) for b in valid],
                    "output_shape": list(out.shape) if hasattr(out, "shape") else str(type(out)),
                },
            )
            records.append({
                "stage": "SingularityCascadeFrameConcat",
                "status": "ok",
                "segments": len(valid),
                "shape": list(out.shape) if hasattr(out, "shape") else str(type(out)),
            })
            return out
        except Exception as e:
            records.append({
                "stage": "SingularityCascadeFrameConcat",
                "status": "failed_using_last_segment_only",
                "segments": len(valid),
                "error": str(e),
            })
            return valid[-1]

    def _concat_frame_batches_for_pause_preview(self, batches, records, segment_index):
        valid = [b for b in batches if b is not None]
        if not valid:
            return None
        if len(valid) == 1:
            self._record_raw_vs_singularity_parity_probe(
                records,
                "EventRawVsSingularityParity_PausePreviewJoin",
                route_kind="cascade_pause_preview_join",
                input_state=valid,
                output_state=valid[0],
                metadata={
                    "segment_index": int(segment_index),
                    "segments": 1,
                    "concat_performed": False,
                    "input_shapes": [list(b.shape) if hasattr(b, "shape") else str(type(b)) for b in valid],
                    "output_shape": list(valid[0].shape) if hasattr(valid[0], "shape") else str(type(valid[0])),
                },
            )
            return valid[0]
        try:
            import torch
            out = torch.cat(valid, dim=0)
            self._record_raw_vs_singularity_parity_probe(
                records,
                "EventRawVsSingularityParity_PausePreviewJoin",
                route_kind="cascade_pause_preview_join",
                input_state=valid,
                output_state=out,
                metadata={
                    "segment_index": int(segment_index),
                    "segments": len(valid),
                    "concat_performed": True,
                    "concat_dim": 0,
                    "input_shapes": [list(b.shape) if hasattr(b, "shape") else str(type(b)) for b in valid],
                    "output_shape": list(out.shape) if hasattr(out, "shape") else str(type(out)),
                },
            )
            records.append({
                "stage": "SingularityCascadePausePreviewConcat",
                "status": "ok",
                "segment_index": int(segment_index),
                "segments": len(valid),
                "shape": list(out.shape) if hasattr(out, "shape") else str(type(out)),
                "formula": "ordered cascade segment outcomes are temporarily stitched only for pause preview, not committed as final video output",
            })
            return out
        except Exception as e:
            records.append({
                "stage": "SingularityCascadePausePreviewConcat",
                "status": "failed_using_current_segment_only",
                "segment_index": int(segment_index),
                "segments": len(valid),
                "error": str(e),
            })
            return valid[-1]

    def _cascade_prompt_for_segment(self, prompt_text, segment_index, records=None, kind="positive"):
        raw = str(prompt_text or "")
        if not raw.strip():
            return raw

        def marker_number(line):
            text = str(line or "").strip()
            if not text:
                return None
            while text.startswith("#"):
                text = text[1:].strip()
            if text.startswith("::") and text.endswith("::"):
                text = text[2:-2].strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1].strip()
            match = re.match(r"(?i)^(?:cascade|segment|prompt)\s*[_\s:-]*(\d+)$", text)
            if not match:
                return None
            try:
                return int(match.group(1))
            except Exception:
                return None

        segments = {}
        current = None
        for line in raw.splitlines():
            number = marker_number(line)
            if number is not None:
                current = number
                segments.setdefault(number, [])
                continue
            if current is not None:
                segments.setdefault(current, []).append(line)

        if not segments:
            return raw

        requested = int(segment_index or 1)
        if requested in segments:
            selected_key = requested
            fallback = False
        else:
            lower = [key for key in segments if key <= requested]
            selected_key = max(lower) if lower else min(segments)
            fallback = True

        selected = "\n".join(segments.get(selected_key, [])).strip()
        if not selected:
            selected = raw.strip()
            fallback = True

        if records is not None:
            records.append({
                "stage": "EventCascadePromptSchedule",
                "status": "selected" if not fallback else "fallback_selected",
                "kind": str(kind or "prompt"),
                "requested_segment": requested,
                "selected_segment": int(selected_key),
                "available_segments": sorted(int(key) for key in segments),
                "text_length": len(selected),
                "formula_role": "per-segment prompt text -> StrategyCandidate carrier",
                "marker_syntax": "### Cascade 1 / [Cascade 1] / ::Cascade 1::",
            })
        return selected

    def _build_cascade_remaining_strategy(
        self,
        *,
        positive_prompt,
        negative_prompt,
        pause_segment_index,
        next_segment_index,
        resume_frame_index,
        frames_per_cascade,
        frames=None,
        records,
    ):
        """
        Build a non-text Strategy memory for the next cascade.

        This does not rewrite the prompt. It gives the Strategy Control Surface a
        measured "remaining route" after a pause cut, so the next sampler segment
        is less likely to reinterpret the global prompt as a fresh full event.
        """
        def clamp(value, lo=0.0, hi=1.0):
            try:
                value = float(value)
            except Exception:
                value = 0.0
            return max(lo, min(hi, value))

        positive = str(positive_prompt or "").lower()
        negative = str(negative_prompt or "").lower()
        text = f"{positive}\n{negative}"
        frames = max(1, int(frames_per_cascade or 1))
        resume = max(1, int(resume_frame_index or 1))
        progress = clamp(resume / float(frames))
        remaining = clamp(1.0 - progress)

        motion_terms = {
            "rotate": ("rotate", "rotation", "turn", "turning", "360", "axis", "spin"),
            "action": (
                "move", "moves", "moving", "motion", "animate", "animation", "walk", "walking",
                "run", "running", "raise", "raises", "lower", "lowers", "lift", "lifts",
                "lean", "leans", "bend", "bends", "sit", "sits", "stand", "stands",
                "open", "opens", "close", "closes", "slide", "slides", "push", "pull",
                "touch", "hold", "grab", "look", "turns", "breath", "breathing",
            ),
            "interaction": (
                "contact", "between", "inside", "outside", "against", "through", "toward",
                "away", "with her", "with his", "object", "hand", "hands", "body",
                "fabric", "hair", "face", "camera", "floor", "background",
            ),
            "endpoint": ("return", "returns", "endpoint", "same state", "starting frame", "original", "precisely", "continuity"),
            "fixed_camera": ("fixed camera", "no camera movement", "same framing", "same lighting"),
            "identity": ("same", "identity", "same expression", "same body", "same hair", "same dress"),
        }
        hits = {
            group: [term for term in terms if term in text]
            for group, terms in motion_terms.items()
        }
        rotation_pressure = clamp(len(hits["rotate"]) / 4.0)
        action_pressure = clamp(len(hits["action"]) / 6.0)
        interaction_pressure = clamp(len(hits["interaction"]) / 6.0)
        motion_pressure = max(rotation_pressure, action_pressure, interaction_pressure)
        endpoint_pressure = clamp(len(hits["endpoint"]) / 5.0)
        anchor_pressure = clamp((len(hits["fixed_camera"]) + len(hits["identity"])) / 5.0)
        route_pressure = clamp((motion_pressure * 0.45) + (endpoint_pressure * 0.35) + (anchor_pressure * 0.20))

        late_cut_pressure = clamp((progress - 0.55) / 0.45)
        restart_risk = clamp(route_pressure * late_cut_pressure)
        tail_observed_behavior = self._compute_cascade_tail_observed_behavior(
            frames,
            resume_frame_index=resume,
            records=records,
        )
        try:
            tail_motion_energy = float(tail_observed_behavior.get("motion_energy", 0.0) or 0.0)
        except Exception:
            tail_motion_energy = 0.0
        motion_memory_pressure = clamp(tail_motion_energy * late_cut_pressure * route_pressure)
        # The later the selected cut is, the less the next high branch should birth
        # a new full route. Low stays close to neutral and only refines continuity.
        high_field_intent_multiplier = 1.0 - (0.72 * restart_risk)
        high_field_window_multiplier = 1.0 - (0.45 * restart_risk)
        low_field_intent_multiplier = 1.0 - (0.25 * restart_risk)
        low_field_window_multiplier = 1.0 - (0.20 * restart_risk)

        memory = {
            "stage": "EventCascadeRemainingStrategy",
            "status": "active" if route_pressure > 0.0 else "neutral",
            "strategy_version": "cascade_remaining_strategy_v1",
            "pause_segment_index": int(pause_segment_index or 1),
            "applies_to_segment": int(next_segment_index or 1),
            "resume_frame_index": int(resume),
            "frames_per_cascade": int(frames),
            "progress_ratio": float(progress),
            "remaining_ratio": float(remaining),
            "rotation_pressure": float(rotation_pressure),
            "action_pressure": float(action_pressure),
            "interaction_pressure": float(interaction_pressure),
            "motion_pressure": float(motion_pressure),
            "endpoint_pressure": float(endpoint_pressure),
            "anchor_pressure": float(anchor_pressure),
            "route_pressure": float(route_pressure),
            "late_cut_pressure": float(late_cut_pressure),
            "restart_risk": float(restart_risk),
            "tail_observed_behavior": tail_observed_behavior,
            "motion_memory_pressure": float(motion_memory_pressure),
            "branch_multipliers": {
                "high_field_intent_multiplier": float(max(0.18, min(1.0, high_field_intent_multiplier))),
                "high_field_window_multiplier": float(max(0.35, min(1.0, high_field_window_multiplier))),
                "low_field_intent_multiplier": float(max(0.65, min(1.0, low_field_intent_multiplier))),
                "low_field_window_multiplier": float(max(0.75, min(1.0, low_field_window_multiplier))),
            },
            "hits": hits,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "formula": (
                "Selected tail Outcome(t-1) plus observed route progress becomes RemainingStrategy(t) "
                "for the next cascade; this is numeric Strategy memory, not prompt rewriting."
            ),
        }
        if records is not None:
            records.append(memory)
        return memory

    def _compute_cascade_tail_observed_behavior(self, frames, *, resume_frame_index, records=None):
        """
        Read the selected tail as ObservedBehavior(t-1), not just as a still frame.

        This is deliberately lightweight: no external vision model, no prompt
        text, only frame-difference energy and a coarse screen-space centroid.
        It gives the next Strategy surface evidence that the cut has a moving
        direction/phase instead of being a fresh static source image.
        """
        neutral = {
            "status": "unavailable",
            "tail_window": 0,
            "resume_frame_index": int(resume_frame_index or 0),
            "motion_energy": 0.0,
            "centroid_shift_x": 0.0,
            "centroid_shift_y": 0.0,
            "direction_hint": "unknown",
            "formula_role": "ObservedBehavior(t-1) tail motion memory",
        }
        try:
            import torch

            t = self._tensor_from_latent_like(frames)
            if t is None or not hasattr(t, "dim"):
                if records is not None:
                    records.append({"stage": "EventCascadeTailObservedBehavior", **neutral})
                return neutral
            if t.dim() == 3:
                t = t.unsqueeze(0)
            elif t.dim() == 5:
                t = t.reshape((-1,) + tuple(t.shape[-3:]))
            if t.dim() != 4:
                if records is not None:
                    records.append({
                        "stage": "EventCascadeTailObservedBehavior",
                        **neutral,
                        "status": "unsupported_shape",
                        "shape": list(t.shape) if hasattr(t, "shape") else None,
                    })
                return {**neutral, "status": "unsupported_shape", "shape": list(t.shape) if hasattr(t, "shape") else None}
            n = int(t.shape[0])
            if n < 2:
                if records is not None:
                    records.append({"stage": "EventCascadeTailObservedBehavior", **neutral, "status": "too_short"})
                return {**neutral, "status": "too_short"}

            end = max(1, min(n, int(resume_frame_index or n)))
            start = max(0, end - 8)
            window = t[start:end].detach().float()
            if window.shape[0] < 2:
                if records is not None:
                    records.append({"stage": "EventCascadeTailObservedBehavior", **neutral, "status": "too_short_after_cut"})
                return {**neutral, "status": "too_short_after_cut"}

            diffs = torch.abs(window[1:] - window[:-1]).mean(dim=-1)
            h = int(diffs.shape[1])
            w = int(diffs.shape[2])
            eps = 1e-8
            x_coords = torch.linspace(-1.0, 1.0, w, device=diffs.device, dtype=diffs.dtype).view(1, 1, w)
            y_coords = torch.linspace(-1.0, 1.0, h, device=diffs.device, dtype=diffs.dtype).view(1, h, 1)
            weights = diffs + eps
            totals = weights.sum(dim=(1, 2)).clamp_min(eps)
            cx = (weights * x_coords).sum(dim=(1, 2)) / totals
            cy = (weights * y_coords).sum(dim=(1, 2)) / totals
            centroid_shift_x = float((cx[-1] - cx[0]).item()) if cx.numel() else 0.0
            centroid_shift_y = float((cy[-1] - cy[0]).item()) if cy.numel() else 0.0
            motion_energy = float(diffs.mean().item()) if diffs.numel() else 0.0
            left_energy = float(diffs[:, :, : max(1, w // 3)].mean().item()) if diffs.numel() else 0.0
            right_energy = float(diffs[:, :, max(0, (2 * w) // 3):].mean().item()) if diffs.numel() else 0.0
            if abs(centroid_shift_x) < 0.015:
                direction_hint = "centered_or_axis_rotation"
            elif centroid_shift_x > 0.0:
                direction_hint = "rightward_screen_motion"
            else:
                direction_hint = "leftward_screen_motion"

            record = {
                "status": "observed",
                "shape": list(t.shape) if hasattr(t, "shape") else None,
                "tail_window": int(window.shape[0]),
                "resume_frame_index": int(resume_frame_index or end),
                "motion_energy": motion_energy,
                "centroid_shift_x": centroid_shift_x,
                "centroid_shift_y": centroid_shift_y,
                "left_motion_energy": left_energy,
                "right_motion_energy": right_energy,
                "direction_hint": direction_hint,
                "formula_role": "ObservedBehavior(t-1) tail motion memory",
                "formula": "Tail frame differences bind the selected Outcome(t-1) to a moving route phase before the next cascade Strategy.",
            }
            if records is not None:
                records.append({"stage": "EventCascadeTailObservedBehavior", **record})
            return record
        except Exception as e:
            failed = {**neutral, "status": "failed", "error": str(e)}
            if records is not None:
                records.append({"stage": "EventCascadeTailObservedBehavior", **failed})
            return failed

    def _build_cascade_phase_prompt_transform(
        self,
        *,
        positive_prompt,
        remaining_strategy,
        applies_to_segment,
        preserve_prompt_carrier=False,
        preserve_reason="",
        records=None,
    ):
        """
        Clean natural-language segment transform for TRANSFORM_PROMPT.

        It is not formula prose and not an appended explanation layer. It replaces
        the next segment's StrategyCandidate with a local continuation route when
        the global prompt would otherwise restart the full cascade event.
        """
        raw = str(positive_prompt or "").strip()
        if not raw:
            return raw, {"status": "empty_prompt"}
        strategy = remaining_strategy if isinstance(remaining_strategy, dict) else {}
        try:
            restart_risk = float(strategy.get("restart_risk", 0.0) or 0.0)
            progress = float(strategy.get("progress_ratio", 0.0) or 0.0)
            rotation_pressure = float(strategy.get("rotation_pressure", 0.0) or 0.0)
            motion_pressure = float(strategy.get("motion_pressure", rotation_pressure) or 0.0)
        except Exception:
            restart_risk = 0.0
            progress = 0.0
            rotation_pressure = 0.0
            motion_pressure = 0.0
        if preserve_prompt_carrier:
            summary = {
                "status": "skipped_same_prompt_strategy_carrier_preserved",
                "transform_version": "cascade_phase_prompt_transform_v2_same_prompt_guard",
                "applies_to_segment": int(applies_to_segment),
                "restart_risk": float(restart_risk),
                "progress_ratio": float(progress),
                "motion_pressure": float(motion_pressure),
                "preserve_reason": str(preserve_reason or "same_prompt_identity_reused"),
                "prompt_continuity_reused": True,
                "positive_prompt_changed": False,
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "policy": "same_prompt_preserve_clean_strategy_candidate",
                "formula": (
                    "Continue reused the same prompt identity; RemainingStrategy stays numeric, "
                    "and the original clean StrategyCandidate is not rewritten for the next segment."
                ),
            }
            if records is not None:
                records.append({"stage": "EventCascadePhasePromptTransform", **summary})
            return raw, summary
        if restart_risk < 0.35 or motion_pressure <= 0.0:
            summary = {
                "status": "not_needed",
                "applies_to_segment": int(applies_to_segment),
                "restart_risk": float(restart_risk),
                "progress_ratio": float(progress),
                "motion_pressure": float(motion_pressure),
                "positive_prompt_changed": False,
            }
            if records is not None:
                records.append({"stage": "EventCascadePhasePromptTransform", **summary})
            return raw, summary

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
        restart_terms = (
            "begins", "begin ", "starts from", "starts ", "video starts",
            "completes the full 360", "full 360", "360-degree rotation",
            "returns precisely", "return precisely", "returns to the original",
            "starting frame", "original front-facing pose", "from the reference image",
            "exact same reference", "same reference photo", "first frame",
        )
        keep = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(term in lower for term in restart_terms):
                continue
            keep.append(sentence)
        if not keep:
            keep = sentences[:2]

        tail_behavior = strategy.get("tail_observed_behavior", {}) if isinstance(strategy, dict) else {}
        direction_hint = str(tail_behavior.get("direction_hint", "existing direction") if isinstance(tail_behavior, dict) else "existing direction")
        has_rotation_route = rotation_pressure > 0.0
        late_phase = progress >= 0.85
        if late_phase:
            local_motion_sentence = (
                "Treat the selected resume frame as a late phase of the event. "
                "Do not perform the earlier middle actions again; continue only the remaining motion needed to converge toward the described endpoint. "
            )
        elif has_rotation_route:
            local_motion_sentence = (
                "Continue the already established rotation in the same direction from this exact pose phase. "
                "Do not reverse the turn and do not restart the earlier part of the motion. "
            )
        else:
            local_motion_sentence = (
                "Continue the already established action and object relationships from this exact temporal phase. "
                "Do not reverse, reset, duplicate, or restart the earlier part of the event. "
            )
        phase_text = (
            "Start this cascade segment from the selected resume frame as the current state, not from the original first frame. "
            + local_motion_sentence +
            "Use the selected frame as the local anchor, preserve identity, camera, background continuity, and object relationships, "
            "and move toward the remaining endpoint of the same event with smooth continuity."
        )
        if late_phase:
            keep_scored = []
            middle_terms = (
                "as ", "begins", "begin", "starts", "start", "revealing", "reveals",
                "lift", "lifts", "lower", "lowers", "gradually", "slowly",
                "while", "when ", "during", "into view", "back rotates into view",
            )
            endpoint_terms = (
                "endpoint", "continuity", "same state", "returns", "return",
                "preserve", "same", "fixed camera", "no camera", "identity",
                "smooth", "consistent", "natural extension",
            )
            for sentence in keep:
                lower = sentence.lower()
                endpoint_score = sum(1 for term in endpoint_terms if term in lower)
                middle_score = sum(1 for term in middle_terms if term in lower)
                if endpoint_score >= middle_score:
                    keep_scored.append(sentence)
            if keep_scored:
                keep = keep_scored
        transformed = " ".join([phase_text] + keep).strip()
        summary = {
            "status": "applied",
            "transform_version": "cascade_phase_prompt_transform_v1",
            "applies_to_segment": int(applies_to_segment),
            "restart_risk": float(restart_risk),
            "progress_ratio": float(progress),
            "motion_pressure": float(motion_pressure),
            "rotation_specific": bool(has_rotation_route),
            "late_phase": bool(late_phase),
            "direction_hint": direction_hint,
            "removed_restart_sentence_count": int(max(0, len(sentences) - len(keep))),
            "positive_prompt_changed": transformed != raw,
            "prompt_text_injection_allowed": False,
            "policy": "clean_segment_strategy_transform_replace_not_append",
            "formula": "Global Strategy is locally re-read after a pause: the next prompt carrier describes the remaining route, not a fresh full event.",
        }
        if records is not None:
            records.append({"stage": "EventCascadePhasePromptTransform", **summary})
        return transformed, summary

    def _activate_cascade_remaining_strategy(self, memory, records=None):
        if not isinstance(memory, dict):
            return
        self._event_cascade_remaining_strategy = memory
        if records is not None:
            records.append({
                "stage": "EventCascadeRemainingStrategyBind",
                "status": str(memory.get("status", "unknown")),
                "applies_to_segment": int(memory.get("applies_to_segment", 0) or 0),
                "resume_frame_index": int(memory.get("resume_frame_index", 0) or 0),
                "restart_risk": float(memory.get("restart_risk", 0.0) or 0.0),
                "branch_multipliers": memory.get("branch_multipliers", {}),
                "formula": "RemainingStrategy is bound to the next segment Strategy Control Surface before sampler execution.",
            })

    def _record_segment_strategy_carrier(self, records, *, segment_index, positive_prompt, negative_prompt, context=None):
        context = context if isinstance(context, dict) else {}

        def text_signature(value):
            return hashlib.sha256(str(value or "").encode("utf-8", errors="ignore")).hexdigest()[:16]

        positive_text = str(positive_prompt or "")
        negative_text = str(negative_prompt or "")
        positive_signature = text_signature(positive_text)
        negative_signature = text_signature(negative_text)
        previous_positive_signature = str(context.get("last_active_positive_signature") or "")
        previous_negative_signature = str(context.get("last_active_negative_signature") or "")
        segment_index_i = int(segment_index or 1)
        changed_positive = bool(previous_positive_signature and positive_signature != previous_positive_signature)
        changed_negative = bool(previous_negative_signature and negative_signature != previous_negative_signature)
        prompt_source = str(context.get("prompt_source") or ("launch_time" if segment_index_i == 1 else "previous_segment"))

        record = {
            "stage": "EventSegmentStrategyCarrierReview",
            "status": "recorded",
            "review_version": "segment_strategy_carrier_review_v1",
            "segment_index": segment_index_i,
            "prompt_source": prompt_source,
            "prompt_transcode_mode": str(context.get("prompt_transcode_mode") or ""),
            "active_positive_signature": positive_signature,
            "active_negative_signature": negative_signature,
            "previous_active_positive_signature": previous_positive_signature,
            "previous_active_negative_signature": previous_negative_signature,
            "positive_changed_from_previous_segment": changed_positive,
            "negative_changed_from_previous_segment": changed_negative,
            "changed_from_previous_segment": bool(changed_positive or changed_negative),
            "positive_prompt_length": len(positive_text),
            "negative_prompt_length": len(negative_text),
            "last_runtime_prompt_update_applies_to_segment": context.get("last_runtime_prompt_update_applies_to_segment", None),
            "formula": (
                "Segment prompt text is the local StrategyCarrier before CLIP encoding; "
                "this record proves whether a cascade continues the same carrier or receives a new runtime Strategy."
            ),
            "control_mode": "REPORT_ONLY",
        }
        records.append(record)

        context["last_active_positive_signature"] = positive_signature
        context["last_active_negative_signature"] = negative_signature
        context["last_segment_index"] = segment_index_i
        context["last_prompt_source"] = prompt_source
        return record

    def _encode_text_with_strategy_cache(self, clip, text, records, *, label, context=None, polarity="positive"):
        context = context if isinstance(context, dict) else {}
        text_value = str(text or "")
        text_signature = hashlib.sha256(text_value.encode("utf-8", errors="ignore")).hexdigest()[:16]
        clip_route_signature = self._object_route_cache_signature(clip)
        cache = context.setdefault("conditioning_cache", {})
        cache_key = f"{str(polarity or 'text')}:{clip_route_signature}:{text_signature}"

        if cache_key in cache:
            records.append({
                "stage": f"Event{label}",
                "status": "reused_conditioning",
                "cache_key": cache_key,
                "clip_route_signature": clip_route_signature,
                "text_signature": text_signature,
                "text_length": len(text_value),
                "formula": (
                    "Same prompt StrategyCarrier and same CLIP/encoder route reuse the already encoded conditioning; "
                    "a changed CLIP/LoRA text route must not reuse stale NumericStrategy conditioning."
                ),
                "control_mode": "REPORT_ONLY",
            })
            return cache[cache_key]

        conditioning = self._encode_text(clip, text_value, records, label=label)
        cache[cache_key] = conditioning
        records.append({
            "stage": f"Event{label}ConditioningCache",
            "status": "stored",
            "cache_key": cache_key,
            "clip_route_signature": clip_route_signature,
            "text_signature": text_signature,
            "text_length": len(text_value),
            "formula": "Prompt StrategyCarrier text encoded once into NumericStrategy conditioning for reuse when text and CLIP/encoder route are unchanged.",
            "control_mode": "REPORT_ONLY",
        })
        return conditioning

    def _compact_route_value_summary(self, value, *, max_keys=12, max_items=8):
        try:
            if value is None:
                return {"type": "NoneType", "present": False}
            if isinstance(value, dict):
                keys = [str(k) for k in list(value.keys())[:max_keys]]
                return {
                    "type": type(value).__name__,
                    "present": True,
                    "len": len(value),
                    "keys": keys,
                }
            if isinstance(value, (list, tuple, set)):
                items = []
                for item in list(value)[:max_items]:
                    items.append(type(item).__name__)
                return {
                    "type": type(value).__name__,
                    "present": True,
                    "len": len(value),
                    "item_types": items,
                }
            shape = getattr(value, "shape", None)
            if shape is not None:
                return {
                    "type": type(value).__name__,
                    "present": True,
                    "shape": [int(x) for x in list(shape)[:8]],
                    "dtype": str(getattr(value, "dtype", "")),
                }
            return {
                "type": type(value).__name__,
                "present": True,
            }
        except Exception as exc:
            return {
                "type": type(value).__name__ if value is not None else "NoneType",
                "present": value is not None,
                "summary_error": str(exc)[:160],
            }

    def _probe_operator_route_object(self, obj, role):
        probe = {
            "role": str(role),
            "present": obj is not None,
            "object_type": type(obj).__name__ if obj is not None else "NoneType",
            "object_module": type(obj).__module__ if obj is not None else "",
            "runtime_object_id": hex(id(obj)) if obj is not None else "",
            "probe_policy": "metadata_only_no_tensor_mutation",
        }
        if obj is None:
            probe["route_signature"] = hashlib.sha256(f"{role}:none".encode("utf-8")).hexdigest()[:16]
            return probe

        attr_names = [
            "patches",
            "object_patches",
            "model_options",
            "model",
            "model_sampling",
            "sampling",
            "model_config",
            "load_device",
            "offload_device",
            "size",
            "loaded_size",
        ]
        attrs = {}
        for attr_name in attr_names:
            try:
                if hasattr(obj, attr_name):
                    attrs[attr_name] = self._compact_route_value_summary(getattr(obj, attr_name))
            except Exception as exc:
                attrs[attr_name] = {
                    "type": "unknown",
                    "present": True,
                    "summary_error": str(exc)[:160],
                }
        probe["attributes"] = attrs

        patch_summary = attrs.get("patches", {})
        object_patch_summary = attrs.get("object_patches", {})
        model_options_summary = attrs.get("model_options", {})
        probe["patch_evidence"] = {
            "patches_count": int(patch_summary.get("len", 0) or 0) if isinstance(patch_summary, dict) else 0,
            "object_patches_count": int(object_patch_summary.get("len", 0) or 0) if isinstance(object_patch_summary, dict) else 0,
            "model_options_count": int(model_options_summary.get("len", 0) or 0) if isinstance(model_options_summary, dict) else 0,
            "has_any_patch_evidence": bool(
                (isinstance(patch_summary, dict) and int(patch_summary.get("len", 0) or 0) > 0)
                or (isinstance(object_patch_summary, dict) and int(object_patch_summary.get("len", 0) or 0) > 0)
            ),
            "interpretation": (
                "LoRA/SD3/compile patches are usually visible as ModelPatcher metadata, "
                "but exact LoRA filenames may not survive inside the runtime MODEL object."
            ),
        }

        stable_probe = dict(probe)
        stable_probe.pop("runtime_object_id", None)
        try:
            raw = json.dumps(stable_probe, sort_keys=True, default=str)
        except Exception:
            raw = repr(stable_probe)
        probe["route_signature"] = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return probe

    def _object_route_cache_signature(self, obj):
        if obj is None:
            return "none"
        try:
            probe = self._probe_operator_route_object(obj, "cache_route")
            route_signature = str(probe.get("route_signature") or "")
            return hashlib.sha256(
                f"{type(obj).__module__}.{type(obj).__name__}:{id(obj)}:{route_signature}".encode("utf-8", errors="ignore")
            ).hexdigest()[:16]
        except Exception:
            return hashlib.sha256(f"{type(obj).__name__}:{id(obj)}".encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _record_external_operator_route_diagnostics(
        self,
        records,
        *,
        primary_model,
        secondary_model,
        clip,
        vae,
        global_steps,
        primary_start_step,
        primary_end_step,
        secondary_start_step,
        secondary_end_step,
        primary_sd3_shift,
        secondary_sd3_shift,
    ):
        try:
            steps = max(int(global_steps or 0), 1)
        except Exception:
            steps = 1
        try:
            high_span = max(0, int(primary_end_step) - int(primary_start_step))
        except Exception:
            high_span = 0
        try:
            low_span = max(0, int(secondary_end_step) - int(secondary_start_step))
        except Exception:
            low_span = 0
        route = {
            "stage": "EventExternalOperatorRouteDiagnostics",
            "status": "recorded",
            "probe_policy": "metadata_only_no_tensor_mutation",
            "primary_model": self._probe_operator_route_object(primary_model, "primary_model_high_operator"),
            "secondary_model": self._probe_operator_route_object(secondary_model, "secondary_model_low_operator"),
            "clip": self._probe_operator_route_object(clip, "clip_text_encoder"),
            "vae": self._probe_operator_route_object(vae, "vae_decoder_encoder"),
            "sampler_window_split": {
                "global_steps": steps,
                "primary_start_step": int(primary_start_step),
                "primary_end_step": int(primary_end_step),
                "primary_span": high_span,
                "primary_fraction": round(float(high_span) / float(steps), 6),
                "secondary_start_step": int(secondary_start_step),
                "secondary_end_step": int(secondary_end_step),
                "secondary_span": low_span,
                "secondary_fraction": round(float(low_span) / float(steps), 6),
            },
            "internal_sd3_shift_request": {
                "primary_sd3_shift": float(primary_sd3_shift or 0.0),
                "secondary_sd3_shift": float(secondary_sd3_shift or 0.0),
                "interpretation": (
                    "Singularity clean node normally expects SD3 shift to be applied outside; "
                    "zero here means passthrough, not proof that incoming models are unshifted."
                ),
            },
            "formula": (
                "External MODEL/CLIP/VAE route is the Operator side of Strategy(t). "
                "If raw Wan is sharper, the first parity question is whether Singularity received the same post-LoRA, post-SD3, post-compile operator objects."
            ),
            "next_route": (
                "Compare this record against raw PNG workflow metadata. "
                "Missing patch evidence on primary/secondary MODEL while raw workflow uses LoRA loaders means the external route is not equivalent."
            ),
            "control_mode": "REPORT_ONLY",
        }
        records.append(route)
        return route

    def _record_workflow_graph_route_diagnostics(
        self,
        records,
        *,
        workflow_prompt=None,
        workflow_extra_pnginfo=None,
        unique_id=None,
    ):
        def safe_node_inputs(node):
            inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
            if not isinstance(inputs, dict):
                return {}
            keep = {}
            allowed_exact = {
                "lora_name",
                "strength_model",
                "strength_clip",
                "unet_name",
                "ckpt_name",
                "clip_name",
                "vae_name",
                "shift",
                "model",
                "clip",
                "vae",
                "sampler_name",
                "scheduler",
                "steps",
                "cfg",
                "start_at_step",
                "end_at_step",
                "add_noise",
                "return_with_leftover_noise",
                "noise_seed",
            }
            for key, value in inputs.items():
                key_s = str(key)
                if key_s in allowed_exact or any(part in key_s.lower() for part in ("lora", "model", "clip", "vae", "shift")):
                    keep[key_s] = _event_json_safe(value)
            return keep

        def active_node_record(node_id, node):
            class_type = str(node.get("class_type") or node.get("type") or "")
            meta = node.get("_meta", {}) if isinstance(node.get("_meta", {}), dict) else {}
            return {
                "id": str(node_id),
                "class_type": class_type,
                "title": str(meta.get("title", "") or ""),
                "inputs": safe_node_inputs(node),
            }

        prompt_is_dict = isinstance(workflow_prompt, dict)
        records_out = []
        lora_nodes = []
        sd3_nodes = []
        sampler_nodes = []
        loader_nodes = []
        clip_nodes = []
        compile_nodes = []
        if prompt_is_dict:
            for node_id, node in workflow_prompt.items():
                if not isinstance(node, dict):
                    continue
                class_type = str(node.get("class_type") or node.get("type") or "")
                low = class_type.lower()
                rec = active_node_record(node_id, node)
                if "lora" in low:
                    lora_nodes.append(rec)
                    records_out.append(rec)
                elif "modelsamplingsd3" in low or ("model" in low and "sampling" in low):
                    sd3_nodes.append(rec)
                    records_out.append(rec)
                elif "ksampler" in low or "sampler" in low:
                    sampler_nodes.append(rec)
                    records_out.append(rec)
                elif "unetloader" in low or "gguf" in low:
                    loader_nodes.append(rec)
                    records_out.append(rec)
                elif "cliploader" in low:
                    clip_nodes.append(rec)
                    records_out.append(rec)
                elif "compile" in low or "torch" in low:
                    compile_nodes.append(rec)
                    records_out.append(rec)

        visible_node_count = None
        try:
            workflow = None
            if isinstance(workflow_extra_pnginfo, dict):
                workflow = workflow_extra_pnginfo.get("workflow")
            if isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list):
                visible_node_count = len(workflow.get("nodes") or [])
        except Exception:
            visible_node_count = None

        status = "recorded" if prompt_is_dict else "unavailable_hidden_prompt"
        record = {
            "stage": "EventWorkflowGraphRouteDiagnostics",
            "status": status,
            "probe_policy": "hidden_prompt_graph_metadata_only_no_prompt_text",
            "node_id": str(unique_id or ""),
            "active_prompt_graph_available": bool(prompt_is_dict),
            "active_prompt_node_count": len(workflow_prompt) if prompt_is_dict else 0,
            "visible_workflow_node_count": visible_node_count,
            "lora_nodes": lora_nodes,
            "sd3_nodes": sd3_nodes,
            "sampler_nodes": sampler_nodes,
            "loader_nodes": loader_nodes,
            "clip_nodes": clip_nodes,
            "compile_nodes": compile_nodes,
            "active_route_nodes": records_out,
            "lora_node_count": len(lora_nodes),
            "sd3_node_count": len(sd3_nodes),
            "sampler_node_count": len(sampler_nodes),
            "compile_node_count": len(compile_nodes),
            "formula": (
                "The ComfyUI active graph is a route witness for Strategy Operator construction: "
                "LoRA -> SD3 shift -> sampler should be visible as graph metadata before tensor execution."
            ),
            "next_route": (
                "Compare lora_nodes and sd3_nodes against EventExternalOperatorRouteDiagnostics patch evidence. "
                "If graph has LoRA but MODEL patch evidence is missing, the operator route is broken before Singularity."
            ),
            "control_mode": "REPORT_ONLY",
        }
        records.append(record)
        return record

    def _record_segment_entry_strategy_return_probe(
        self,
        records,
        *,
        segment_index,
        source_image=None,
        scaled_image=None,
        previous_segment_latent=None,
        wan_positive=None,
        wan_negative=None,
        wan_latent=None,
        latent_after_high=None,
        latent_after_low=None,
        frames_out=None,
        strategy_carrier_context=None,
    ):
        """
        r110 report-only probe for the real cascade continuity problem:
        decoded tail image continuity may be exact while latent Strategy memory
        is lost between previous segment latent tail and the next Wan/sampler entry.
        """
        def _frame_slice(obj, mode="first"):
            try:
                t = self._tensor_from_latent_like(obj)
                if t is None:
                    return None
                if hasattr(t, "dim"):
                    if t.dim() == 4 and int(t.shape[0]) > 0:
                        return t[-1:] if mode == "last" else t[:1]
                    if t.dim() == 3:
                        return t.unsqueeze(0)
                return t
            except Exception:
                return None

        def _latent_time_slice(obj, mode="first"):
            try:
                t = self._tensor_from_latent_like(obj)
                if t is None:
                    return None
                if hasattr(t, "dim") and t.dim() == 5 and int(t.shape[2]) > 0:
                    return t[:, :, -1:, :, :] if mode == "last" else t[:, :, :1, :, :]
                return t
            except Exception:
                return None

        def _find_named_tensor(obj, key_name):
            try:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if str(k) == str(key_name) and hasattr(v, "detach") and hasattr(v, "shape"):
                            return v
                    for v in obj.values():
                        found = _find_named_tensor(v, key_name)
                        if found is not None:
                            return found
                if isinstance(obj, (list, tuple)):
                    for v in obj:
                        found = _find_named_tensor(v, key_name)
                        if found is not None:
                            return found
            except Exception:
                return None
            return None

        def _compare(a, b):
            try:
                import torch
                if a is None or b is None:
                    return {"status": "unavailable"}
                af = torch.nan_to_num(a.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                bf = torch.nan_to_num(b.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                if list(af.shape) != list(bf.shape):
                    return {
                        "status": "shape_mismatch",
                        "a_shape": [int(x) for x in list(af.shape)],
                        "b_shape": [int(x) for x in list(bf.shape)],
                    }
                d = bf - af
                abs_mean = float(d.abs().mean().item())
                norm = float(torch.linalg.vector_norm(d).item())
                max_abs = float(d.abs().max().item())
                if abs_mean <= 1.0e-7 and max_abs <= 1.0e-6:
                    band = "exact"
                elif abs_mean <= 0.003:
                    band = "near_exact"
                elif abs_mean <= 0.03:
                    band = "small_shift"
                elif abs_mean <= 0.10:
                    band = "visible_shift"
                else:
                    band = "strong_shift"
                return {
                    "status": "ok",
                    "band": band,
                    "abs_mean": abs_mean,
                    "norm": norm,
                    "max_abs": max_abs,
                    "shape": [int(x) for x in list(af.shape)],
                }
            except Exception as e:
                return {"status": "failed", "error": str(e)[:240]}

        try:
            source_frame = _frame_slice(source_image, "first")
            scaled_frame = _frame_slice(scaled_image, "first")
            decoded_first = _frame_slice(frames_out, "first")
            decoded_second = None
            try:
                ft = self._tensor_from_latent_like(frames_out)
                if ft is not None and hasattr(ft, "dim") and ft.dim() == 4 and int(ft.shape[0]) > 1:
                    decoded_second = ft[1:2]
            except Exception:
                decoded_second = None

            previous_latent_tail = _latent_time_slice(previous_segment_latent, "last")
            wan_latent_first = _latent_time_slice(wan_latent, "first")
            high_first = _latent_time_slice(latent_after_high, "first")
            low_first = _latent_time_slice(latent_after_low, "first")
            concat_latent = _find_named_tensor(wan_positive, "concat_latent_image")
            concat_mask = _find_named_tensor(wan_positive, "concat_mask")
            concat_first = _latent_time_slice(concat_latent, "first")

            image_metrics = {
                "source_vs_scaled": _compare(source_frame, scaled_frame),
                "source_vs_decoded_first": _compare(source_frame, decoded_first),
                "source_vs_decoded_second": _compare(source_frame, decoded_second),
            }
            latent_metrics = {
                "previous_latent_tail_vs_wan_concat_first": _compare(previous_latent_tail, concat_first),
                "previous_latent_tail_vs_wan_latent_first": _compare(previous_latent_tail, wan_latent_first),
                "wan_concat_first_vs_wan_latent_first": _compare(concat_first, wan_latent_first),
                "wan_concat_first_vs_high_first": _compare(concat_first, high_first),
                "wan_concat_first_vs_low_first": _compare(concat_first, low_first),
                "previous_latent_tail_vs_high_first": _compare(previous_latent_tail, high_first),
                "previous_latent_tail_vs_low_first": _compare(previous_latent_tail, low_first),
            }

            def _band(metric):
                return metric.get("band", metric.get("status", "unknown")) if isinstance(metric, dict) else "unknown"

            image_entry_band = _band(image_metrics.get("source_vs_decoded_second", {}))
            latent_memory_band = _band(latent_metrics.get("previous_latent_tail_vs_wan_concat_first", {}))
            sampler_entry_band = _band(latent_metrics.get("wan_concat_first_vs_low_first", {}))
            if latent_memory_band in ("strong_shift", "visible_shift", "small_shift"):
                status = "latent_memory_bridge_changed"
            elif sampler_entry_band in ("strong_shift", "visible_shift"):
                status = "sampler_entry_reinterprets_conditioning"
            elif image_entry_band in ("visible_shift", "strong_shift"):
                status = "visible_entry_reinterprets_source"
            else:
                status = "entry_strategy_return_observed"

            ctx = strategy_carrier_context if isinstance(strategy_carrier_context, dict) else {}
            record = {
                "stage": "EventSegmentEntryStrategyReturnProbe",
                "status": status,
                "probe_version": "segment_entry_strategy_return_probe_v1",
                "segment_index": int(segment_index),
                "source_image_probe": self._event_tensor_probe(source_frame, label=f"cascade_{segment_index}.entry_source_image"),
                "scaled_image_probe": self._event_tensor_probe(scaled_frame, label=f"cascade_{segment_index}.entry_scaled_image"),
                "previous_latent_tail_probe": self._event_tensor_probe(previous_latent_tail, label=f"cascade_{segment_index}.previous_latent_tail"),
                "wan_concat_latent_first_probe": self._event_tensor_probe(concat_first, label=f"cascade_{segment_index}.wan_concat_latent_first"),
                "wan_concat_mask_probe": self._event_tensor_probe(concat_mask, label=f"cascade_{segment_index}.wan_concat_mask"),
                "wan_latent_first_probe": self._event_tensor_probe(wan_latent_first, label=f"cascade_{segment_index}.wan_latent_first"),
                "latent_after_high_first_probe": self._event_tensor_probe(high_first, label=f"cascade_{segment_index}.latent_after_high_first"),
                "latent_after_low_first_probe": self._event_tensor_probe(low_first, label=f"cascade_{segment_index}.latent_after_low_first"),
                "decoded_first_frame_probe": self._event_tensor_probe(decoded_first, label=f"cascade_{segment_index}.decoded_first_frame"),
                "decoded_second_frame_probe": self._event_tensor_probe(decoded_second, label=f"cascade_{segment_index}.decoded_second_frame"),
                "image_entry_metrics": image_metrics,
                "latent_entry_metrics": latent_metrics,
                "entry_status_summary": {
                    "image_entry_band": image_entry_band,
                    "latent_memory_band": latent_memory_band,
                    "sampler_entry_band": sampler_entry_band,
                    "latent_memory_bridge_available": previous_latent_tail is not None and concat_first is not None,
                    "decoded_second_available": decoded_second is not None,
                },
                "strategy_carrier_context": {
                    "prompt_source": ctx.get("prompt_source", ""),
                    "prompt_transcode_mode": ctx.get("prompt_transcode_mode", ""),
                    "prompt_continuity_reused": bool(ctx.get("prompt_continuity_reused", False)),
                    "prompt_continuity_policy": ctx.get("prompt_continuity_policy", ""),
                    "current_active_positive_signature": ctx.get("current_active_positive_signature", ""),
                    "current_active_negative_signature": ctx.get("current_active_negative_signature", ""),
                },
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
                "formula": (
                    "Segment-entry Strategy return compares the preserved visible source with the hidden latent "
                    "memory route: previous latent tail -> Wan concat latent -> sampler high/low first slice -> "
                    "decoded first/second frame. This names where continuity is lost before active correction."
                ),
                "next_route": (
                    "If visible tail is preserved but latent_memory_band is high, test a future bounded latent-memory "
                    "bridge. If sampler_entry_band is high, test sampler-entry Strategy pressure before postprocess blending."
                ),
            }
            records.append(record)
            return record
        except Exception as e:
            record = {
                "stage": "EventSegmentEntryStrategyReturnProbe",
                "status": "failed",
                "segment_index": int(segment_index) if segment_index is not None else None,
                "error": str(e)[:240],
                "control_mode": "REPORT_ONLY",
                "active_control_allowed": False,
            }
            records.append(record)
            return record

    def _apply_segment_entry_latent_memory_bridge(
        self,
        *,
        segment_index,
        previous_segment_latent=None,
        wan_positive=None,
        wan_negative=None,
        wan_latent=None,
        records=None,
        strategy_carrier_context=None,
    ):
        """
        r113 bounded active research surface.

        The visible selected frame can be exact while WanImageToVideo rebuilds a
        fresh hidden latent entry. This bridge gives the next segment's first
        latent slice a small, clamped memory of the previous segment latent tail.
        It is intentionally not a full latent replacement.
        """
        if records is None:
            records = []

        def _latent_time_slice(obj, mode="first"):
            try:
                t = self._tensor_from_latent_like(obj)
                if t is None:
                    return None
                if hasattr(t, "dim") and t.dim() == 5 and int(t.shape[2]) > 0:
                    return t[:, :, -1:, :, :] if mode == "last" else t[:, :, :1, :, :]
                return t
            except Exception:
                return None

        def _find_named_tensor(obj, key_name):
            try:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if str(k) == str(key_name) and hasattr(v, "detach") and hasattr(v, "shape"):
                            return v
                    for v in obj.values():
                        found = _find_named_tensor(v, key_name)
                        if found is not None:
                            return found
                if isinstance(obj, (list, tuple)):
                    for v in obj:
                        found = _find_named_tensor(v, key_name)
                        if found is not None:
                            return found
            except Exception:
                return None
            return None

        def _replace_named_tensor(obj, key_name, replacement):
            try:
                if isinstance(obj, dict):
                    changed = False
                    out = {}
                    for k, v in obj.items():
                        if str(k) == str(key_name) and hasattr(v, "detach") and hasattr(v, "shape"):
                            out[k] = replacement
                            changed = True
                        else:
                            new_v, child_changed = _replace_named_tensor(v, key_name, replacement)
                            out[k] = new_v
                            changed = changed or child_changed
                    return (out if changed else obj), changed
                if isinstance(obj, list):
                    changed = False
                    out = []
                    for v in obj:
                        new_v, child_changed = _replace_named_tensor(v, key_name, replacement)
                        out.append(new_v)
                        changed = changed or child_changed
                    return (out if changed else obj), changed
                if isinstance(obj, tuple):
                    changed = False
                    out = []
                    for v in obj:
                        new_v, child_changed = _replace_named_tensor(v, key_name, replacement)
                        out.append(new_v)
                        changed = changed or child_changed
                    return (tuple(out) if changed else obj), changed
            except Exception:
                return obj, False
            return obj, False

        def _compare(a, b):
            try:
                import torch
                if a is None or b is None:
                    return {"status": "unavailable"}
                af = torch.nan_to_num(a.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                bf = torch.nan_to_num(b.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                if list(af.shape) != list(bf.shape):
                    return {
                        "status": "shape_mismatch",
                        "a_shape": [int(x) for x in list(af.shape)],
                        "b_shape": [int(x) for x in list(bf.shape)],
                    }
                d = bf - af
                abs_mean = float(d.abs().mean().item())
                norm = float(torch.linalg.vector_norm(d).item())
                max_abs = float(d.abs().max().item())
                if abs_mean <= 1.0e-7 and max_abs <= 1.0e-6:
                    band = "exact"
                elif abs_mean <= 0.003:
                    band = "near_exact"
                elif abs_mean <= 0.03:
                    band = "small_shift"
                elif abs_mean <= 0.10:
                    band = "visible_shift"
                else:
                    band = "strong_shift"
                return {
                    "status": "ok",
                    "band": band,
                    "abs_mean": abs_mean,
                    "norm": norm,
                    "max_abs": max_abs,
                    "shape": [int(x) for x in list(af.shape)],
                }
            except Exception as e:
                return {"status": "failed", "error": str(e)[:240]}

        def _blend_first_slice(base_tensor, previous_tail, *, alpha, max_step):
            try:
                import torch
                if base_tensor is None or previous_tail is None:
                    return base_tensor, {
                        "status": "unavailable",
                        "reason": "missing_tensor",
                    }
                if not (hasattr(base_tensor, "dim") and base_tensor.dim() == 5 and int(base_tensor.shape[2]) > 0):
                    return base_tensor, {
                        "status": "unsupported_shape",
                        "shape": [int(x) for x in list(base_tensor.shape)] if hasattr(base_tensor, "shape") else None,
                    }
                current_first = base_tensor[:, :, :1, :, :]
                target = previous_tail.to(device=base_tensor.device, dtype=torch.float32)
                current_f = torch.nan_to_num(current_first.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
                if list(current_f.shape) != list(target.shape):
                    return base_tensor, {
                        "status": "shape_mismatch",
                        "current_shape": [int(x) for x in list(current_f.shape)],
                        "target_shape": [int(x) for x in list(target.shape)],
                    }
                raw_delta = torch.nan_to_num(target - current_f, nan=0.0, posinf=0.0, neginf=0.0)
                bounded_delta = torch.clamp(raw_delta, min=-float(max_step), max=float(max_step))
                memory_delta = bounded_delta * float(alpha)
                out = base_tensor.clone()
                out[:, :, :1, :, :] = (current_f + memory_delta).to(dtype=base_tensor.dtype, device=base_tensor.device)
                first_after = out[:, :, :1, :, :]
                return out, {
                    "status": "applied",
                    "alpha": float(alpha),
                    "max_step": float(max_step),
                    "raw_delta_abs_mean": float(raw_delta.abs().mean().item()),
                    "raw_delta_max_abs": float(raw_delta.abs().max().item()),
                    "bounded_delta_abs_mean": float(bounded_delta.abs().mean().item()),
                    "bounded_delta_max_abs": float(bounded_delta.abs().max().item()),
                    "memory_delta_abs_mean": float(memory_delta.abs().mean().item()),
                    "memory_delta_max_abs": float(memory_delta.abs().max().item()),
                    "before_metric": _compare(target, current_first),
                    "after_metric": _compare(target, first_after),
                }
            except Exception as e:
                return base_tensor, {
                    "status": "failed",
                    "error": str(e)[:240],
                }

        def _control_float(controls, key, default, min_value, max_value):
            try:
                value = controls.get(key, controls.get(f"bridge_{key}", default)) if isinstance(controls, dict) else default
                out = float(value)
            except Exception:
                out = float(default)
            return max(float(min_value), min(float(max_value), out))

        try:
            mode = str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY") or "OBSERVE_ONLY").upper()
            strengths = getattr(self, "_event_delta_strengths", {}) or {}
            high_strength = float(strengths.get("high", 1.0) or 1.0)
            low_strength = float(strengths.get("low", 1.0) or 1.0)
        except Exception:
            mode = "OBSERVE_ONLY"
            high_strength = 1.0
            low_strength = 1.0

        bridge_controls = getattr(self, "_event_latent_memory_bridge_controls", {}) or {}

        previous_tail = _latent_time_slice(previous_segment_latent, "last")
        wan_samples = self._tensor_from_latent_like(wan_latent)
        wan_first = _latent_time_slice(wan_latent, "first")
        concat_latent = _find_named_tensor(wan_positive, "concat_latent_image")
        concat_first = _latent_time_slice(concat_latent, "first")
        active_requested = bool(mode == "LATENT_MEMORY_BRIDGE" and int(segment_index or 0) > 1)
        available = bool(previous_tail is not None and wan_first is not None)

        wan_alpha = _control_float(bridge_controls, "wan_alpha", 0.10, 0.0, 0.50)
        concat_alpha = _control_float(bridge_controls, "concat_alpha", 0.06, 0.0, 0.50)
        wan_max_step = _control_float(bridge_controls, "wan_max_step", 0.45, 0.0, 2.0)
        concat_max_step = _control_float(bridge_controls, "concat_max_step", 0.28, 0.0, 2.0)

        record = {
            "stage": "EventSegmentEntryLatentMemoryBridge",
            "status": "inactive_mode",
            "bridge_version": "segment_entry_latent_memory_bridge_v2_explicit_controls",
            "segment_index": int(segment_index) if segment_index is not None else None,
            "mode": mode,
            "control_mode": mode,
            "active_control_allowed": bool(active_requested),
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "available": bool(available),
            "requested_strengths": {
                "high_delta_strength": float(high_strength),
                "low_delta_strength": float(low_strength),
            },
            "bridge_controls": {
                "source": "explicit_widgets",
                "wan_alpha": float(wan_alpha),
                "concat_alpha": float(concat_alpha),
                "wan_max_step": float(wan_max_step),
                "concat_max_step": float(concat_max_step),
                "meaning": (
                    "These controls tune the latent-memory bridge only. "
                    "They no longer reuse high_delta_strength/low_delta_strength."
                ),
            },
            "bridge_params": {
                "wan_latent_alpha": float(wan_alpha),
                "wan_latent_max_step": float(wan_max_step),
                "concat_latent_alpha": float(concat_alpha),
                "concat_latent_max_step": float(concat_max_step),
                "temporal_scope": "first_latent_slice_only",
            },
            "before_metrics": {
                "previous_tail_vs_wan_first": _compare(previous_tail, wan_first),
                "previous_tail_vs_concat_first": _compare(previous_tail, concat_first),
            },
            "strategy_carrier_context": strategy_carrier_context if isinstance(strategy_carrier_context, dict) else {},
            "formula": (
                "Bounded latent-memory bridge: previous latent tail contributes a small clamped first-slice memory "
                "to the next segment Wan latent entry before high sampler. R113 keeps this as its own Strategy-return "
                "surface instead of coupling it to high/low delta scale."
            ),
        }

        if not active_requested:
            if int(segment_index or 0) > 1 and available:
                record["status"] = "report_only_available"
            elif int(segment_index or 0) <= 1:
                record["status"] = "not_applicable_first_segment"
            else:
                record["status"] = "unavailable"
            records.append(record)
            return wan_positive, wan_negative, wan_latent

        if not available:
            record["status"] = "unavailable"
            record["reason"] = "previous tail or Wan latent first slice missing"
            records.append(record)
            return wan_positive, wan_negative, wan_latent

        bridged_wan_latent = wan_latent
        bridged_wan_samples, wan_bridge = _blend_first_slice(
            wan_samples,
            previous_tail,
            alpha=wan_alpha,
            max_step=wan_max_step,
        )
        wan_applied = bool(isinstance(wan_bridge, dict) and wan_bridge.get("status") == "applied")
        if wan_applied:
            if isinstance(wan_latent, dict) and "samples" in wan_latent:
                bridged_wan_latent = dict(wan_latent)
                bridged_wan_latent["samples"] = bridged_wan_samples
            else:
                bridged_wan_latent = bridged_wan_samples

        bridged_wan_positive = wan_positive
        concat_bridge = {"status": "unavailable"}
        concat_replaced = False
        if concat_latent is not None:
            bridged_concat, concat_bridge = _blend_first_slice(
                concat_latent,
                previous_tail,
                alpha=concat_alpha,
                max_step=concat_max_step,
            )
            if isinstance(concat_bridge, dict) and concat_bridge.get("status") == "applied":
                bridged_wan_positive, concat_replaced = _replace_named_tensor(
                    wan_positive,
                    "concat_latent_image",
                    bridged_concat,
                )

        record.update({
            "status": "applied" if wan_applied or concat_replaced else "not_applied",
            "wan_latent_bridge": wan_bridge,
            "concat_latent_bridge": concat_bridge,
            "concat_latent_replaced": bool(concat_replaced),
            "after_metrics": {
                "previous_tail_vs_wan_first": _compare(previous_tail, _latent_time_slice(bridged_wan_latent, "first")),
                "previous_tail_vs_concat_first": _compare(previous_tail, _latent_time_slice(_find_named_tensor(bridged_wan_positive, "concat_latent_image"), "first")),
            },
            "policy": "bounded_first_slice_memory_return_before_high_sampler",
            "next_route": (
                "Compare against r110 neutral run. If seam motion improves without background/identity noise, "
                "promote to a narrower configurable research surface; otherwise reduce alpha or move to sampler-entry pressure."
            ),
        })
        records.append(record)
        return bridged_wan_positive, wan_negative, bridged_wan_latent

    def _run_event_horizon_segment_core(
        self,
        *,
        segment_index,
        source_image,
        previous_segment_latent=None,
        primary_model,
        secondary_model,
        clip,
        vae,
        positive_prompt,
        negative_prompt,
        active_branch_mode,
        width,
        height,
        frames,
        batch_size,
        seed,
        sampler_name,
        scheduler,
        global_steps,
        primary_cfg,
        secondary_cfg,
        primary_start_step,
        primary_end_step,
        secondary_start_step,
        secondary_end_step,
        primary_sd3_shift,
        secondary_sd3_shift,
        decode_tile_size,
        decode_overlap,
        decode_temporal_size,
        decode_temporal_overlap,
        image_upscale_method,
        image_crop,
        stage_delay_seconds,
        records,
        barrier_records,
        strategy_carrier_context=None,
    ):
        segment_label = f"cascade_{segment_index}"
        records.append({
            "stage": "SingularityCascadeSegmentBegin",
            "status": "begin",
            "segment_index": segment_index,
            "frames": int(frames),
            "formula": "source frame + segment transition = EventSingularity_segment = decoded frame batch",
        })

        segment_positive_prompt = self._cascade_prompt_for_segment(positive_prompt, segment_index, records, kind="positive")
        segment_negative_prompt = self._cascade_prompt_for_segment(negative_prompt, segment_index, records, kind="negative")
        self._record_segment_strategy_carrier(
            records,
            segment_index=segment_index,
            positive_prompt=segment_positive_prompt,
            negative_prompt=segment_negative_prompt,
            context=strategy_carrier_context,
        )
        positive = self._encode_text_with_strategy_cache(
            clip,
            segment_positive_prompt,
            records,
            label=f'{segment_label}_TextEncodePositive',
            context=strategy_carrier_context,
            polarity="positive",
        )
        negative = self._encode_text_with_strategy_cache(
            clip,
            segment_negative_prompt,
            records,
            label=f'{segment_label}_TextEncodeNegative',
            context=strategy_carrier_context,
            polarity="negative",
        )
        self._stage_delay(stage_delay_seconds, records, f"cascade_{segment_index}_after_text_encode")
        scaled_image = self._scale_image(
            source_image,
            width,
            height,
            image_upscale_method,
            image_crop,
            records,
            segment_index=segment_index,
            route_label=segment_label,
        )

        wan_positive, wan_negative, wan_latent = self._wan_image_to_video(
            positive, negative, vae, scaled_image, width, height, frames, batch_size, records,
            segment_index=segment_index,
            route_label=segment_label,
        )
        wan_positive, wan_negative, wan_latent = self._apply_segment_entry_latent_memory_bridge(
            segment_index=segment_index,
            previous_segment_latent=previous_segment_latent,
            wan_positive=wan_positive,
            wan_negative=wan_negative,
            wan_latent=wan_latent,
            records=records,
            strategy_carrier_context=strategy_carrier_context,
        )
        self._stage_delay(stage_delay_seconds, records, f"cascade_{segment_index}_after_wan_image_to_video")

        high_model = self._apply_sd3_shift(primary_model, primary_sd3_shift, f"{segment_label}_high", records)
        high_window = EventSamplerWindow(
            branch_name=f"{segment_label}_high",
            branch_role="cascade_segment_high_motion_structure",
            seed=int(seed) + int(segment_index) - 1,
            steps=int(global_steps),
            cfg=float(primary_cfg),
            sampler_name=str(sampler_name),
            scheduler=str(scheduler),
            start_at_step=int(primary_start_step),
            end_at_step=int(primary_end_step),
            add_noise="enable",
            return_with_leftover_noise="enable",
            sd3_shift=float(primary_sd3_shift),
        )
        print(f"[RAW5] high sampler launching seg={segment_index} cfg={primary_cfg} range={primary_start_step}-{primary_end_step}")
        latent_after_high, raw_delta_norm, delta_high = self._event_sample_window(
            high_model, wan_positive, wan_negative, wan_latent, high_window, records
        )
        print(f"[RAW5] high sampler done seg={segment_index} raw_delta_norm={raw_delta_norm}")
        self._stage_delay(stage_delay_seconds, records, f"cascade_{segment_index}_after_high_sampler")
        if delta_high is None:
            records.append({
                "stage": f"EventMathDualBranchDeltaCoupling_{segment_label}",
                "status": "high_delta_unavailable",
                "segment_index": int(segment_index),
            })
        delta_low = None

        final_latent = latent_after_high
        print(f"[RAW5] branch decision seg={segment_index} mode={active_branch_mode}")
        if active_branch_mode == "DUAL_HIGH_LOW":
            latent_before_low = latent_after_high
            print(f"[RAW5] entering DUAL_HIGH_LOW path seg={segment_index}")

            low_model = self._apply_sd3_shift(secondary_model or primary_model, secondary_sd3_shift, f"{segment_label}_low", records)
            
            low_delta_strength = getattr(self, "_event_delta_strengths", {}).get("low", 1.0)
            print(f"[RAW5] recording cfg policy seg={segment_index} secondary_cfg={secondary_cfg}, raw_delta_norm={raw_delta_norm}, strength={low_delta_strength}")
            modified_secondary_cfg = self._bounded_strategy_cfg(
                secondary_cfg,
                raw_delta_norm,
                low_delta_strength,
                records,
                f"cascade_{segment_index}_low_cfg",
            )

            print(f"[RAW5] FORMULA_RESULT seg={segment_index} cfg_policy_result={modified_secondary_cfg}")
            print(f"[RAW5] DUAL branch active, preparing low (1-4 range) seg={segment_index}")
                
            low_window = EventSamplerWindow(
                branch_name=f"{segment_label}_low",
                branch_role="cascade_segment_low_detail_refinement",
                seed=int(seed) + int(segment_index) - 1,
                steps=int(global_steps),
                cfg=float(modified_secondary_cfg),
                sampler_name=str(sampler_name),
                scheduler=str(scheduler),
                start_at_step=int(secondary_start_step),
                end_at_step=int(secondary_end_step),
                add_noise="disable",
                return_with_leftover_noise="disable",
                sd3_shift=float(secondary_sd3_shift),
            )
            print(f"[RAW5] low launched seg={segment_index} cfg={modified_secondary_cfg}")
            latent_after_low, _, delta_low = self._event_sample_window(
                low_model, wan_positive, wan_negative, latent_before_low, low_window, records
            )
            self._stage_delay(stage_delay_seconds, records, f"cascade_{segment_index}_after_low_sampler")
            print(f"[RAW5] low done seg={segment_index} cfg={modified_secondary_cfg}")
            if delta_low is None:
                records.append({
                    "stage": f"EventMathDualBranchDeltaCoupling_{segment_label}",
                    "status": "low_delta_unavailable",
                    "segment_index": int(segment_index),
                })
            final_latent = latent_after_low

        # _dual_branch_delta_coupling_math excised (physical cut #21): removed smart post-hoc alignment scoring on raw branch deltas.
        # Let the raw interaction stand without an interpretive comfort layer.

        frames_out = self._decode_tiled(
            vae, final_latent,
            decode_tile_size, decode_overlap, decode_temporal_size, decode_temporal_overlap,
            records,
            segment_index=segment_index,
            route_label=segment_label,
        )
        self._math_tensor_summary(frames_out, records, f"EventMath_cascade_{segment_index}_decoded_frames", strict=False)
        self._frame_motion_math(frames_out, records, f"EventMath_cascade_{segment_index}_frame_motion")
        self._record_segment_entry_strategy_return_probe(
            records,
            segment_index=segment_index,
            source_image=source_image,
            scaled_image=scaled_image,
            previous_segment_latent=previous_segment_latent,
            wan_positive=wan_positive,
            wan_negative=wan_negative,
            wan_latent=wan_latent,
            latent_after_high=latent_after_high,
            latent_after_low=final_latent,
            frames_out=frames_out,
            strategy_carrier_context=strategy_carrier_context,
        )
        self._stage_delay(stage_delay_seconds, records, f"cascade_{segment_index}_after_decode")
        last_frame = self._last_frame_image(frames_out, width, height)
        records.append({
            "stage": "SingularityCascadeSegmentEnd",
            "status": "ok",
            "segment_index": segment_index,
            "frames": int(frames),
            "last_frame_for_next_segment": True,
        })
        return frames_out, last_frame, final_latent

    def run(
        self,
        primary_model,
        clip,
        vae,
        source_image_file,
        positive_prompt,
        negative_prompt,
        event_strategy,
        generation_target,
        terminal_mode,
        enable_continuation_outputs,
        execution_mode,
        branch_mode,
        cascade_count,
        cascade_mode,
        frames_per_cascade,
        width,
        height,
        frames,
        batch_size,
        fps,
        seed,
        sampler_name,
        scheduler,
        global_steps,
        primary_cfg,
        secondary_cfg,
        primary_start_step,
        primary_end_step,
        secondary_start_step,
        secondary_end_step,
        primary_sd3_shift,
        secondary_sd3_shift,
        decode_tile_size,
        decode_overlap,
        decode_temporal_size,
        decode_temporal_overlap,
        image_upscale_method,
        image_crop,
        stage_delay_seconds,
        save_video,
        video_format,
        force_vhs_video_combine,
        save_frames,
        save_report,
        output_target,
        save_output_image,
        save_prefix,
        output_folder_mode,
        output_folder,
        custom_output_folder,
        report_detail,
        pause_after_cascade_1=False,
        pause_after_cascade_2=False,
        pause_after_cascade_3=False,
        pause_after_cascade_4=False,
        secondary_model=None,
        image=None,
        mask=None,
        use_formula_recommendation=False,
        prompt_transcode_mode="REPORT_ONLY",
        auto_calibration_mode="OFF",
        selected_tail_index=-1,
        workflow_prompt=None,
        workflow_extra_pnginfo=None,
    ):
        run_id = now_run_id(prefix="Singularity")
        pause_node_id = _singularity_pause_key(
            getattr(self, "_singularity_node_id", None)
            or getattr(self, "_event_horizon_node_id", None)
            or run_id
        )
        self._event_strategy_coupling = {"low_strength_multiplier": 1.0}

        branch_barrier_records = []
        execution_records = []
        cascade_execution_plan = None
        runtime_aliases = self._event_runtime_aliases()
        execution_records.append({
            "stage": "SingularityRuntimeVersion",
            "status": "loaded",
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "math_report_enabled": True,
            "expected_math_stages": [
                "EventMath_high_latent_before",
                "EventMath_high_latent_after",
                "EventMath_low_latent_before",
                "EventMath_low_latent_after",
                "EventMathStrategyCarrierCoupling",
                "EventMathDualBranchDeltaCoupling",
                "EventMath_decoded_frame_motion",
                "EventUniversalMath_EventTextEncodePositive",
                "EventUniversalMath_EventTextEncodeNegative",
                "EventUniversalMath_EventImageScaleStart",
                "EventUniversalMath_EventWanImageToVideoSeed",
                "EventUniversalMath_EventSamplerHigh",
                "EventUniversalMath_EventCleanupBetweenSamplers",
                "EventUniversalMath_EventSamplerLow",
                "EventUniversalMath_EventVAEDecodeTiled",
                "EventUniversalMath_EventVideoCombine",
                "EventUniversalMath_EventModelShift_high",
                "EventUniversalMath_EventModelShift_low",
                "EventUniversalMath_EventCleanupBetweenSamplers",
                "EventUniversalMath_EventVideoSaveBegin",
                "EventUniversalBoundary_EventCascadeBoundary",
                "EventCoreBodyInit",
                "EventOneNodePolicy",
                "EventCoreBodyConsistencyAudit",
                "EventCoreBodyStageOrderAudit",
                "EventCoreBodyCompletionGate",
                "EventCoreBodySummary",
                "EventCoreBodyReportCard",
                "EventCoreBodyFinalize",
            ],
        })
        execution_records.append({
            "stage": "EventWorkflowBinding",
            "status": "recorded",
            "runtime_class": self.__class__.__name__,
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "alias_count": len(runtime_aliases),
            "aliases": runtime_aliases,
            "compatibility": {
                "Singularity": bool("Singularity" in runtime_aliases),
            },
            "formula": "The public workflow node alias resolves to one internal Event Core Body runtime implementation.",
        })
        self._record_workflow_graph_route_diagnostics(
            execution_records,
            workflow_prompt=workflow_prompt,
            workflow_extra_pnginfo=workflow_extra_pnginfo,
            unique_id=getattr(self, "_singularity_node_id", None),
        )
        input_normalization = getattr(self, "_event_input_normalization", None)
        if isinstance(input_normalization, dict):
            adjustments = input_normalization.get("adjustments", [])
            reason_histogram = {}
            for adj in adjustments:
                if not isinstance(adj, dict):
                    continue
                reason = str(adj.get("reason", "unknown") or "unknown")
                reason_histogram[reason] = int(reason_histogram.get(reason, 0)) + 1
            execution_records.append({
                "stage": "EventInputNormalization",
                "status": "recorded",
                "normalized_values": input_normalization.get("normalized", {}),
                "adjustment_count": len(adjustments),
                "adjustment_reason_histogram": reason_histogram,
                "normalized_signature": input_normalization.get("normalized_signature", ""),
                "formula": "input configuration is normalized before stage execution to preserve deterministic route semantics",
            })
            if adjustments:
                execution_records.append({
                    "stage": "EventInputNormalizationAdjustments",
                    "status": "recorded",
                    "adjustments": adjustments,
                })
        auto_calibration_state = getattr(self, "_event_auto_calibration_state", None)
        if isinstance(auto_calibration_state, dict):
            execution_records.append({
                "stage": "EventAutoCalibrationPrepare",
                "status": auto_calibration_state.get("status", "unknown"),
                "mode": str(auto_calibration_state.get("mode", auto_calibration_mode) or "OFF"),
                "scene_key": auto_calibration_state.get("scene_key", ""),
                "cache_path": auto_calibration_state.get("cache_path", ""),
                "cache_entry_found": bool(auto_calibration_state.get("cache_entry_found", False)),
                "requested": auto_calibration_state.get("requested", {}),
                "applied": auto_calibration_state.get("applied", {}),
                "background_anchor_preservation_control": getattr(self, "_event_background_anchor_preservation_control", {}),
                "formula": auto_calibration_state.get("formula", "Auto calibration observes one run and can apply a cached scene-local recommendation on the next run."),
            })
        self._event_control_warning(
            execution_records,
            getattr(self, "_event_math_control_mode", "OBSERVE_ONLY"),
            getattr(self, "_event_delta_strengths", {}).get("high", 1.0),
            getattr(self, "_event_delta_strengths", {}).get("low", 1.0),
        )
        strategy_control_plan = self._event_strategy_control_surface_plan(execution_records)
        execution_records.append({
            "stage": "EventMathControlSummary",
            "status": "recorded",
            "math_control_mode": getattr(self, "_event_math_control_mode", "OBSERVE_ONLY"),
            "high_delta_strength_requested": getattr(self, "_event_delta_strengths", {}).get("high", 1.0),
            "low_delta_strength_requested": getattr(self, "_event_delta_strengths", {}).get("low", 1.0),
            "strategy_control_surface_version": strategy_control_plan.get("version", ""),
            "strategy_control_surface_status": strategy_control_plan.get("status", ""),
            "strategy_control_surface_policy": strategy_control_plan.get("policy", ""),
            "strategy_control_surface_active_allowed": strategy_control_plan.get("active_control_allowed", False),
            "strategy_control_surface_branch_policies": strategy_control_plan.get("branch_policies", {}),
            "sampler_trace_mode": getattr(self, "_event_sampler_trace", {}).get("mode", "OFF"),
            "sampler_trace_max_steps": getattr(self, "_event_sampler_trace", {}).get("max_steps", 64),
            "active_generation_math_path": (
                strategy_control_plan.get("active_generation_math_path", "observe_only")
            ),
            "strategy_carrier_coupling": "enabled_for_low_branch_without_step_replacement",
            "precision_step": 0.0001,
            "precision_round": 0.0001,
        })
        execution_records.append({
            "stage": "EventUniversalPipelineMap",
            "status": "recorded",
            "formula": "Every internal stage is read as NodeInputState + NodeObservedBehavior = NodeSState = NodeOutputState",
            "route": [
                "EventTextEncodePositive",
                "EventTextEncodeNegative",
                "EventImageScaleStart",
                "EventWanImageToVideoSeed",
                "EventModelShiftHigh",
                "EventSamplerHigh",
                "EventCleanupBetweenSamplers",
                "EventModelShiftLow",
                "EventSamplerLow",
                "EventVAEDecodeTiled",
                "SingularityCascadeBoundary",
                "EventVideoSaveBegin",
                "EventVideoCombine",
                "EventSaveReport",
                "EventCleanupAfterGeneration",
            ],
            "status_note": "r44 keeps the Event Core Body inside one Singularity node; EVENT_PACKET/S-Wire are internal runtime body, not a manual visual graph",
        })
        execution_records.append({
            "stage": "EventOneNodePolicy",
            "status": "active",
            "external_visual_node": "Singularity",
            "manual_event_graph_required": False,
            "internal_body": [
                "EventPacket",
                "FormulaReader",
                "RoleResolver",
                "RouteMemory",
                "SState",
                "UniversalStageMath",
                "UniversalBoundaryMath",
                "ReportBuilder",
            ],
            "reason": "Avoids manual ComfyUI workflow errors, wire-order drift, lost route memory, and mismatched Event logic between hidden stages.",
        })

        prompt_key_source = (
            str(positive_prompt or "") + "\n---NEGATIVE---\n" + str(negative_prompt or "") +
            f"\nclip={type(clip).__name__}:{id(clip)}"
        )
        encoder_cache_key = hashlib.sha256(prompt_key_source.encode("utf-8", errors="ignore")).hexdigest()[:24]
        model_route_builder_policy = {
            "current_mode": "external_model_route_observed",
            "current_external_route": [
                "UNET loader",
                "LoRA loader/matrix",
                "Torch compile guard",
                "ModelSamplingSD3 shift",
                "Singularity sampler core",
            ],
            "future_internal_route": [
                "internal UNET/CLIP/VAE loader",
                "internal LoRA matrix switch",
                "internal compile guard/adapter",
                "internal SD3 route builder",
                "Singularity sampler core",
            ],
            "integration_options": {
                "A_external_minimal": "public-safe route; external loader/LoRA/compile nodes feed Singularity",
                "B_internal_builder": "Singularity owns model names, LoRA matrix and compile guard but keeps external MODEL sockets available",
                "C_full_one_node_body": "Singularity loads from model selection to video save; highest observability, highest compatibility risk",
            },
            "order_law": "Torch compile belongs after LoRA application and before ModelSamplingSD3/sampler use.",
            "formula_role": "MODEL Operator route is part of Strategy(t), not a separate convenience chain.",
            "side_effect_policy": "Internalization must preserve fixed-seed output or report the route as controlled intervention.",
        }
        execution_records.append({
            "stage": "EventModelRouteBuilderPolicy",
            "status": "observer_policy_active",
            "formula": "MODEL + LoRA + compile + SD3 shift form the Operator side of Strategy(t); order is observable and must not be hidden.",
            **model_route_builder_policy,
        })
        self._record_external_operator_route_diagnostics(
            execution_records,
            primary_model=primary_model,
            secondary_model=secondary_model,
            clip=clip,
            vae=vae,
            global_steps=global_steps,
            primary_start_step=primary_start_step,
            primary_end_step=primary_end_step,
            secondary_start_step=secondary_start_step,
            secondary_end_step=secondary_end_step,
            primary_sd3_shift=primary_sd3_shift,
            secondary_sd3_shift=secondary_sd3_shift,
        )
        execution_records.append({
            "stage": "EventRuntimeLayerProbes",
            "status": "observer_only",
            "runtime_monitor": "active",
            "compile_guard": "observed_as_external_or_future_internal_adapter",
            "encoder_cache": "key_computed_cache_disabled_until_equivalence_proof",
            "branch_barrier": "smart branch barrier records preserved StrategyCarrier and released memory actions per phase",
            "lora_matrix_switch": "external_model_clip_route_observed; future internal matrix must precede compile",
            "prompt_operator_panel": "prompt fields observed as StrategyCandidate carriers; no prompt rewrite",
            "cascade_prompt_schedule": "active via markers only: ### Cascade 1 / [Cascade 1] / ::Cascade 1::",
            "test_runner": "not active inside generation node; report data is structured for external runner",
            "universal_input_normalization": "readers and RoleResolver remain internal; fixed Wan interface still current",
            "encoder_cache_key_preview": encoder_cache_key,
            "formula": "Forgotten runtime-layer ideas are present as internal observer records before active intervention.",
        })

        saved_video_path = ""
        saved_report_path = ""
        video_ui_payload = {}
        generated_frames = None
        generated_latent = None
        source_preview = None
        result_preview = None
        result_status = "NONE"
        failure_reason = ""
        ui_images = []  # safety: always defined, even if early failure before the try block

        # r14d compatibility guard: old saved workflows or stale node widgets
        # must not crash if output_target is missing during execution.
        if "output_target" not in locals() or output_target is None:
            output_target = "USER_D_AI_NSFW"

        # Internal image picker: if external IMAGE socket is not connected, use source_image_file.
        uploaded_image = None
        if image is None:
            uploaded_image = self._load_image_from_upload(source_image_file, execution_records)
            if uploaded_image is not None:
                image = uploaded_image
        source_preview = self._representative_preview_frame(image, width, height, mode="first")

        requested_video = generation_target in ("VIDEO", "AUTO")
        requested_image = generation_target in ("IMAGE", "AUTO")

        if branch_mode == "SINGLE":
            active_branch_mode = "SINGLE"
        else:
            active_branch_mode = "DUAL_HIGH_LOW"  # raw: no fallback, let observed behavior emerge if secondary missing

        mode_to_count = {
            "SOLO_1": 1,
            "CASCADE_2": 2,
            "CASCADE_3": 3,
            "CASCADE_4": 4,
            "CASCADE_5": 5,
        }
        mode_count = int(mode_to_count.get(str(cascade_mode), int(cascade_count or 1)))
        numeric_count = int(cascade_count or mode_count or 1)
        requested_cascade_count = max(mode_count, numeric_count)
        requested_cascade_count = max(1, min(5, requested_cascade_count))
        frames = int(frames_per_cascade or frames)
        legacy_pause_flags = {
            1: bool(pause_after_cascade_1),
            2: bool(pause_after_cascade_2),
            3: bool(pause_after_cascade_3),
            4: bool(pause_after_cascade_4),
        }
        pause_after_segments = [
            int(segment_index)
            for segment_index in range(1, int(requested_cascade_count))
            if legacy_pause_flags.get(segment_index, False)
        ]
        ignored_pause_after_segments = [
            int(segment_index)
            for segment_index, enabled in legacy_pause_flags.items()
            if enabled and segment_index >= int(requested_cascade_count)
        ]
        expected_output_frames = (
            int(frames)
            if int(requested_cascade_count) <= 1
            else int(frames) + (int(requested_cascade_count) - 1) * max(0, int(frames) - 1)
        )
        cascade_execution_plan = {
            "policy_version": "cascade_plan_v1_legacy_flags",
            "policy": "LEGACY_FLAGS",
            "requested_segments": int(requested_cascade_count),
            "final_segment_index": int(requested_cascade_count),
            "frames_per_cascade": int(frames),
            "expected_output_frames_if_no_trims": int(expected_output_frames),
            "pause_after_segments": pause_after_segments,
            "pause_count": len(pause_after_segments),
            "legacy_pause_flags": {str(k): bool(v) for k, v in legacy_pause_flags.items()},
            "ignored_pause_after_segments": ignored_pause_after_segments,
            "supports_dynamic_n_cascade": False,
            "future_policy_note": "Current public route is fixed to five cascades; future route can replace legacy flags with N-segment pause policies.",
            "formula": "CascadePlan defines StrategyCarrier boundaries before execution; Gate verifies the actual route against this plan.",
        }

        packet = make_event_packet(metadata={
            "created_by": "SingularityCascade",
            "version": EVENT_HORIZON_RUNTIME_VERSION,
            "run_id": run_id,
            "node_role": "terminal_event_horizon_cascade",
            "cascade_execution_plan": cascade_execution_plan,
        })

        packet = self._event_core_body_init(packet, execution_records, run_id, route_name="wan_terminal_one_node")
        execution_records.append({
            "stage": "SingularityCascadePlan",
            "status": "recorded",
            **cascade_execution_plan,
        })
        prompt_strategy_packet = build_prompt_strategy_packet(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            source_image_file=source_image_file,
            cascade_execution_plan=cascade_execution_plan,
            math_controls={
                "math_control_mode": getattr(self, "_event_math_control_mode", "OBSERVE_ONLY"),
                "high_delta_strength": getattr(self, "_event_delta_strengths", {}).get("high", 1.0),
                "low_delta_strength": getattr(self, "_event_delta_strengths", {}).get("low", 1.0),
                "latent_memory_bridge": getattr(self, "_event_latent_memory_bridge_controls", {}),
                "sampler_trace_mode": getattr(self, "_event_sampler_trace", {}).get("mode", "OFF"),
                "sampler_trace_max_steps": getattr(self, "_event_sampler_trace", {}).get("max_steps", 64),
                "prompt_transcode_mode": prompt_transcode_mode,
            },
        )
        packet.setdefault("metadata", {})["prompt_strategy_packet"] = prompt_strategy_packet
        execution_records.append(prompt_strategy_packet)
        prompt_transcode_mode_aliases = {
            "0": "REPORT_ONLY",
            "1": "TRANSFORM_PROMPT",
            "2": "TRANSFORM_PROMPT",
            "REPORT_ONLY": "REPORT_ONLY",
            "TRANSFORM_PROMPT": "TRANSFORM_PROMPT",
            "TRANSFORM_STRUCTURED_PROMPT": "TRANSFORM_PROMPT",
            "APPEND_TRANSCODE": "TRANSFORM_PROMPT",
            "APPEND_STRUCTURED_TRANSCODE": "TRANSFORM_PROMPT",
        }
        prompt_transcode_mode_raw = str(prompt_transcode_mode or "REPORT_ONLY").strip()
        prompt_transcode_mode_n = prompt_transcode_mode_aliases.get(
            prompt_transcode_mode_raw.upper(),
            "REPORT_ONLY",
        )
        transcode = prompt_strategy_packet.get("model_language_transcode", {}) if isinstance(prompt_strategy_packet, dict) else {}
        prompt_idempotence = prompt_strategy_packet.get("prompt_idempotence", {}) if isinstance(prompt_strategy_packet, dict) else {}
        object_topology_map = prompt_strategy_packet.get("object_topology_map", {}) if isinstance(prompt_strategy_packet, dict) else {}
        object_relation_ontology = prompt_strategy_packet.get("object_relation_ontology", {}) if isinstance(prompt_strategy_packet, dict) else {}
        original_positive_text = str(positive_prompt or "")
        original_negative_text = str(negative_prompt or "")
        original_positive_signature = hashlib.sha256(original_positive_text.encode("utf-8", errors="ignore")).hexdigest()[:16]
        original_negative_signature = hashlib.sha256(original_negative_text.encode("utf-8", errors="ignore")).hexdigest()[:16]

        def sanitize_prompt_layer(base_text, clean_text, idempotence_action):
            base_text = str(base_text or "")
            clean_text = str(clean_text or "").strip()
            if str(idempotence_action or "") != "stripped_generated_strategy_tail":
                return base_text, False
            if not clean_text:
                return base_text, False
            if clean_text == base_text.strip():
                return base_text, False
            return clean_text, True

        positive_prompt_transformed = False
        negative_prompt_transformed = False
        positive_prompt_sanitized = False
        negative_prompt_sanitized = False
        if isinstance(prompt_idempotence, dict):
            positive_prompt, positive_prompt_sanitized = sanitize_prompt_layer(
                positive_prompt,
                transcode.get("transformed_positive_prompt", "") if isinstance(transcode, dict) else "",
                prompt_idempotence.get("idempotence_action", ""),
            )

        active_positive_signature = hashlib.sha256(str(positive_prompt or "").encode("utf-8", errors="ignore")).hexdigest()[:16]
        active_negative_signature = hashlib.sha256(str(negative_prompt or "").encode("utf-8", errors="ignore")).hexdigest()[:16]

        def _prompt_normalized_text_signature(text):
            normalized = " ".join(str(text or "").split())
            return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()[:16]

        active_positive_normalized_signature = _prompt_normalized_text_signature(positive_prompt)
        active_negative_normalized_signature = _prompt_normalized_text_signature(negative_prompt)

        semantic_density_context_map = (
            prompt_strategy_packet.get("semantic_density_context_map", {})
            if isinstance(prompt_strategy_packet, dict)
            else {}
        )
        prompt_purity_lock_record = {
            "stage": "EventPromptPurityLock",
            "status": "locked",
            "raw_prompt_transcode_mode": prompt_transcode_mode_raw,
            "prompt_transcode_mode": prompt_transcode_mode_n,
            "formula": "Prompt words stay as the user's clean StrategyCandidate; formula math is sorted in semantic density/context space instead of being injected into CLIP text.",
            "prompt_purity_lock": True,
            "clip_positive_uses_raw_prompt": not positive_prompt_sanitized,
            "clip_positive_uses_sanitized_raw_prompt": bool(positive_prompt_sanitized),
            "clip_negative_uses_raw_prompt": True,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "positive_prompt_transformed": False,
            "negative_prompt_transformed": False,
            "positive_prompt_sanitized": bool(positive_prompt_sanitized),
            "negative_prompt_sanitized": bool(negative_prompt_sanitized),
            "semantic_density_context_map": semantic_density_context_map if isinstance(semantic_density_context_map, dict) else {},
            "model_facing_prompt_policy": "raw_user_prompt_or_sanitized_raw_prompt_only",
            "control_mode": "SEMANTIC_DENSITY_MAP_ONLY" if prompt_transcode_mode_n != "REPORT_ONLY" else "REPORT_ONLY",
            "active_control_allowed": False,
        }
        packet.setdefault("metadata", {})["prompt_purity_lock"] = prompt_purity_lock_record
        execution_records.append(prompt_purity_lock_record)

        transcode_apply_record = {
            "stage": "EventPromptStrategyTranscodeApply",
            "status": "semantic_map_only" if prompt_transcode_mode_n != "REPORT_ONLY" else "report_only",
            "raw_prompt_transcode_mode": prompt_transcode_mode_raw,
            "prompt_transcode_mode": prompt_transcode_mode_n,
            "formula": "Prompt transcode is now a semantic density/context map. It may sort Strategy meaning in report/control space, but it must not become extra prompt text.",
            "transcode_policy": "semantic_density_map_only_no_text_injection",
            "transformation_policy": "prompt_purity_lock_semantic_map_only",
            "prompt_purity_lock": True,
            "prompt_text_injection_allowed": False,
            "semantic_math_in_prompt_allowed": False,
            "prompt_idempotence_policy": transcode.get("idempotence_policy", "") if isinstance(transcode, dict) else "",
            "existing_strategy_transform_detected": bool(prompt_idempotence.get("existing_strategy_transform_detected", False)) if isinstance(prompt_idempotence, dict) else False,
            "prompt_idempotence_action": prompt_idempotence.get("idempotence_action", "") if isinstance(prompt_idempotence, dict) else "",
            "detected_strategy_transform_marker": prompt_idempotence.get("detected_marker", "") if isinstance(prompt_idempotence, dict) else "",
            "strategy_transform_stripped_character_count": prompt_idempotence.get("stripped_character_count", 0) if isinstance(prompt_idempotence, dict) else 0,
            "sanitized_positive_signature": prompt_idempotence.get("sanitized_positive_signature", "") if isinstance(prompt_idempotence, dict) else "",
            "free_math_policy": "Math, meaning, semantics, logic, and Strategy remain linked and free; Singularity does not define a fixed physics solver or convert the formula into prompt prose here.",
            "object_topology_policy": "If an object carrier is detected, it is mapped as topology evidence and density pressure, not appended as instructions.",
            "object_relation_ontology_policy": "If object/contact carriers are detected, their roles remain report/control-space carriers unless a future bounded tensor/weight route is explicitly enabled.",
            "negative_transform_policy": transcode.get("negative_transform_policy", "") if isinstance(transcode, dict) else "",
            "object_topology_status": object_topology_map.get("status") if isinstance(object_topology_map, dict) else "",
            "object_relation_ontology_status": object_relation_ontology.get("status") if isinstance(object_relation_ontology, dict) else "",
            "object_relation_strategy_point": object_relation_ontology.get("strategy_point") if isinstance(object_relation_ontology, dict) else "",
            "object_relation_sentence_count": len(object_relation_ontology.get("positive_strategy_sentences", []) or []) if isinstance(object_relation_ontology, dict) else 0,
            "rigid_object_count": object_topology_map.get("rigid_object_count", 0) if isinstance(object_topology_map, dict) else 0,
            "rigidity_lock_recommended": bool(object_topology_map.get("rigidity_lock_recommended", False)) if isinstance(object_topology_map, dict) else False,
            "contact_depth_axis_recommended": bool(object_topology_map.get("contact_depth_axis_recommended", False)) if isinstance(object_topology_map, dict) else False,
            "contact_depth_axis_hint": object_topology_map.get("contact_depth_axis_hint", "") if isinstance(object_topology_map, dict) else "",
            "rigidity_transform_applied": bool(
                False
            ),
            "object_relation_ontology_applied": bool(
                False
            ),
            "original_prompt_meaning_preserved": True,
            "original_prompt_preserved": not positive_prompt_sanitized,
            "positive_prompt_transformed": bool(positive_prompt_transformed),
            "negative_prompt_transformed": bool(negative_prompt_transformed),
            "positive_prompt_sanitized": bool(positive_prompt_sanitized),
            "negative_prompt_sanitized": bool(negative_prompt_sanitized),
            "positive_transcode_added": False,
            "negative_transcode_added": False,
            "original_positive_signature": original_positive_signature,
            "active_positive_signature": active_positive_signature,
            "original_negative_signature": original_negative_signature,
            "active_negative_signature": active_negative_signature,
            "active_positive_normalized_signature": active_positive_normalized_signature,
            "active_negative_normalized_signature": active_negative_normalized_signature,
            "active_control_allowed": False,
            "control_mode": "SEMANTIC_DENSITY_MAP_ONLY" if prompt_transcode_mode_n != "REPORT_ONLY" else "REPORT_ONLY",
            "semantic_density_context_map": semantic_density_context_map if isinstance(semantic_density_context_map, dict) else {},
            "model_freedom_policy": "The model still receives one clean text-conditioning route; Singularity maps the Strategy outside prompt prose.",
        }
        packet.setdefault("metadata", {})["prompt_strategy_transcode_apply"] = transcode_apply_record
        execution_records.append(transcode_apply_record)
        segment_strategy_carrier_context = {
            "prompt_source": "launch_time_sanitized_raw" if positive_prompt_sanitized else "launch_time_raw",
            "prompt_transcode_mode": prompt_transcode_mode_n,
            "launch_raw_positive_signature": original_positive_signature,
            "launch_raw_negative_signature": original_negative_signature,
            "launch_raw_positive_normalized_signature": _prompt_normalized_text_signature(original_positive_text),
            "launch_raw_negative_normalized_signature": _prompt_normalized_text_signature(original_negative_text),
            "launch_active_positive_signature": active_positive_signature,
            "launch_active_negative_signature": active_negative_signature,
            "launch_active_positive_normalized_signature": active_positive_normalized_signature,
            "launch_active_negative_normalized_signature": active_negative_normalized_signature,
            "current_active_positive_signature": active_positive_signature,
            "current_active_negative_signature": active_negative_signature,
            "current_active_positive_normalized_signature": active_positive_normalized_signature,
            "current_active_negative_normalized_signature": active_negative_normalized_signature,
            "current_positive_prompt_transformed": bool(positive_prompt_transformed),
            "current_negative_prompt_transformed": bool(negative_prompt_transformed),
            "current_positive_prompt_sanitized": bool(positive_prompt_sanitized),
            "current_negative_prompt_sanitized": bool(negative_prompt_sanitized),
            "launch_prompt_transcode_mode": prompt_transcode_mode_n,
            "last_runtime_prompt_update_applies_to_segment": None,
        }

        def _prompt_text_signature(text):
            return hashlib.sha256(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:16]

        def _normalize_prompt_mode_for_runtime(value):
            raw = str(value or prompt_transcode_mode_n or "REPORT_ONLY").strip()
            return prompt_transcode_mode_aliases.get(raw.upper(), prompt_transcode_mode_n)

        def _transform_prompt_pair_for_runtime(raw_positive, raw_negative, mode_value, applies_to_segment, source):
            mode_n = _normalize_prompt_mode_for_runtime(mode_value)
            raw_positive = str(raw_positive or "")
            raw_negative = str(raw_negative or "")
            local_plan = dict(cascade_execution_plan)
            local_plan["runtime_prompt_applies_to_segment"] = int(applies_to_segment)
            local_plan["runtime_prompt_source"] = str(source or "node_widgets_at_continue_click")
            local_packet = build_prompt_strategy_packet(
                positive_prompt=raw_positive,
                negative_prompt=raw_negative,
                source_image_file=source_image_file,
                cascade_execution_plan=local_plan,
                math_controls={
                    "math_control_mode": getattr(self, "_event_math_control_mode", "OBSERVE_ONLY"),
                    "high_delta_strength": getattr(self, "_event_delta_strengths", {}).get("high", 1.0),
                    "low_delta_strength": getattr(self, "_event_delta_strengths", {}).get("low", 1.0),
                    "latent_memory_bridge": getattr(self, "_event_latent_memory_bridge_controls", {}),
                    "sampler_trace_mode": getattr(self, "_event_sampler_trace", {}).get("mode", "OFF"),
                    "sampler_trace_max_steps": getattr(self, "_event_sampler_trace", {}).get("max_steps", 64),
                    "prompt_transcode_mode": mode_n,
                },
            )
            local_transcode = local_packet.get("model_language_transcode", {}) if isinstance(local_packet, dict) else {}
            local_idempotence = local_packet.get("prompt_idempotence", {}) if isinstance(local_packet, dict) else {}

            def runtime_sanitize_layer(base_text, clean_text, idempotence_action):
                base_text = str(base_text or "")
                clean_text = str(clean_text or "").strip()
                if str(idempotence_action or "") != "stripped_generated_strategy_tail":
                    return base_text, False
                if not clean_text:
                    return base_text, False
                if clean_text == base_text.strip():
                    return base_text, False
                return clean_text, True

            active_positive = raw_positive
            active_negative = raw_negative
            positive_transformed = False
            negative_transformed = False
            positive_sanitized = False
            negative_sanitized = False
            if isinstance(local_idempotence, dict):
                active_positive, positive_sanitized = runtime_sanitize_layer(
                    raw_positive,
                    local_transcode.get("transformed_positive_prompt", "") if isinstance(local_transcode, dict) else "",
                    local_idempotence.get("idempotence_action", ""),
                )
            local_density_map = (
                local_packet.get("semantic_density_context_map", {})
                if isinstance(local_packet, dict)
                else {}
            )

            return active_positive, active_negative, {
                "prompt_transcode_mode": mode_n,
                "positive_prompt_transformed": bool(positive_transformed),
                "negative_prompt_transformed": bool(negative_transformed),
                "positive_prompt_sanitized": bool(positive_sanitized),
                "negative_prompt_sanitized": bool(negative_sanitized),
                "prompt_purity_lock": True,
                "prompt_text_injection_allowed": False,
                "semantic_math_in_prompt_allowed": False,
                "control_mode": "SEMANTIC_DENSITY_MAP_ONLY" if mode_n != "REPORT_ONLY" else "REPORT_ONLY",
                "raw_positive_signature": _prompt_text_signature(raw_positive),
                "raw_negative_signature": _prompt_text_signature(raw_negative),
                "active_positive_signature": _prompt_text_signature(active_positive),
                "active_negative_signature": _prompt_text_signature(active_negative),
                "existing_strategy_transform_detected": bool(local_idempotence.get("existing_strategy_transform_detected", False)) if isinstance(local_idempotence, dict) else False,
                "prompt_idempotence_action": local_idempotence.get("idempotence_action", "") if isinstance(local_idempotence, dict) else "",
                "strategy_transform_stripped_character_count": local_idempotence.get("stripped_character_count", 0) if isinstance(local_idempotence, dict) else 0,
                "semantic_density_context_map": local_density_map if isinstance(local_density_map, dict) else {},
                "object_relation_ontology_status": (
                    local_packet.get("object_relation_ontology", {}).get("status", "")
                    if isinstance(local_packet, dict) and isinstance(local_packet.get("object_relation_ontology", {}), dict)
                    else ""
                ),
                "object_topology_status": (
                    local_packet.get("object_topology_map", {}).get("status", "")
                    if isinstance(local_packet, dict) and isinstance(local_packet.get("object_topology_map", {}), dict)
                    else ""
                ),
                "contact_depth_axis_recommended": (
                    bool(local_packet.get("object_topology_map", {}).get("contact_depth_axis_recommended", False))
                    if isinstance(local_packet, dict) and isinstance(local_packet.get("object_topology_map", {}), dict)
                    else False
                ),
                "contact_depth_axis_hint": (
                    local_packet.get("object_topology_map", {}).get("contact_depth_axis_hint", "")
                    if isinstance(local_packet, dict) and isinstance(local_packet.get("object_topology_map", {}), dict)
                    else ""
                ),
            }

        def _apply_runtime_prompt_update(prompt_update, pause_segment_index, applies_to_segment):
            nonlocal positive_prompt, negative_prompt
            if not isinstance(prompt_update, dict) or not prompt_update:
                execution_records.append({
                    "stage": "EventCascadePromptRuntimeUpdate",
                    "status": "no_prompt_payload",
                    "pause_segment_index": int(pause_segment_index),
                    "applies_to_segment": int(applies_to_segment),
                    "formula": "No runtime prompt update was sent; the next cascade reuses the current StrategyCandidate text.",
                })
                return

            old_positive_signature = _prompt_text_signature(positive_prompt)
            old_negative_signature = _prompt_text_signature(negative_prompt)
            old_positive_normalized_signature = _prompt_normalized_text_signature(positive_prompt)
            old_negative_normalized_signature = _prompt_normalized_text_signature(negative_prompt)
            raw_positive = str(prompt_update.get("positive_prompt", ""))
            raw_negative = str(prompt_update.get("negative_prompt", ""))
            positive_prompt_present = bool(prompt_update.get("positive_prompt_present", True))
            negative_prompt_present = bool(prompt_update.get("negative_prompt_present", True))
            mode_value = prompt_update.get("prompt_transcode_mode", prompt_transcode_mode_n)
            source = prompt_update.get("source", "node_widgets_at_continue_click")
            mode_n = _normalize_prompt_mode_for_runtime(mode_value)
            raw_positive_signature = _prompt_text_signature(raw_positive)
            raw_negative_signature = _prompt_text_signature(raw_negative)
            raw_positive_normalized_signature = _prompt_normalized_text_signature(raw_positive)
            raw_negative_normalized_signature = _prompt_normalized_text_signature(raw_negative)

            def _canonical_prompt_text(text):
                return " ".join(str(text or "").split())

            def _stored_prompt_signature_matches(raw_sig, raw_norm_sig, signature_key, normalized_key):
                stored_signature = str(segment_strategy_carrier_context.get(signature_key) or "")
                stored_normalized = str(segment_strategy_carrier_context.get(normalized_key) or "")
                return bool(
                    (stored_signature and raw_sig == stored_signature)
                    or (stored_normalized and raw_norm_sig == stored_normalized)
                )

            def _stored_prompt_pair_matches(positive_signature_key, positive_normalized_key, negative_signature_key, negative_normalized_key):
                return bool(
                    _stored_prompt_signature_matches(
                        raw_positive_signature,
                        raw_positive_normalized_signature,
                        positive_signature_key,
                        positive_normalized_key,
                    )
                    and _stored_prompt_signature_matches(
                        raw_negative_signature,
                        raw_negative_normalized_signature,
                        negative_signature_key,
                        negative_normalized_key,
                    )
                )

            launch_positive_matches = (
                raw_positive_signature == str(segment_strategy_carrier_context.get("launch_raw_positive_signature") or "")
                or raw_positive_normalized_signature == str(segment_strategy_carrier_context.get("launch_raw_positive_normalized_signature") or "")
            )
            launch_negative_matches = (
                raw_negative_signature == str(segment_strategy_carrier_context.get("launch_raw_negative_signature") or "")
                or raw_negative_normalized_signature == str(segment_strategy_carrier_context.get("launch_raw_negative_normalized_signature") or "")
            )
            launch_raw_matches = bool(launch_positive_matches and launch_negative_matches)
            launch_active_matches = _stored_prompt_pair_matches(
                "launch_active_positive_signature",
                "launch_active_positive_normalized_signature",
                "launch_active_negative_signature",
                "launch_active_negative_normalized_signature",
            )
            last_runtime_raw_matches = _stored_prompt_pair_matches(
                "last_runtime_raw_positive_signature",
                "last_runtime_raw_positive_normalized_signature",
                "last_runtime_raw_negative_signature",
                "last_runtime_raw_negative_normalized_signature",
            )
            last_runtime_active_matches = _stored_prompt_pair_matches(
                "last_runtime_active_positive_signature",
                "last_runtime_active_positive_normalized_signature",
                "last_runtime_active_negative_signature",
                "last_runtime_active_negative_normalized_signature",
            )
            stored_current_active_matches = _stored_prompt_pair_matches(
                "current_active_positive_signature",
                "current_active_positive_normalized_signature",
                "current_active_negative_signature",
                "current_active_negative_normalized_signature",
            )
            current_active_matches = bool(
                (
                    raw_positive_signature == old_positive_signature
                    or raw_positive_normalized_signature == old_positive_normalized_signature
                )
                and (
                    raw_negative_signature == old_negative_signature
                    or raw_negative_normalized_signature == old_negative_normalized_signature
                )
            )
            positive_identity_matches = bool(
                launch_positive_matches
                or _stored_prompt_signature_matches(
                    raw_positive_signature,
                    raw_positive_normalized_signature,
                    "launch_active_positive_signature",
                    "launch_active_positive_normalized_signature",
                )
                or _stored_prompt_signature_matches(
                    raw_positive_signature,
                    raw_positive_normalized_signature,
                    "current_active_positive_signature",
                    "current_active_positive_normalized_signature",
                )
                or raw_positive_signature == old_positive_signature
                or raw_positive_normalized_signature == old_positive_normalized_signature
            )
            negative_identity_matches = bool(
                launch_negative_matches
                or _stored_prompt_signature_matches(
                    raw_negative_signature,
                    raw_negative_normalized_signature,
                    "launch_active_negative_signature",
                    "launch_active_negative_normalized_signature",
                )
                or _stored_prompt_signature_matches(
                    raw_negative_signature,
                    raw_negative_normalized_signature,
                    "current_active_negative_signature",
                    "current_active_negative_normalized_signature",
                )
                or raw_negative_signature == old_negative_signature
                or raw_negative_normalized_signature == old_negative_normalized_signature
            )
            raw_negative_canonical = _canonical_prompt_text(raw_negative)
            old_negative_canonical = _canonical_prompt_text(negative_prompt)
            launch_negative_canonical = _canonical_prompt_text(original_negative_text)
            negative_payload_missing = not negative_prompt_present or raw_negative_canonical == ""
            negative_payload_truncated = bool(
                raw_negative_canonical
                and len(raw_negative_canonical) >= 64
                and not negative_identity_matches
                and (
                    raw_negative_canonical in old_negative_canonical
                    or raw_negative_canonical in launch_negative_canonical
                    or old_negative_canonical in raw_negative_canonical
                    or launch_negative_canonical in raw_negative_canonical
                )
            )
            raw_positive_canonical = _canonical_prompt_text(raw_positive)
            old_positive_canonical = _canonical_prompt_text(positive_prompt)
            launch_positive_canonical = _canonical_prompt_text(original_positive_text)
            positive_payload_missing = not positive_prompt_present or raw_positive_canonical == ""
            positive_payload_truncated = bool(
                raw_positive_canonical
                and len(raw_positive_canonical) >= 64
                and not positive_identity_matches
                and (
                    raw_positive_canonical in old_positive_canonical
                    or raw_positive_canonical in launch_positive_canonical
                    or old_positive_canonical in raw_positive_canonical
                    or launch_positive_canonical in raw_positive_canonical
                )
            )
            mode_matches = mode_n == str(segment_strategy_carrier_context.get("prompt_transcode_mode") or prompt_transcode_mode_n)
            positive_payload_transforms_to_current_active = False
            positive_payload_transform_preview_signature = ""
            positive_payload_transform_preview_normalized_signature = ""
            if mode_matches and not positive_identity_matches and raw_positive.strip():
                try:
                    preview_positive, _preview_negative, _preview_summary = _transform_prompt_pair_for_runtime(
                        raw_positive,
                        raw_negative,
                        mode_value,
                        applies_to_segment,
                        source,
                    )
                    positive_payload_transform_preview_signature = _prompt_text_signature(preview_positive)
                    positive_payload_transform_preview_normalized_signature = _prompt_normalized_text_signature(preview_positive)
                    positive_payload_transforms_to_current_active = bool(
                        positive_payload_transform_preview_signature == old_positive_signature
                        or positive_payload_transform_preview_normalized_signature == old_positive_normalized_signature
                        or _stored_prompt_signature_matches(
                            positive_payload_transform_preview_signature,
                            positive_payload_transform_preview_normalized_signature,
                            "current_active_positive_signature",
                            "current_active_positive_normalized_signature",
                        )
                        or _stored_prompt_signature_matches(
                            positive_payload_transform_preview_signature,
                            positive_payload_transform_preview_normalized_signature,
                            "launch_active_positive_signature",
                            "launch_active_positive_normalized_signature",
                        )
                    )
                except Exception:
                    positive_payload_transforms_to_current_active = False
                    positive_payload_transform_preview_signature = ""
                    positive_payload_transform_preview_normalized_signature = ""
            positive_strategy_identity_matches = bool(
                positive_identity_matches
                or positive_payload_transforms_to_current_active
                or (
                    mode_matches
                    and not positive_identity_matches
                    and (positive_payload_missing or positive_payload_truncated)
                )
            )
            negative_payload_reuse_previous_active = bool(
                mode_n == str(segment_strategy_carrier_context.get("prompt_transcode_mode") or prompt_transcode_mode_n)
                and positive_strategy_identity_matches
                and not negative_identity_matches
                and (negative_payload_missing or negative_payload_truncated)
            )
            positive_payload_reuse_previous_active = bool(
                mode_n == str(segment_strategy_carrier_context.get("prompt_transcode_mode") or prompt_transcode_mode_n)
                and not positive_identity_matches
                and (positive_payload_missing or positive_payload_truncated)
            )
            negative_strategy_identity_matches = bool(
                negative_identity_matches
                or negative_payload_reuse_previous_active
            )
            same_prompt_match_basis = ""
            if mode_matches:
                for candidate_basis, candidate_matches in (
                    ("launch_raw", launch_raw_matches),
                    ("last_runtime_raw", last_runtime_raw_matches),
                    ("current_active", current_active_matches),
                    ("stored_current_active", stored_current_active_matches),
                    ("launch_active", launch_active_matches),
                    ("last_runtime_active", last_runtime_active_matches),
                ):
                    if candidate_matches:
                        same_prompt_match_basis = candidate_basis
                        break
                if (
                    not same_prompt_match_basis
                    and positive_payload_reuse_previous_active
                    and negative_strategy_identity_matches
                ):
                    if positive_payload_missing:
                        same_prompt_match_basis = "positive_payload_missing_reuse"
                    elif negative_payload_truncated:
                        same_prompt_match_basis = "prompt_payload_truncated_reuse"
                    else:
                        same_prompt_match_basis = "positive_payload_truncated_reuse"
                if not same_prompt_match_basis and negative_payload_reuse_previous_active:
                    same_prompt_match_basis = (
                        "negative_payload_missing_reuse"
                        if negative_payload_missing
                        else "negative_payload_truncated_reuse"
                    )
            same_prompt = bool(same_prompt_match_basis)

            if same_prompt:
                transform_summary = {
                    "prompt_transcode_mode": mode_n,
                    "positive_prompt_transformed": bool(segment_strategy_carrier_context.get("current_positive_prompt_transformed", positive_prompt_transformed)),
                    "negative_prompt_transformed": bool(segment_strategy_carrier_context.get("current_negative_prompt_transformed", negative_prompt_transformed)),
                    "positive_prompt_sanitized": bool(segment_strategy_carrier_context.get("current_positive_prompt_sanitized", positive_prompt_sanitized)),
                    "negative_prompt_sanitized": bool(segment_strategy_carrier_context.get("current_negative_prompt_sanitized", negative_prompt_sanitized)),
                    "prompt_purity_lock": True,
                    "prompt_text_injection_allowed": False,
                    "semantic_math_in_prompt_allowed": False,
                    "control_mode": "SEMANTIC_DENSITY_MAP_ONLY" if mode_n != "REPORT_ONLY" else "REPORT_ONLY",
                    "raw_positive_signature": raw_positive_signature,
                    "raw_negative_signature": raw_negative_signature,
                    "raw_positive_normalized_signature": raw_positive_normalized_signature,
                    "raw_negative_normalized_signature": raw_negative_normalized_signature,
                    "active_positive_signature": old_positive_signature,
                    "active_negative_signature": old_negative_signature,
                    "existing_strategy_transform_detected": False,
                    "prompt_idempotence_action": "reuse_existing_active_strategy_carrier",
                    "strategy_transform_stripped_character_count": 0,
                    "object_relation_ontology_status": object_relation_ontology.get("status", "") if isinstance(object_relation_ontology, dict) else "",
                    "object_topology_status": object_topology_map.get("status", "") if isinstance(object_topology_map, dict) else "",
                    "contact_depth_axis_recommended": bool(object_topology_map.get("contact_depth_axis_recommended", False)) if isinstance(object_topology_map, dict) else False,
                    "contact_depth_axis_hint": object_topology_map.get("contact_depth_axis_hint", "") if isinstance(object_topology_map, dict) else "",
                    "prompt_continuity_reused": True,
                    "prompt_continuity_policy": "same_prompt_identity_reuses_current_active_strategy_carrier",
                    "same_prompt_match_basis": same_prompt_match_basis,
                    "same_prompt_mode_matches": bool(mode_matches),
                    "launch_raw_matches": bool(launch_raw_matches),
                    "launch_active_matches": bool(launch_active_matches),
                    "last_runtime_raw_matches": bool(last_runtime_raw_matches),
                    "last_runtime_active_matches": bool(last_runtime_active_matches),
                    "current_active_matches": bool(current_active_matches),
                    "stored_current_active_matches": bool(stored_current_active_matches),
                    "positive_identity_matches": bool(positive_identity_matches),
                    "positive_payload_transforms_to_current_active": bool(positive_payload_transforms_to_current_active),
                    "positive_strategy_identity_matches": bool(positive_strategy_identity_matches),
                    "positive_payload_transform_preview_signature": positive_payload_transform_preview_signature,
                    "positive_payload_transform_preview_normalized_signature": positive_payload_transform_preview_normalized_signature,
                    "negative_identity_matches": bool(negative_identity_matches),
                    "positive_prompt_present_in_continue_payload": bool(positive_prompt_present),
                    "negative_prompt_present_in_continue_payload": bool(negative_prompt_present),
                    "positive_payload_missing": bool(positive_payload_missing),
                    "positive_payload_truncated": bool(positive_payload_truncated),
                    "positive_prompt_payload_reused_previous_active": bool(positive_payload_reuse_previous_active),
                    "positive_prompt_payload_mismatch_policy": (
                        "reuse_previous_active_when_positive_payload_is_missing_or_truncated_and_negative_strategy_matches"
                        if positive_payload_reuse_previous_active
                        else "none"
                    ),
                    "negative_payload_missing": bool(negative_payload_missing),
                    "negative_payload_truncated": bool(negative_payload_truncated),
                    "negative_prompt_payload_reused_previous_active": bool(negative_payload_reuse_previous_active),
                    "negative_prompt_payload_mismatch_policy": (
                        "reuse_previous_active_when_positive_identity_matches_and_negative_payload_is_missing_or_truncated"
                        if negative_payload_reuse_previous_active
                        else "none"
                    ),
                }
                next_positive = positive_prompt
                next_negative = negative_prompt
                route_policy = "same_prompt_identity_reuse_active_strategy_carrier"
                update_status = "reused_active_strategy"
                update_formula = (
                    "Continue sent the same prompt identity; the next segment reuses the current clean StrategyCandidate "
                    "and keeps semantic math outside the CLIP prompt text route."
                )
            else:
                next_positive, next_negative, transform_summary = _transform_prompt_pair_for_runtime(
                    raw_positive,
                    raw_negative,
                    mode_value,
                    applies_to_segment,
                    source,
                )
                transform_summary["prompt_continuity_reused"] = False
                transform_summary["prompt_continuity_policy"] = "raw_prompt_changed_runtime_strategy_update"
                transform_summary["raw_positive_normalized_signature"] = raw_positive_normalized_signature
                transform_summary["raw_negative_normalized_signature"] = raw_negative_normalized_signature
                transform_summary["same_prompt_match_basis"] = "changed"
                transform_summary["same_prompt_mode_matches"] = bool(mode_matches)
                transform_summary["launch_raw_matches"] = bool(launch_raw_matches)
                transform_summary["launch_active_matches"] = bool(launch_active_matches)
                transform_summary["last_runtime_raw_matches"] = bool(last_runtime_raw_matches)
                transform_summary["last_runtime_active_matches"] = bool(last_runtime_active_matches)
                transform_summary["current_active_matches"] = bool(current_active_matches)
                transform_summary["stored_current_active_matches"] = bool(stored_current_active_matches)
                transform_summary["positive_identity_matches"] = bool(positive_identity_matches)
                transform_summary["positive_payload_transforms_to_current_active"] = bool(positive_payload_transforms_to_current_active)
                transform_summary["positive_strategy_identity_matches"] = bool(positive_strategy_identity_matches)
                transform_summary["positive_payload_transform_preview_signature"] = positive_payload_transform_preview_signature
                transform_summary["positive_payload_transform_preview_normalized_signature"] = positive_payload_transform_preview_normalized_signature
                transform_summary["negative_identity_matches"] = bool(negative_identity_matches)
                transform_summary["positive_prompt_present_in_continue_payload"] = bool(positive_prompt_present)
                transform_summary["negative_prompt_present_in_continue_payload"] = bool(negative_prompt_present)
                transform_summary["positive_payload_missing"] = bool(positive_payload_missing)
                transform_summary["positive_payload_truncated"] = bool(positive_payload_truncated)
                transform_summary["positive_prompt_payload_reused_previous_active"] = False
                transform_summary["positive_prompt_payload_mismatch_policy"] = "changed_runtime_strategy_update"
                transform_summary["negative_payload_missing"] = bool(negative_payload_missing)
                transform_summary["negative_payload_truncated"] = bool(negative_payload_truncated)
                transform_summary["negative_prompt_payload_reused_previous_active"] = False
                transform_summary["negative_prompt_payload_mismatch_policy"] = "changed_runtime_strategy_update"
                route_policy = "runtime_prompt_payload_at_continue_click"
                update_status = "applied"
                update_formula = "Continue carried a changed per-cascade prompt Strategy; the next segment is encoded from the clean local prompt while semantic math remains a density/context map."

            positive_prompt = next_positive
            negative_prompt = next_negative
            segment_strategy_carrier_context["prompt_source"] = (
                "continue_same_prompt_reuse_active_strategy" if same_prompt else "continue_widget_payload"
            )
            segment_strategy_carrier_context["prompt_transcode_mode"] = transform_summary.get("prompt_transcode_mode", mode_value)
            segment_strategy_carrier_context["prompt_continuity_reused"] = bool(transform_summary.get("prompt_continuity_reused", False))
            segment_strategy_carrier_context["prompt_continuity_policy"] = str(transform_summary.get("prompt_continuity_policy", ""))
            segment_strategy_carrier_context["last_runtime_prompt_update_applies_to_segment"] = int(applies_to_segment)
            segment_strategy_carrier_context["last_runtime_prompt_update_active_positive_signature"] = transform_summary.get("active_positive_signature", "")
            segment_strategy_carrier_context["last_runtime_prompt_update_active_negative_signature"] = transform_summary.get("active_negative_signature", "")
            segment_strategy_carrier_context["last_runtime_raw_positive_signature"] = raw_positive_signature
            segment_strategy_carrier_context["last_runtime_raw_negative_signature"] = raw_negative_signature
            segment_strategy_carrier_context["last_runtime_raw_positive_normalized_signature"] = raw_positive_normalized_signature
            segment_strategy_carrier_context["last_runtime_raw_negative_normalized_signature"] = raw_negative_normalized_signature
            segment_strategy_carrier_context["last_runtime_active_positive_signature"] = transform_summary.get("active_positive_signature", "")
            segment_strategy_carrier_context["last_runtime_active_negative_signature"] = transform_summary.get("active_negative_signature", "")
            segment_strategy_carrier_context["last_runtime_active_positive_normalized_signature"] = _prompt_normalized_text_signature(next_positive)
            segment_strategy_carrier_context["last_runtime_active_negative_normalized_signature"] = _prompt_normalized_text_signature(next_negative)
            segment_strategy_carrier_context["current_active_positive_signature"] = transform_summary.get("active_positive_signature", "")
            segment_strategy_carrier_context["current_active_negative_signature"] = transform_summary.get("active_negative_signature", "")
            segment_strategy_carrier_context["current_active_positive_normalized_signature"] = _prompt_normalized_text_signature(next_positive)
            segment_strategy_carrier_context["current_active_negative_normalized_signature"] = _prompt_normalized_text_signature(next_negative)
            segment_strategy_carrier_context["current_positive_prompt_transformed"] = bool(transform_summary.get("positive_prompt_transformed", False))
            segment_strategy_carrier_context["current_negative_prompt_transformed"] = bool(transform_summary.get("negative_prompt_transformed", False))
            segment_strategy_carrier_context["current_positive_prompt_sanitized"] = bool(transform_summary.get("positive_prompt_sanitized", False))
            segment_strategy_carrier_context["current_negative_prompt_sanitized"] = bool(transform_summary.get("negative_prompt_sanitized", False))

            update_record = {
                "stage": "EventCascadePromptRuntimeUpdate",
                "status": update_status,
                "pause_segment_index": int(pause_segment_index),
                "applies_to_segment": int(applies_to_segment),
                "payload_version": str(prompt_update.get("payload_version", "")),
                "source": str(source),
                "formula": update_formula,
                "route_policy": route_policy,
                "old_active_positive_signature": old_positive_signature,
                "old_active_negative_signature": old_negative_signature,
                **transform_summary,
                "positive_prompt_changed_from_previous_active": bool(transform_summary.get("active_positive_signature") != old_positive_signature),
                "negative_prompt_changed_from_previous_active": bool(transform_summary.get("active_negative_signature") != old_negative_signature),
                "positive_prompt_length": len(str(next_positive or "")),
                "negative_prompt_length": len(str(next_negative or "")),
                "active_control_allowed": True,
                "control_mode": "CASCADE_LOCAL_PROMPT_STRATEGY",
            }
            packet.setdefault("metadata", {})["last_runtime_prompt_update"] = update_record
            execution_records.append(update_record)

        conflict_ids = []
        relation_ids = []
        signal_ids = []

        packet = record_stage(
            packet,
            stage_name="WanEventWorkflowCore",
            action="INIT_EVENT_HORIZON_CASCADE_WORKFLOW",
            observed_behavior="Terminal-first Singularity workflow initialized",
            metadata={
                "run_id": run_id,
                "generation_target": generation_target,
                "cascade_mode": cascade_mode,
                "cascade_count": requested_cascade_count,
                "frames_per_cascade": frames,
                "terminal_mode": terminal_mode,
                "enable_continuation_outputs": enable_continuation_outputs,
                "execution_mode": execution_mode,
                "active_branch_mode": active_branch_mode,
                "cascade_execution_plan": cascade_execution_plan,
            },
        )

        packet["metadata"]["event_program_status"] = {
            "current_mode": execution_mode,
            "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
            "runtime_name": EVENT_HORIZON_RUNTIME_NAME,
            "math_report_enabled": True,
            "terminal_mode": bool(terminal_mode),
            "generation_target": generation_target,
            "result_status": result_status,
            "program_state": "terminal_save_first_generation_body",
            "final_target_output": "saved_video_or_saved_frames",
            "cascade_execution_plan": cascade_execution_plan,
            "normal_outputs": "status_paths_report_only_no_image_sockets",
            "continuation_outputs": "removed_from_main_terminal_node_in_r15_use_future_extractor_node",
            "sampler_replacement": "EventSamplerCore boundary replacement; native step loop pending",
            "no_fake_success_rule": "source image is never presented as successful VIDEO result",
        }

        packet["metadata"]["terminal_ui_model"] = {
            "start_preview": "input image",
            "result_preview": "generated frames/video representative frame if available",
            "save_first": True,
            "default_target": "VIDEO",
        }

        packet["metadata"]["wan_workflow_interface"] = {
            "interface_variant": "terminal_B_no_lora_strings",
            "source_workflow": "Singularity / Wan2.2 I2V Quant 14B",
            "lora_policy": "Current public route applies LoRA outside; future internal route must apply LoRA before compile.",
            "model_route_builder_policy": model_route_builder_policy,
            "required_external": ["primary_model", "clip", "vae", "source_image_file or image for VIDEO"],
            "optional_external": ["secondary_model", "mask", "external image socket"],
            "width": width,
            "height": height,
            "frames": frames,
            "batch_size": batch_size,
            "fps": fps,
            "seed": seed,
            "generation_target": generation_target,
            "video_format": video_format,
            "force_vhs_video_combine": force_vhs_video_combine,
            "output_target": output_target,
            "output_folder_mode": output_folder_mode,
            "output_folder": output_folder,
            "custom_output_folder": custom_output_folder,
            "preferred_video_dir": r"D:\AI NSFW\VID",
            "preferred_image_dir": r"D:\AI NSFW\PIC",
            "preview_policy": "source preview only; result last-frame only when continuation is enabled",
            "event_strategy_note": "optional future control field; currently stored in Event Program report, not required for animation",
            "cascade_prompt_schedule_policy": "If prompt markers are present, each cascade segment can receive its own StrategyCandidate text; during a pause, Continue can also send current prompt widgets as the next segment Strategy.",
            "execution_mode": execution_mode,
            "branch_mode_requested": branch_mode,
            "branch_mode_active": active_branch_mode,

            "ksampler_replacement": "EventSamplerCore",
            "ksampler_windows": {
                "primary": {"start": primary_start_step, "end": primary_end_step, "add_noise": "enable", "return_leftover_noise": "enable"},
                "secondary": {"start": secondary_start_step, "end": secondary_end_step, "add_noise": "disable", "return_leftover_noise": "disable"},
                "global_steps": global_steps,
            },
        }

        packet["metadata"]["wan_event_internal_topology"] = {
            "universal_event_node_rule": "every internal stage must pass through an Singularity; relation center is Event Singularity",
            "event_horizon_model": {
                "Singularity": "boundary layer where a technical node input/output transition becomes Event-Horizon-aware",
                "EventSingularity": "EventSingularity center where input state, observed behavior, and output state collapse into one relation point",
                "formula": "NodeInputState + NodeObservedBehavior = EventSingularity = NodeOutputState"
            },
            "model_route_builder": {
                "current": "external loader/LoRA/compile/SD3 route is observed as Strategy Operator input",
                "future": "internal route builder may own loaders, LoRA matrix, compile guard and SD3 patching after equivalence gates",
                "strict_order": ["loader", "lora_matrix", "compile_guard", "sd3_shift", "sampler"],
            },
            "terminal_route": [
                "EventTextEncodePositive",
                "EventTextEncodeNegative",
                "EventImageScaleStart",
                "EventWanImageToVideoSeed",
                "EventModelShiftHigh",
                "EventSamplerHigh",
                "EventCleanupBetweenSamplers",
                "EventModelShiftLow",
                "EventSamplerLow",
                "EventVAEDecodeTiled",
                "EventVideoCombine",
                "EventSaveReport",
            ],
            "internal_hidden_states": [
                "conditioning_positive",
                "conditioning_negative",
                "wan_latent_seed",
                "latent_before_high",
                "latent_after_high",
                "delta_high",
                "latent_before_low",
                "latent_after_low",
                "delta_low",
                "decoded_frames",
                "video_file",
                "S0_wan_terminal",
            ],
        }

        if str(positive_prompt or "").strip():
            packet, sig, proj, conf = self._make_text_signal(packet, positive_prompt, "StrategyCurrent", "wan_positive_prompt_route", "EventTextEncodePositive")
            signal_ids.append(sig["id"])
            conflict_ids.extend(conf)
        if str(negative_prompt or "").strip():
            packet, sig, proj, conf = self._make_text_signal(packet, negative_prompt, "NegativeConstraint", "wan_negative_prompt_route", "EventTextEncodeNegative")
            signal_ids.append(sig["id"])
            conflict_ids.extend(conf)
        if str(event_strategy or "").strip():
            packet, sig, proj, conf = self._make_text_signal(packet, event_strategy, "EventProgramStrategy", "wan_event_strategy_route", "Wan_event_strategy")
            signal_ids.append(sig["id"])
            conflict_ids.extend(conf)

        if image is not None:
            packet, sig, proj, conf = _read_signal(
                packet, TECH_IMAGE, SPACE_IMAGE, image, "EventSourcePreview", "SourceAnchor",
                "wan_source_image_route", "ImageOutcomeReader", route_position="start_preview"
            )
            signal_ids.append(sig["id"])
            conflict_ids.extend(conf)

        if generation_target == "VIDEO" and image is None:
            failure_reason = "VIDEO target requires source image for WanImageToVideo; no image was connected."
            result_status = "FAILED"
            packet, cid = self._fail_conflict(packet, "WanEventWorkflowCore/InputValidation", failure_reason, {"generation_target": generation_target})
            conflict_ids.append(cid)

        if execution_mode in ("RUN", "TRY_EVENT_HORIZON_CASCADE", "TRY_WAN_SOLO_EXECUTION") and result_status != "FAILED":
            execution_records.append({
                "stage": "SingularityCascadeExecutionGate",
                "status": "running",
                "execution_mode": execution_mode,
                "cascade_count": requested_cascade_count,
                "frames_per_cascade": frames,
                "cascade_execution_plan": cascade_execution_plan,
            })
            try:
                if image is None:
                    raise RuntimeError("Singularity VIDEO route requires source image input.")

                execution_records.append({
                    "stage": "SingularityCascadeSegmentBegin",
                    "status": "begin",
                    "segment_index": 1,
                    "frames": int(frames),
                    "route": "initial_body",
                    "formula": "source image + first sampler body = EventSingularity_segment_1 = decoded frame batch",
                })

                segment_positive_prompt = self._cascade_prompt_for_segment(positive_prompt, 1, execution_records, kind="positive")
                segment_negative_prompt = self._cascade_prompt_for_segment(negative_prompt, 1, execution_records, kind="negative")
                self._record_segment_strategy_carrier(
                    execution_records,
                    segment_index=1,
                    positive_prompt=segment_positive_prompt,
                    negative_prompt=segment_negative_prompt,
                    context=segment_strategy_carrier_context,
                )
                positive = self._encode_text_with_strategy_cache(
                    clip,
                    segment_positive_prompt,
                    execution_records,
                    label='TextEncodePositive',
                    context=segment_strategy_carrier_context,
                    polarity="positive",
                )
                negative = self._encode_text_with_strategy_cache(
                    clip,
                    segment_negative_prompt,
                    execution_records,
                    label='TextEncodeNegative',
                    context=segment_strategy_carrier_context,
                    polarity="negative",
                )
                execution_records.append({"stage": "EventTextEncode", "status": "ok", "formula": "prompt + CLIP behavior = S_text = conditioning"})
                self._stage_delay(stage_delay_seconds, execution_records, "after_text_encode")

                scaled_image = self._scale_image(
                    image,
                    width,
                    height,
                    image_upscale_method,
                    image_crop,
                    execution_records,
                    segment_index=1,
                    route_label="first_body",
                )

                wan_positive, wan_negative, wan_latent = self._wan_image_to_video(
                    positive, negative, vae, scaled_image, width, height, frames, batch_size, execution_records,
                    segment_index=1,
                    route_label="first_body",
                )
                self._stage_delay(stage_delay_seconds, execution_records, "after_wan_image_to_video")

                packet, wan_latent_sig, wan_latent_proj, conf = _read_signal(
                    packet, TECH_LATENT, SPACE_LATENT, wan_latent,
                    "EventWanImageToVideoSeed", "OutcomePrevious",
                    "wan_i2v_latent_seed_route", "LatentEventReader", route_position="wan_i2v_latent_seed"
                )
                conflict_ids.extend(conf)

                high_model = self._apply_sd3_shift(primary_model, primary_sd3_shift, "high", execution_records)

                print(f"[RAW5] FIRST BODY high steps: {primary_start_step}-{primary_end_step} cfg={primary_cfg} global={global_steps}")
                print(f"[RAW5] FIRST BODY low steps target: {secondary_start_step}-{secondary_end_step} cfg={secondary_cfg}")

                high_window = EventSamplerWindow(
                    branch_name="high",
                    branch_role="coarse_motion_global_trajectory",
                    seed=int(seed),
                    steps=int(global_steps),
                    cfg=float(primary_cfg),
                    sampler_name=str(sampler_name),
                    scheduler=str(scheduler),
                    start_at_step=int(primary_start_step),
                    end_at_step=int(primary_end_step),
                    add_noise="enable",
                    return_with_leftover_noise="enable",
                    sd3_shift=float(primary_sd3_shift),
                )

                # Fixed unpack: _event_sample_window now consistently returns (latent_after, raw_delta_norm, delta_tensor)
                # (the 2-value unpack was stale from before the raw_delta return was added for formula in segments).
                # Without this, ValueError on return, caught by outer except, low force block never reached.
                # This was the root cause of "low does not run" + only high raw printed + 60s even after unconditional force code.
                latent_after_high, _high_raw_delta, _high_delta = self._event_sample_window(
                    high_model, wan_positive, wan_negative, wan_latent, high_window, execution_records
                )
                self._stage_delay(stage_delay_seconds, execution_records, "after_high_sampler")
                packet["metadata"].setdefault("event_sampler_results", {})["high"] = {
                    "ok": True,
                    "raw_delta_norm": _high_raw_delta,
                    "first_body": True,
                }

                # === DUMB DIRECT LOW BYPASS FOR FIRST BODY (img2vid start) ===
                # Full study of entire _knowledge_base + the VERY LAST node Gemini made (r58_InputIntegrityHardening.zip extracted nodes.py + core/):
                # Even in final r58 (and r55/r49): normalize_clean had secondary_start_n = primary_end_n (or clamp min=primary_end) + "forced_to_primary_end_for_dual_branch",
                # and the first-body low path was ONLY "if active_branch_mode == "DUAL_HIGH_LOW" and secondary_model is not None"" (plus branch_barrier smart cleanup).
                # + wrapper _event_sample_window (EventSamplerCore boundary) for low.
                # This + pause/cascade_1 logic is why low never started after high in 0-1/1-4 LightV2x "img to vid start" despite correct prints.
                # Per core directive + user: direct physical cut, user steps literal (never mutated), formula reads raw values as ObservedBehavior,
                # direct _low_level (let native KSamplerAdvanced own the exact 1-4 window + progress, no custom mechanism).
                # We force the low here when *user's* secondary range has start < end (for acceleration), using secondary or primary_model with native CFG.
                # Records kept minimal for tail/Mirror/formula. Later segments keep their paths. Old overriding DUAL block removed.
                # See also nodes.py _normalize_clean_inputs (no primary_end link, dual_branch=True hard).
                packet, high_sig, high_proj, conf = _read_signal(
                    packet, TECH_LATENT, SPACE_LATENT, latent_after_high,
                    "EventSamplerHigh_latent_after", "OutcomeNext",
                    "wan_high_after_route", "LatentEventReader", route_position="after_event_sampler_high"
                )
                conflict_ids.extend(conf)

                delta_high, delta_err = compute_tensor_delta(wan_latent, latent_after_high)

                if delta_high is not None:
                    packet, delta_sig, delta_proj, conf = _read_signal(
                        packet, TECH_DELTA, SPACE_DELTA, delta_high,
                        "EventSamplerHigh_delta", "ObservedBehaviorCurrent",
                        "wan_high_delta_route", "DeltaReader",
                        metadata={"before_signal_id": wan_latent_sig["id"], "after_signal_id": high_sig["id"], "before_ref": extract_latent_samples(wan_latent)}
                    )
                    conflict_ids.extend(conf)
                    rel = make_event_relation(
                        relation_type=REL_TRANSFORMS_INTO,
                        source_signal_ids=[wan_latent_sig["id"], delta_sig["id"]],
                        target_signal_ids=[high_sig["id"]],
                        source_projection_ids=[wan_latent_proj["id"], delta_proj["id"]],
                        target_projection_ids=[high_proj["id"]],
                        operator_name="EventSamplerCore",
                        formula_meaning="EventSamplerHigh transforms Wan latent seed into high/noise motion latent",
                        local_strategy_id="S0_wan_terminal.high_sampler",
                        equality_status=EQ_UNKNOWN,
                        metadata={"branch": "high", "sampler_replacement": "EventSamplerCore"},
                    )
                    packet = add_relation(packet, rel)
                    relation_ids.append(rel["id"])
                else:
                    execution_records.append({"stage": "EventSamplerHigh_delta", "status": "unavailable", "error": str(delta_err)})

                final_latent = latent_after_high
                delta_low = None

                # === ALWAYS FORCE LOW AFTER HIGH IN FIRST BODY (img to vid start) ===
                # Dumb physical: for the critical first segment with 0-1 high / 1-4 low acceleration,
                # always run the low directly using the user-specified secondary steps and native CFG.
                # This bypasses any guard, any step linking from normalize, any wrapper, any pause/cascade logic.
                # Matches the requirement "0-1 / 1-4 steps always for light v2x" and the direct low call to let native KSampler own the window.
                # The condition on start < end was not reliably triggering in practice (even when prints showed 1-4),
                # so we force it unconditionally here for the first body. Later segments use their own logic.
                print("[RAW] *** FORCING low immediately after first body high (direct _low_level_sampler_operation) ***")
                print(f"[RAW5] FIRST BODY forcing low with steps {secondary_start_step}-{secondary_end_step}")
                # Debug prints modeled after Gemini's last r59 attempt in the live ComfyUI-Event-Equality-Core when she was forcing low to run
                print(f"\n[EVENT HORIZON DEBUG] RUNNING LOW SAMPLER (direct bypass for first body).")
                print(f"[EVENT HORIZON DEBUG] primary_start={primary_start_step}, primary_end={primary_end_step}")
                print(f"[EVENT HORIZON DEBUG] secondary_start={secondary_start_step}, secondary_end={secondary_end_step}")
                print(f"[EVENT HORIZON DEBUG] window_steps={int(secondary_end_step) - int(secondary_start_step)}")
                import torch
                raw_delta_norm = 0.0
                if delta_high is not None:
                    try:
                        raw_delta_norm = float(torch.linalg.vector_norm(delta_high).item())
                    except Exception:
                        raw_delta_norm = 0.0
                low_delta_strength = float(getattr(self, "_event_delta_strengths", {}).get("low", 1.0) or 1.0)
                low_cfg = self._bounded_strategy_cfg(
                    secondary_cfg,
                    raw_delta_norm,
                    low_delta_strength,
                    execution_records,
                    "first_body_low_cfg",
                )
                low_model = self._apply_sd3_shift(secondary_model or primary_model, secondary_sd3_shift, "low", execution_records)
                # DIRECT, no EventSamplerCore / wrapper / per-step math owning the KSamplerAdvanced window.
                # This is the dumb physical bypass for the exact symptom (low never starts after high in first body + pause/cascade_1).
                print(f"[RAW] *** DIRECT LOW BYPASS CALL: start={secondary_start_step} end={secondary_end_step} cfg={low_cfg} (strength={low_delta_strength}, raw_delta_norm={raw_delta_norm}) (this is the 1-4 refinement, NOT another high) ***")
                self._record_sampler_route_parity_probe(
                    execution_records,
                    "EventRawVsSingularityParity_FirstBodyLowDirectBegin",
                    branch_name="low",
                    route_variant="first_body_direct_low_bypass",
                    latent_before=latent_after_high,
                    latent_after=None,
                    model=low_model,
                    seed=int(seed),
                    steps=int(global_steps),
                    cfg=low_cfg,
                    sampler_name=str(sampler_name),
                    scheduler=str(scheduler),
                    start_at_step=int(secondary_start_step),
                    end_at_step=int(secondary_end_step),
                    add_noise="disable",
                    return_leftover_noise="disable",
                    sd3_shift=float(secondary_sd3_shift),
                    segment_index=1,
                    extra={
                        "branch_role": "refinement_detail_stabilization_first_body_bypass",
                        "math_control_mode": str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY")),
                        "reason": "first body uses direct KSamplerAdvanced bypass while later cascade segments use EventSamplerCore",
                        "raw_high_delta_norm_for_cfg_policy": raw_delta_norm,
                        "low_delta_strength": low_delta_strength,
                    },
                )
                latent_after_low_native = self._low_level_sampler_operation(
                    model=low_model,
                    positive=wan_positive,
                    negative=wan_negative,
                    latent=latent_after_high,
                    seed=int(seed),
                    steps=int(global_steps),
                    cfg=low_cfg,
                    sampler_name=str(sampler_name),
                    scheduler=str(scheduler),
                    start_at_step=int(secondary_start_step),
                    end_at_step=int(secondary_end_step),
                    add_noise="disable",
                    return_leftover_noise="disable",
                )
                self._record_sampler_route_parity_probe(
                    execution_records,
                    "EventRawVsSingularityParity_FirstBodyLowDirectRawAfter",
                    branch_name="low",
                    route_variant="first_body_direct_low_bypass",
                    latent_before=latent_after_high,
                    latent_after=latent_after_low_native,
                    model=low_model,
                    seed=int(seed),
                    steps=int(global_steps),
                    cfg=low_cfg,
                    sampler_name=str(sampler_name),
                    scheduler=str(scheduler),
                    start_at_step=int(secondary_start_step),
                    end_at_step=int(secondary_end_step),
                    add_noise="disable",
                    return_leftover_noise="disable",
                    sd3_shift=float(secondary_sd3_shift),
                    segment_index=1,
                    extra={
                        "branch_role": "refinement_detail_stabilization_first_body_bypass",
                        "math_control_mode": str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY")),
                        "probe_scope": "direct_low_native_output_before_delta_overlay",
                    },
                )
                latent_after_low = self._apply_latent_delta_control(
                    latent_after_high,
                    latent_after_low_native,
                    "low",
                    execution_records,
                )
                self._record_sampler_route_parity_probe(
                    execution_records,
                    "EventRawVsSingularityParity_FirstBodyLowDirectControlledAfter",
                    branch_name="low",
                    route_variant="first_body_direct_low_bypass",
                    latent_before=latent_after_high,
                    latent_after=latent_after_low,
                    model=low_model,
                    seed=int(seed),
                    steps=int(global_steps),
                    cfg=low_cfg,
                    sampler_name=str(sampler_name),
                    scheduler=str(scheduler),
                    start_at_step=int(secondary_start_step),
                    end_at_step=int(secondary_end_step),
                    add_noise="disable",
                    return_leftover_noise="disable",
                    sd3_shift=float(secondary_sd3_shift),
                    segment_index=1,
                    extra={
                        "branch_role": "refinement_detail_stabilization_first_body_bypass",
                        "math_control_mode": str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY")),
                        "probe_scope": "direct_low_output_after_delta_overlay",
                    },
                )
                final_latent = latent_after_low
                print(f"[RAW] low done with cfg={low_cfg} (first body direct bypass)")
                print(f"[RAW] LOW COMPLETE - final_latent now from low (should fix noise break at end if previous was double-high)")
                delta_low_native, _ = compute_tensor_delta(latent_after_high, latent_after_low_native)
                low_native_delta_norm = 0.0
                if delta_low_native is not None:
                    import torch
                    low_native_delta_norm = float(torch.linalg.vector_norm(delta_low_native).item())
                delta_low_effective, _ = compute_tensor_delta(latent_after_high, latent_after_low)
                low_effective_delta_norm = 0.0
                if delta_low_effective is not None:
                    import torch
                    low_effective_delta_norm = float(torch.linalg.vector_norm(delta_low_effective).item())
                print(f"[RAW] low native_delta_norm={low_native_delta_norm} effective_delta_norm={low_effective_delta_norm}")

                # === CONNECT THE LOW "MAKARONY" / PASTA (internal stage wiring for first body) ===
                # To make the direct low part of the expected internal node chain (the "pasta").
                # The expected records in runtime (from packet) include EventMath_low_latent_before/after,
                # EventUniversalMath_EventSamplerLow, finite guard, etc.
                # Without these, the low in first body (the critical img-to-vid start) is "disconnected" from
                # reports, SState, formula integrity checks, Tail3 scoring, etc.
                # We replicate the key post-processing from _event_sample_window (but keep direct native call
                # for the sampler itself, no EventSamplerCore wrapper owning progress/steps).
                self._math_tensor_summary(latent_after_high, execution_records, "EventMath_low_latent_before", strict=False)
                self._math_tensor_summary(latent_after_low, execution_records, "EventMath_low_latent_after", reference=latent_after_high, strict=False)

                self._finite_guard(latent_after_low, execution_records, "EventFiniteGuard_low_latent_after", strict=True)

                self._event_universal_stage_math(
                    execution_records,
                    "EventSamplerLow",
                    input_state=latent_after_high,
                    output_state=latent_after_low,
                    observed_behavior=f"low sampler (direct first body bypass) transformed high latent through step window {secondary_start_step}->{secondary_end_step}",
                    formula_role="LATENT OutcomePrevious (high) + direct low sampler update = LATENT OutcomeNext (first body StrategyCarrier)",
                    route_id="route_sampler_low_first_body",
                    next_requirement="VAE decode requires event-compatible low latent",
                    control_mode=str(getattr(self, "_event_math_control_mode", "OBSERVE_ONLY")),
                    metadata={
                        "branch": "low",
                        "start_at_step": int(secondary_start_step),
                        "end_at_step": int(secondary_end_step),
                        "add_noise": "disable",
                        "return_with_leftover_noise": "disable",
                        "sampler_execution_path": "direct_low_level_bypass",
                    },
                )

                # Record enough for math/tail/Mirror without letting wrapper own the sampler call
                execution_records.append({
                    "stage": "event_sampler_begin",
                    "status": "begin",
                    "branch_name": "low",
                    "branch_role": "refinement_detail_stabilization_first_body_bypass",
                    "start_at_step": int(secondary_start_step),
                    "end_at_step": int(secondary_end_step),
                    "replacement_layer": "direct_low_level_bypass",
                })
                execution_records.append({
                    "stage": "event_sampler_end",
                    "status": "ok",
                    "branch_name": "low",
                })
                packet["metadata"].setdefault("event_sampler_results", {})["low"] = {
                    "ok": True,
                    "direct_bypass": True,
                    "cfg_used": low_cfg,
                    "native_delta_norm": low_native_delta_norm,
                    "effective_delta_norm": low_effective_delta_norm,
                    "cfg_policy": "model_native_cfg_preserved_in_LATENT_DELTA_SCALE",
                }

                packet, low_before_sig, low_before_proj, conf = _read_signal(
                    packet, TECH_LATENT, SPACE_LATENT, latent_after_high,
                    "EventSamplerLow_latent_before", "OutcomePrevious",
                    "wan_low_before_route", "LatentEventReader", route_position="before_event_sampler_low"
                )
                conflict_ids.extend(conf)

                packet, low_after_sig, low_after_proj, conf = _read_signal(
                    packet, TECH_LATENT, SPACE_LATENT, latent_after_low,
                    "EventSamplerLow_latent_after", "OutcomeNext",
                    "wan_low_after_route", "LatentEventReader", route_position="after_event_sampler_low"
                )
                conflict_ids.extend(conf)

                delta_low, delta_err = compute_tensor_delta(latent_after_high, latent_after_low)
                if delta_low is not None:
                    packet, delta2_sig, delta2_proj, conf = _read_signal(
                        packet, TECH_DELTA, SPACE_DELTA, delta_low,
                        "EventSamplerLow_delta", "ObservedBehaviorCurrent",
                        "wan_low_delta_route", "DeltaReader",
                        metadata={"before_signal_id": high_sig["id"], "after_signal_id": low_after_sig["id"], "before_ref": extract_latent_samples(latent_after_high)}
                    )
                    conflict_ids.extend(conf)
                    rel = make_event_relation(
                        relation_type=REL_TRANSFORMS_INTO,
                        source_signal_ids=[high_sig["id"], delta2_sig["id"]],
                        target_signal_ids=[low_after_sig["id"]],
                        source_projection_ids=[high_proj["id"], delta2_proj["id"]],
                        target_projection_ids=[low_after_proj["id"]],
                        operator_name="DirectLowLevelBypass",
                        formula_meaning="(first body direct bypass per KB) high Outcome + low sampler ObservedBehavior = low StrategyCarrier admissible continuation",
                        local_strategy_id="S0_wan_terminal.low_sampler",
                        equality_status=EQ_UNKNOWN,
                        metadata={
                            "branch": "low",
                            "sampler": "direct_low_level_bypass",
                            "cfg_used": low_cfg,
                            "native_delta_norm": low_native_delta_norm,
                            "effective_delta_norm": low_effective_delta_norm,
                            "delta_role": "ObservedBehaviorCurrent",
                        },
                    )
                    packet = add_relation(packet, rel)
                    relation_ids.append(rel["id"])
                else:
                    execution_records.append({"stage": "EventSamplerLow_delta", "status": "unavailable", "error": str(delta_err)})

                execution_records.append({
                    "stage": "EventMathStrategyProposal_low",
                    "status": "recorded",
                    "native_delta_norm": low_native_delta_norm,
                    "effective_delta_norm": low_effective_delta_norm,
                    "formula": "Outcome(high) + low sampler ObservedBehavior = StrategyCarrier for decode/continuation (first body)"
                })

                # _dual_branch_delta_coupling_math excised (physical cut #21): removed smart post-hoc alignment scoring on raw branch deltas.
                # Let the raw interaction stand without an interpretive comfort layer.

                generated_latent = final_latent
                print(f"[RAW] decode + pause logic will use LOW result (not high). pause_after_cascade_1={pause_after_cascade_1}")

                generated_frames = self._decode_tiled(
                    vae, generated_latent,
                    decode_tile_size, decode_overlap, decode_temporal_size, decode_temporal_overlap,
                    execution_records,
                    segment_index=1,
                    route_label="first_body",
                )
                self._math_tensor_summary(generated_frames, execution_records, "EventMath_decoded_frames", strict=False)
                self._frame_motion_math(generated_frames, execution_records, "EventMath_decoded_frame_motion")
                execution_records.append({
                    "stage": "SingularityCascadeSegmentEnd",
                    "status": "ok",
                    "segment_index": 1,
                    "frames": int(generated_frames.shape[0]) if hasattr(generated_frames, "shape") and len(generated_frames.shape) > 0 else int(frames),
                    "last_frame_for_next_segment": True,
                    "route": "initial_body",
                })
                self._stage_delay(stage_delay_seconds, execution_records, "after_decode")

                packet, img_sig, img_proj, conf = _read_signal(
                    packet, TECH_IMAGE, SPACE_IMAGE, generated_frames,
                    "EventVAEDecodeTiled_frames", "VisibleOutcome",
                "wan_decoded_frames_route", "ImageOutcomeReader", route_position="decoded_frames"
                )
                conflict_ids.extend(conf)

                # Singularity extension.
                self._pause_flag_triggered = False
                # Resume cache logic fully excised (physical cut #16): no more protected state carry-over for pause/resume.
                # Same-run pause/continue trims frames/latent and resumes segment 2 from the selected MirrorCut anchor.
                segment_batches = [generated_frames]
                current_cascade_image = self._last_frame_image(generated_frames, width, height)
                start_segment = 2
                pending_resume_frame_index = None
                pending_resume_target_t = None

                # Handle pause at cascade 1
                if pause_after_cascade_1 and start_segment == 2:
                    resume_frame_index, _pause_frames, prompt_update = self._wait_for_cascade_continue(
                        pause_node_id,
                        generated_frames,
                        1,
                        execution_records,
                        stitched_preview_frames=generated_frames,
                        fps=fps,
                    )
                    generated_frames, generated_latent, target_t = self._trim_cascade_resume_state(
                        generated_frames,
                        generated_latent,
                        resume_frame_index,
                        execution_records,
                        "SingularityCascadeResume",
                    )

                    segment_batches = [generated_frames]
                    current_cascade_image = self._last_frame_image(generated_frames, width, height)
                    pending_resume_frame_index = int(resume_frame_index)
                    pending_resume_target_t = int(target_t)
                    _apply_runtime_prompt_update(prompt_update, 1, start_segment)
                    remaining_strategy = self._build_cascade_remaining_strategy(
                        positive_prompt=positive_prompt,
                        negative_prompt=negative_prompt,
                        pause_segment_index=1,
                        next_segment_index=start_segment,
                        resume_frame_index=resume_frame_index,
                        frames_per_cascade=frames,
                        frames=generated_frames,
                        records=execution_records,
                    )
                    self._activate_cascade_remaining_strategy(remaining_strategy, execution_records)
                    if str(prompt_transcode_mode_n or "").upper() == "TRANSFORM_PROMPT":
                        phase_positive, phase_summary = self._build_cascade_phase_prompt_transform(
                            positive_prompt=positive_prompt,
                            remaining_strategy=remaining_strategy,
                            applies_to_segment=start_segment,
                            preserve_prompt_carrier=bool(segment_strategy_carrier_context.get("prompt_continuity_reused", False)),
                            preserve_reason=str(segment_strategy_carrier_context.get("prompt_continuity_policy", "")),
                            records=execution_records,
                        )
                        if phase_summary.get("status") == "applied" and phase_positive != positive_prompt:
                            positive_prompt = phase_positive
                            segment_strategy_carrier_context["prompt_source"] = "cascade_phase_prompt_transform"
                            segment_strategy_carrier_context["current_active_positive_signature"] = _prompt_text_signature(positive_prompt)
                            segment_strategy_carrier_context["current_active_positive_normalized_signature"] = _prompt_normalized_text_signature(positive_prompt)
                            segment_strategy_carrier_context["current_positive_prompt_transformed"] = True
                    pause_after_cascade_1 = False
                    self._pause_flag_triggered = False
                    if requested_cascade_count < start_segment:
                        requested_cascade_count = start_segment
                        execution_records.append({
                            "stage": "SingularityCascadeResumeCountLift",
                            "status": "resume_requires_next_segment",
                            "start_segment": start_segment,
                            "cascade_count": requested_cascade_count,
                        })
                    execution_records.append({
                        "stage": "SingularityCascadeResume",
                        "status": "resumed_same_run",
                        "node_id": pause_node_id,
                        "resume_frame_index": int(resume_frame_index),
                        "latent_temporal_target_t": int(target_t),
                        "start_segment": start_segment,
                        "formula": "same-run pause selected a MirrorCut frame and can carry a fresh prompt Strategy for the next cascade segment",
                    })
                    print(f"[RAW] PAUSE AFTER CASCADE 1 CONTINUED in same prompt from frame {resume_frame_index}.")

                if requested_cascade_count >= start_segment:
                    execution_records.append({
                        "stage": "SingularityCascadeBegin",
                        "status": "begin",
                        "cascade_count": requested_cascade_count,
                        "frames_per_cascade": frames,
                        "cascade_execution_plan": cascade_execution_plan,
                    })
                    for segment_index in range(start_segment, requested_cascade_count + 1):
                        previous_segment_frames = segment_batches[-1] if segment_batches else None
                        previous_segment_latent = generated_latent
                        segment_source_image = current_cascade_image
                        continuity_resume_frame_index = pending_resume_frame_index
                        continuity_resume_target_t = pending_resume_target_t
                        next_frames, current_cascade_image, generated_latent = self._run_event_horizon_segment_core(
                            segment_index=segment_index,
                            source_image=segment_source_image,
                            previous_segment_latent=previous_segment_latent,
                            primary_model=primary_model,
                            secondary_model=secondary_model,
                            clip=clip,
                            vae=vae,
                            positive_prompt=positive_prompt,
                            negative_prompt=negative_prompt,
                            active_branch_mode=active_branch_mode,
                            width=width,
                            height=height,
                            frames=frames,
                            batch_size=batch_size,
                            seed=seed,
                            sampler_name=sampler_name,
                            scheduler=scheduler,
                            global_steps=global_steps,
                            primary_cfg=primary_cfg,
                            secondary_cfg=secondary_cfg,
                            primary_start_step=primary_start_step,
                            primary_end_step=primary_end_step,
                            secondary_start_step=secondary_start_step,
                            secondary_end_step=secondary_end_step,
                            primary_sd3_shift=primary_sd3_shift,
                            secondary_sd3_shift=secondary_sd3_shift,
                            decode_tile_size=decode_tile_size,
                            decode_overlap=decode_overlap,
                            decode_temporal_size=decode_temporal_size,
                            decode_temporal_overlap=decode_temporal_overlap,
                            image_upscale_method=image_upscale_method,
                            image_crop=image_crop,

                            stage_delay_seconds=stage_delay_seconds,
                            records=execution_records,

                            barrier_records=branch_barrier_records,
                            strategy_carrier_context=segment_strategy_carrier_context,
                        )
                        if segment_index > 1:
                            next_frames = self._drop_first_frame_batch(next_frames, execution_records, segment_index)
                        self._record_cascade_strategy_continuity_probe(
                            execution_records,
                            segment_index=segment_index,
                            previous_frames=previous_segment_frames,
                            next_source_image=segment_source_image,
                            next_frames=next_frames,
                            resume_frame_index=continuity_resume_frame_index,
                            latent_temporal_target_t=continuity_resume_target_t,
                            strategy_carrier_context=segment_strategy_carrier_context,
                        )
                        pending_resume_frame_index = None
                        pending_resume_target_t = None
                        self._cascade_boundary_math(segment_batches[-1], next_frames, execution_records, segment_index)
                        segment_batches.append(next_frames)
                        
                        pause_flags = {
                            2: pause_after_cascade_2,
                            3: pause_after_cascade_3,
                            4: pause_after_cascade_4
                        }
                        if pause_flags.get(segment_index, False):
                            pause_preview_frames = self._concat_frame_batches_for_pause_preview(
                                segment_batches,
                                execution_records,
                                segment_index,
                            )
                            resume_frame_index, _pause_frames, prompt_update = self._wait_for_cascade_continue(
                                pause_node_id,
                                next_frames,
                                segment_index,
                                execution_records,
                                stitched_preview_frames=pause_preview_frames,
                                fps=fps,
                            )
                            next_frames, generated_latent, target_t = self._trim_cascade_resume_state(
                                next_frames,
                                generated_latent,
                                resume_frame_index,
                                execution_records,
                                f"SingularityCascadeResume_{segment_index}",
                            )
                            segment_batches[-1] = next_frames
                            current_cascade_image = self._last_frame_image(next_frames, width, height)
                            pending_resume_frame_index = int(resume_frame_index)
                            pending_resume_target_t = int(target_t)
                            _apply_runtime_prompt_update(prompt_update, segment_index, int(segment_index) + 1)
                            remaining_strategy = self._build_cascade_remaining_strategy(
                                positive_prompt=positive_prompt,
                                negative_prompt=negative_prompt,
                                pause_segment_index=segment_index,
                                next_segment_index=int(segment_index) + 1,
                                resume_frame_index=resume_frame_index,
                                frames_per_cascade=frames,
                                frames=next_frames,
                                records=execution_records,
                            )
                            self._activate_cascade_remaining_strategy(remaining_strategy, execution_records)
                            if str(prompt_transcode_mode_n or "").upper() == "TRANSFORM_PROMPT":
                                phase_positive, phase_summary = self._build_cascade_phase_prompt_transform(
                                    positive_prompt=positive_prompt,
                                    remaining_strategy=remaining_strategy,
                                    applies_to_segment=int(segment_index) + 1,
                                    preserve_prompt_carrier=bool(segment_strategy_carrier_context.get("prompt_continuity_reused", False)),
                                    preserve_reason=str(segment_strategy_carrier_context.get("prompt_continuity_policy", "")),
                                    records=execution_records,
                                )
                                if phase_summary.get("status") == "applied" and phase_positive != positive_prompt:
                                    positive_prompt = phase_positive
                                    segment_strategy_carrier_context["prompt_source"] = "cascade_phase_prompt_transform"
                                    segment_strategy_carrier_context["current_active_positive_signature"] = _prompt_text_signature(positive_prompt)
                                    segment_strategy_carrier_context["current_active_positive_normalized_signature"] = _prompt_normalized_text_signature(positive_prompt)
                                    segment_strategy_carrier_context["current_positive_prompt_transformed"] = True
                            self._pause_flag_triggered = False
                            execution_records.append({
                                "stage": "SingularityCascadeResume",
                                "status": "resumed_same_run",
                                "node_id": pause_node_id,
                                "segment_index": int(segment_index),
                                "resume_frame_index": int(resume_frame_index),
                                "latent_temporal_target_t": int(target_t),
                                "start_segment": int(segment_index) + 1,
                                "formula": "same-run pause selected a MirrorCut frame, trimmed the decoded frame batch, and continued with an optional local prompt Strategy for the next segment",
                            })
                            print(f"[RAW] PAUSE AFTER CASCADE {segment_index} CONTINUED in same prompt from frame {resume_frame_index}.")

                    generated_frames = self._concat_frame_batches(segment_batches, execution_records)
                    self._frame_motion_math(generated_frames, execution_records, "EventMath_concatenated_frame_motion")
                    self._cascade_seam_motion_review(
                        segment_batches,
                        generated_frames,
                        execution_records,
                        frames_per_cascade=int(frames),
                    )
                    completed_segments = len(segment_batches)
                    execution_records.append({
                        "stage": "SingularityCascadeEnd",
                        "status": "ok",
                        "segments": completed_segments,
                        "requested_segments": requested_cascade_count,
                        "total_requested_frames": requested_cascade_count * int(frames),
                        "actual_output_frames": int(generated_frames.shape[0]) if generated_frames is not None and hasattr(generated_frames, "shape") else None,
                    })
                    self._event_universal_stage_math(
                        execution_records,
                        "SingularityCascadeEnd",
                        input_state=segment_batches[0] if segment_batches else None,
                        output_state=generated_frames,
                        observed_behavior="multiple cascade frame batches concatenated into one continuous output sequence",
                        formula_role="FRAME_BATCHES segment outcomes -> FRAMES full cascade outcome",
                        route_id="route_cascade_concat",
                        next_requirement="video combine requires one ordered frame batch",
                        control_mode="REPORT_ONLY",
                        metadata={"segments": completed_segments, "requested_segments": requested_cascade_count, "frames_per_cascade": int(frames)},
                    )

                if save_frames or save_output_image or generation_target == "IMAGE":
                    try:
                        self._save_image_attempt(generated_frames, save_prefix, execution_records)
                    except Exception as e:
                        execution_records.append({"stage": "EventSaveFrames", "status": "failed", "error": str(e)})

                if requested_video and save_video:
                    try:
                        saved_video_path = self._video_combine_attempt(generated_frames, fps, save_prefix, execution_records, video_format=video_format, force_vhs=force_vhs_video_combine, output_target=output_target, output_folder_mode=output_folder_mode, output_folder=output_folder, custom_output_folder=custom_output_folder)
                        result_status = "VIDEO"
                    except Exception as e:
                        execution_records.append({"stage": "EventVideoCombine", "status": "failed", "error": str(e)})
                        if generation_target == "VIDEO":
                            result_status = "FAILED"
                            failure_reason = f"VIDEO target requested but video combine failed: {e}"
                            packet, cid = self._fail_conflict(packet, "EventVideoCombine", failure_reason, {"generation_target": generation_target})
                            conflict_ids.append(cid)
                        else:
                            result_status = "FRAMES"
                else:
                    result_status = "FRAMES" if generated_frames is not None else "NONE"

                ui_images = []

                video_ui_payload = self._extract_vhs_ui_payload_from_records(execution_records)

            except InterruptProcessingException as e:
                failure_reason = str(e) or "Singularity cascade paused run was cancelled."
                result_status = "CANCELLED"
                execution_records.append({
                    "stage": "SingularityCascadeCancel",
                    "status": "cancelled",
                    "error": failure_reason,
                })
            except Exception as e:
                failure_reason = str(e)
                result_status = "FAILED"
                execution_records.append({"stage": "terminal_wan_execution", "status": "failed", "error": failure_reason})
                packet, cid = self._fail_conflict(packet, "WanEventWorkflowCore/Execution", failure_reason, {"execution_mode": execution_mode, "generation_target": generation_target})
                conflict_ids.append(cid)

        elif execution_mode == "REPORT_ONLY":
            result_status = "INCOMPLETE"
            execution_records.append({"stage": "execution", "status": "skipped_report_only"})
        else:
            result_status = "INCOMPLETE"
            execution_records.append({
                "stage": "execution",
                "status": "skipped_unsupported_execution_mode",
                "execution_mode": execution_mode,
                "expected_modes": ["RUN", "TRY_EVENT_HORIZON_CASCADE", "TRY_WAN_SOLO_EXECUTION", "REPORT_ONLY"],
            })

        # r12 safety: saved_video_path must be a clean string path, never raw VHS payload.
        if not isinstance(saved_video_path, str):
            raw_saved_video_payload = str(saved_video_path)[:1000]
            try:
                saved_video_path = self._extract_path_from_video_result(saved_video_path, save_prefix, execution_records)
            except Exception as e:
                execution_records.append({"stage": "EventVideoPathFinalCoerce", "status": "failed", "error": str(e), "raw_payload": raw_saved_video_payload})
                saved_video_path = ""
            else:
                execution_records.append({"stage": "EventVideoPathFinalCoerce", "status": "ok", "path": saved_video_path, "raw_payload": raw_saved_video_payload})

        barrier_phase_counts = {}
        barrier_with_strategy = 0
        barrier_preserved = 0
        for item in branch_barrier_records:
            phase = str(item.get("barrier_phase", "unknown"))
            barrier_phase_counts[phase] = barrier_phase_counts.get(phase, 0) + 1
            if item.get("strategy_state_required"):
                barrier_with_strategy += 1
                if item.get("strategy_state_preserved"):
                    barrier_preserved += 1


        packet["metadata"]["branch_barrier_records"] = branch_barrier_records
        packet["metadata"]["branch_barrier_summary"] = {
            "record_count": len(branch_barrier_records),
            "phase_counts": barrier_phase_counts,
            "strategy_state_checks": barrier_with_strategy,
            "strategy_state_preserved_count": barrier_preserved,
            "observer_only": True,
        }
        packet["metadata"]["execution_records"] = execution_records
        packet["metadata"]["result_status"] = {
            "generation_target": generation_target,
            "result_status": result_status,
            "failure_reason": failure_reason,
            "saved_video_path": saved_video_path,
            "cascade_execution_plan": cascade_execution_plan,
            "fake_success_prevented": True,
            "source_image_was_not_returned_as_successful_video": True,
            "source_image_file": str(source_image_file) if source_image_file else "",
        }
        packet["metadata"]["program_output_policy"] = {
            "terminal_mode": bool(terminal_mode),
            "normal_result": "saved files + status, no IMAGE outputs",
            "continuation_outputs": "removed from main terminal node in r15",
            "image_passthrough_policy": "never claim source image as generated VIDEO result",
            "video_required_for_success": generation_target == "VIDEO",
        }
        # update program status with final result
        packet["metadata"]["event_program_status"]["result_status"] = result_status
        packet["metadata"]["event_program_status"]["saved_video_path"] = saved_video_path
        packet["metadata"]["event_program_status"]["failure_reason"] = failure_reason

        # Tail formula evidence must exist before Event Core finalization/report
        # so Strategy Matrix can see tail->next-source as an observed collision.
        tail_result = self._select_best_tail_frames(generated_frames, count=TAIL_CANDIDATE_COUNT, records=execution_records)
        tail_candidate_frames = tail_result["frames"] if tail_result else None
        tail_scores = tail_result.get("scores", [0.0] * TAIL_CANDIDATE_COUNT) if tail_result else [0.0] * TAIL_CANDIDATE_COUNT
        mirror_breaks = tail_result.get("mirror_break_scores", [0.5] * TAIL_CANDIDATE_COUNT) if tail_result else [0.5] * TAIL_CANDIDATE_COUNT
        admissible_conts = tail_result.get("admissible_continuation_scores", [0.5] * TAIL_CANDIDATE_COUNT) if tail_result else [0.5] * TAIL_CANDIDATE_COUNT

        continuation_fitness = self._compute_continuation_fitness(
            tail_candidate_frames,
            source_image=uploaded_image,
            records=execution_records,
        )

        formula_best_index = 0
        if tail_scores:
            formula_best_index = tail_scores.index(max(tail_scores))

        continuation_scores = [0.0] * TAIL_CANDIDATE_COUNT
        if continuation_fitness and continuation_fitness.get("fitness"):
            fit = continuation_fitness["fitness"]
            for i in range(min(TAIL_CANDIDATE_COUNT, len(fit))):
                sys_score = tail_scores[i] if i < len(tail_scores) else 0.0
                cont_score = fit[i]
                adm = admissible_conts[i] if i < len(admissible_conts) else 0.5
                continuation_scores[i] = 0.50 * sys_score + 0.30 * cont_score + 0.20 * adm

            max_c = max(continuation_scores) or 1.0
            if max_c > 0:
                continuation_scores = [s / max_c for s in continuation_scores]

        system_best_for_continuation = 0
        if continuation_scores:
            system_best_for_continuation = continuation_scores.index(max(continuation_scores))

        try:
            manual_sel = int(float(str(selected_tail_index).strip())) if selected_tail_index is not None else -1
        except Exception:
            manual_sel = -1
        valid_tail_indices = tuple(range(TAIL_CANDIDATE_COUNT))
        if manual_sel not in (-1,) + valid_tail_indices:
            manual_sel = -1
        if use_formula_recommendation:
            initial_selected_index = system_best_for_continuation
        else:
            initial_selected_index = manual_sel if manual_sel in valid_tail_indices else -1

        tail_formula_summary = {
            "tail_5_frames_available": tail_candidate_frames is not None,
            "tail_5_count": tail_candidate_frames.shape[0] if tail_candidate_frames is not None and hasattr(tail_candidate_frames, "shape") else 0,
            "tail_scores": tail_scores,
            "mirror_break_scores": mirror_breaks,
            "admissible_continuation_scores": admissible_conts,
            "formula_best_tail_index": formula_best_index,
            "initial_selected_tail_index": initial_selected_index,
            "system_best_for_continuation": system_best_for_continuation,
            "continuation_fitness_scores": continuation_scores,
            "formula_recommendation_selection_policy": (
                "system_best_for_continuation_when_enabled; recency_is_tiebreaker_only"
                if use_formula_recommendation
                else "manual_selected_tail_index_or_none"
            ),
            "report_timing": "pre_finalize",
            "formula_role": "tail candidate as admissible Outcome(t+1)+ObservedBehavior(t+1) for next cascade Strategy",
        }
        tail_formula_summary["tail_3_frames_available"] = tail_formula_summary["tail_5_frames_available"]
        tail_formula_summary["tail_3_count"] = min(3, tail_formula_summary["tail_5_count"])
        packet["metadata"]["tail_5_formula_summary"] = tail_formula_summary
        packet["metadata"]["tail_3_formula_summary"] = tail_formula_summary
        if tail_candidate_frames is not None:
            tail_best_frames = {
                "description": "Last 5 candidate frames from the end of the generated sequence (formula proposes admissible continuation per KB Mirror reading)",
                "shape": list(tail_candidate_frames.shape) if hasattr(tail_candidate_frames, "shape") else None,
                "scores": tail_scores,
                "mirror_break_scores": mirror_breaks,
                "admissible_continuation_scores": admissible_conts,
                "formula_best_index": formula_best_index,
                "user_selected_index": manual_sel,
                "mirror_note": "Human Strategy (green outline / manual click) meets Formula ObservedBehavior (motion scores). Selection is always bidirectional reweighting - formula proposes, human Strategy decides or overrides.",
            }
            packet["metadata"]["tail_5_best_frames"] = tail_best_frames
            packet["metadata"]["tail_3_best_frames"] = tail_best_frames

        background_anchor_card = self._event_background_anchor_preservation_card(
            generated_frames,
            records=execution_records,
        )
        execution_records.append(background_anchor_card)
        packet["metadata"]["background_anchor_preservation"] = background_anchor_card

        packet = self._event_core_body_finalize(packet, execution_records, result_status, saved_video_path, failure_reason)
        packet["metadata"]["execution_records"] = execution_records

        packet, sstate = build_sstate_from_packet(packet, position="S0_wan_terminal", active_relation_ids=relation_ids)

        packet = record_stage(
            packet,
            stage_name="WanEventWorkflowCore",
            action="BUILD_TERMINAL_WAN_REPORT",
            observed_behavior="Built terminal Wan Event Program report with save-first output policy",
            input_signal_ids=signal_ids,
            relation_ids=relation_ids,
            sstate_ids=[sstate["id"]] if sstate else [],
            conflict_ids=conflict_ids,
            formula_note="terminal external inputs -> internal Event-Horizon-aware subnodes -> EventSamplerCore high/low -> decoded frames/video -> saved result",
            metadata={
                "result_status": result_status,
                "failure_reason": failure_reason,
                "saved_video_path": saved_video_path,
                "execution_records": execution_records,
            },
        )

        if save_report:
            report = build_markdown_report(packet)
            if not str(report or "").strip():
                execution_records.append({
                    "stage": "EventSaveReportBuild",
                    "status": "empty_report_from_builder",
                    "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
                })
            else:
                execution_records.append({
                    "stage": "EventSaveReportBuild",
                    "status": "ok",
                    "chars": len(str(report)),
                    "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
                })
            try:
                saved_report_path = self._save_report_file(report, save_prefix, "COMFY_OUTPUT", "DEFAULT", "default", "")
                try:
                    report_size = Path(saved_report_path).stat().st_size
                except Exception:
                    report_size = -1
                execution_records.append({
                    "stage": "EventSaveReport",
                    "status": "standard_comfy_output_ok",
                    "path": saved_report_path,
                    "bytes": report_size,
                    "nonempty": bool(report_size and report_size > 0),
                    "runtime_version": EVENT_HORIZON_RUNTIME_VERSION,
                })
                if report_size <= 0:
                    raise RuntimeError(f"Report was saved but is empty: {saved_report_path}")
                sidecar_info = self._save_runtime_monitor_sidecars(packet, saved_report_path, saved_video_path, save_prefix)
                execution_records.append({
                    "stage": "EventRuntimeMonitorSidecars",
                    "status": sidecar_info.get("status", "unknown"),
                    "json_path": sidecar_info.get("json_path", ""),
                    "csv_path": sidecar_info.get("csv_path", ""),
                    "diff_path": sidecar_info.get("diff_path", ""),
                    "previous_json_path": sidecar_info.get("previous_json_path", ""),
                    "record_count": sidecar_info.get("record_count", 0),
                    "motion_stage": sidecar_info.get("motion_stage", ""),
                    "motion_profile": sidecar_info.get("motion_profile", ""),
                    "motion_stability_score": sidecar_info.get("motion_stability_score", ""),
                    "observer_only": True,
                    "formula": "Runtime Monitor writes machine-readable ObservedBehavior sidecars without changing generation.",
                })
                if hasattr(self, "_event_auto_calibration_finalize"):
                    try:
                        execution_records.append(self._event_auto_calibration_finalize(
                            packet,
                            execution_records,
                            saved_report_path=saved_report_path,
                            saved_video_path=saved_video_path,
                        ))
                    except Exception as e:
                        execution_records.append({
                            "stage": "EventAutoCalibrationResult",
                            "status": "failed",
                            "error": str(e),
                            "active_control_allowed": False,
                        })
                report = build_markdown_report(packet)
                report_size = self._rewrite_report_file(saved_report_path, report)
            except Exception as e:
                execution_records.append({"stage": "EventSaveReport", "status": "failed", "error": str(e), "runtime_version": EVENT_HORIZON_RUNTIME_VERSION})
                saved_report_path = ""
        else:
            report = ""
            saved_report_path = ""
            execution_records.append({"stage": "EventSaveReport", "status": "disabled_by_user", "runtime_version": EVENT_HORIZON_RUNTIME_VERSION})

        # Preview model:
        # - source_preview always shows the input/source.
        # - result_preview shows generated frames only if real generation produced frames.
        # - if generation failed, result_preview is a blank placeholder, not the source image.
        if enable_continuation_outputs and generated_frames is not None and result_status in ("FRAMES", "VIDEO"):
            result_preview = self._representative_preview_frame(generated_frames, width, height, mode="last")
        else:
            result_preview = self._placeholder_image(width, height)

        # Tail math was computed before Event Core finalization/report; here we only expose the already-scored frames.
        # Split into 5 separate IMAGE outputs for the UI layer to use.
        tail_frame_0 = None
        tail_frame_1 = None
        tail_frame_2 = None
        tail_frame_3 = None
        tail_frame_4 = None
        if tail_candidate_frames is not None and hasattr(tail_candidate_frames, "shape") and tail_candidate_frames.shape[0] > 0:
            tail_frame_0 = tail_candidate_frames[0:1]
            if tail_candidate_frames.shape[0] >= 2:
                tail_frame_1 = tail_candidate_frames[1:2]
            if tail_candidate_frames.shape[0] >= 3:
                tail_frame_2 = tail_candidate_frames[2:3]
            if tail_candidate_frames.shape[0] >= 4:
                tail_frame_3 = tail_candidate_frames[3:4]
            if tail_candidate_frames.shape[0] >= 5:
                tail_frame_4 = tail_candidate_frames[4:5]

        if not ui_images:
            ui_images = self._make_ui_previews(source_preview, result_preview, save_prefix, execution_records, include_result_preview=enable_continuation_outputs)
        packet["metadata"]["ui_preview"] = {
            "source_preview": "source image or upload",
            "result_preview": "disabled; no PreviewImage calls in terminal node",
            "continuation_seed_frame": "not emitted by main terminal node in r15; use future extractor/chain node",
            "tail_frames_exposed": "tail_frame_0 / tail_frame_1 / tail_frame_2 / tail_frame_3 / tail_frame_4 as separate IMAGE outputs (5 candidate frames from tail)",
            "ui_images_count": len(ui_images),
            "video_ui_payload_returned": bool(video_ui_payload),
            "tail_5_frames_available": tail_candidate_frames is not None,
            "tail_5_count": tail_candidate_frames.shape[0] if tail_candidate_frames is not None and hasattr(tail_candidate_frames, "shape") else 0,
            "tail_3_frames_available": tail_candidate_frames is not None,
            "tail_3_count": min(3, tail_candidate_frames.shape[0]) if tail_candidate_frames is not None and hasattr(tail_candidate_frames, "shape") else 0,
            "tail_scores": tail_scores,  # blended system + KB formula admissible continuation scores
            "mirror_break_scores": mirror_breaks,  # per-candidate semantic distance (lower = better event continuation)
            "admissible_continuation_scores": admissible_conts,  # right side of formula: how well candidate continues the event (B+ + O+)
            "formula_best_tail_index": formula_best_index,  # what the raw formula recommends as the best of the tail candidates
            "initial_selected_tail_index": initial_selected_index,  # respects the use_formula_recommendation toggle (default manual)
            "system_best_for_continuation": system_best_for_continuation,  # hybrid: raw deltas/motion + fitness + admissible continuation
            "continuation_fitness_scores": continuation_scores if 'continuation_scores' in locals() else [],
            "formula_raw_note": "KB Mirror Core bidirectional: left=segment Strategy (Outcome+Observed via raw_delta + motion at high/low seam), right=candidate tail as admissible causal continuation. MirrorBreak = semantic distance. See FORMULA_TAIL_MIRROR_BREAK records. Always manual green; gold only proposal when use_formula_recommendation."
        }

        if tail_candidate_frames is not None:
            tail_best_frames = {
                "description": "Last 5 candidate frames from the end of the generated sequence (formula proposes admissible continuation per KB Mirror reading)",
                "shape": list(tail_candidate_frames.shape) if hasattr(tail_candidate_frames, "shape") else None,
                "scores": tail_scores,
                "mirror_break_scores": mirror_breaks,
                "admissible_continuation_scores": admissible_conts,
                "formula_best_index": formula_best_index,
                "user_selected_index": manual_sel if 'manual_sel' in locals() else initial_selected_index,
                "mirror_note": "Human Strategy (green outline / manual click) meets Formula ObservedBehavior (motion scores). Selection is always bidirectional reweighting — formula proposes, human Strategy decides or overrides."
            }
            packet["metadata"]["tail_5_best_frames"] = tail_best_frames
            packet["metadata"]["tail_3_best_frames"] = tail_best_frames

        status = (
            f"Singularity v{EVENT_HORIZON_RUNTIME_VERSION} | target={generation_target} | result={result_status} | "
            f"terminal={terminal_mode} | continuation={enable_continuation_outputs} | "
            f"video_path={saved_video_path or 'none'} | report_path={saved_report_path or 'none'}"
        )
        if failure_reason:
            status += f" | failure={failure_reason}"

        continuation_image = self._representative_preview_frame(generated_frames, width, height, mode="last") if (enable_continuation_outputs and generated_frames is not None) else self._placeholder_image(width, height)
        continuation_latent = generated_latent if enable_continuation_outputs else None
        continuation_packet = packet if enable_continuation_outputs else None

        result_tuple = (
            status,
            saved_video_path,
            saved_report_path,
            report,
            tail_frame_0,
            tail_frame_1,
            tail_frame_2,
            tail_frame_3,
            tail_frame_4,
        )

        if not video_ui_payload:
            video_ui_payload = {}
            
        if video_ui_payload:
            return {"ui": video_ui_payload, "result": result_tuple}

        return result_tuple







