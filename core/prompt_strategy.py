import hashlib
import math
import re
from collections import Counter


_WORD_RE = re.compile(r"[^\W_]+(?:[-'][^\W_]+)?", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u9fff]")


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "the", "to", "with", "without", "this", "that", "into",
    "over", "under", "very", "more", "less", "best", "worst", "low", "high",
    "quality", "bad", "good", "image", "video", "frame", "frames",
}


_RELATION_KEYWORDS = {
    "motion": {
        "move", "moves", "moving", "motion", "walk", "walking", "turn", "turning",
        "rotate", "rotating", "breathe", "breathing", "rise", "fall", "sway",
        "shake", "push", "pull", "lean", "lift", "lower", "gesture", "dynamic",
        "animation", "animated", "action", "flow", "flows", "flowing", "run",
        "running", "jump", "jumping", "swing", "swinging", "движение", "двигаться",
        "двигается", "анимация", "динамика", "идет", "идти", "поворот", "дышит",
        "动", "动作", "运动", "移动", "走", "转", "呼吸",
    },
    "contact": {
        "touch", "touching", "hold", "holding", "grab", "grabbing", "press",
        "pressing", "contact", "interaction", "interact", "hands", "hand",
        "fingers", "finger", "connect", "connected", "close", "near", "between",
        "касание", "касается", "рука", "руки", "пальцы", "контакт", "рядом",
        "触摸", "手", "手指", "接触", "互动",
    },
    "spatial": {
        "left", "right", "front", "back", "behind", "above", "below", "inside",
        "outside", "center", "centre", "near", "far", "depth", "foreground",
        "background", "side", "up", "down", "camera", "angle", "pose", "position",
        "слева", "справа", "внутри", "снаружи", "центр", "поза", "позиция",
        "камера", "ракурс", "背景", "前景", "左", "右", "中心", "姿势",
    },
    "temporal": {
        "continuous", "continuity", "stable", "smooth", "sequence", "same", "keep",
        "consistent", "temporal", "across", "frames", "frame", "loop", "cascade",
        "resume", "continue", "продолжение", "стабильно", "плавно", "каскад",
        "кадр", "кадры", "连续", "稳定", "一致", "帧",
    },
    "identity_anchor": {
        "same", "identity", "character", "face", "eyes", "hair", "body", "source",
        "reference", "anchor", "preserve", "consistent", "clothes", "outfit",
        "персонаж", "лицо", "волосы", "тело", "исходник", "сохранить",
        "一致", "角色", "脸", "头发", "身体", "参考",
    },
    "texture_light": {
        "texture", "detail", "details", "sharp", "sharpness", "soft", "blur", "light",
        "lighting", "shadow", "highlight", "reflection", "color", "colour", "skin",
        "cloth", "fabric", "water", "liquid", "smoke", "grain", "noise", "текстура",
        "детали", "свет", "тень", "цвет", "размытие", "шум", "纹理", "细节",
        "光", "阴影", "颜色", "模糊", "噪声",
    },
    "camera_style": {
        "camera", "lens", "shot", "close-up", "closeup", "wide", "pan", "zoom",
        "cinematic", "anime", "realistic", "illustration", "sketch", "style",
        "камера", "объектив", "крупный", "план", "стиль", "аниме", "реализм",
        "镜头", "相机", "风格", "动画",
    },
    "object_topology": {
        "object", "prop", "tool", "device", "toy", "accessory", "instrument",
        "rod", "stick", "pole", "staff", "shaft", "bar", "tube", "cylinder",
        "handle", "bottle", "glass", "cup", "blade", "sword", "knife",
        "solid", "hard", "rigid", "fixed", "stable", "shape", "volume",
        "length", "axis", "dildo", "vibrator", "plug", "объект", "предмет",
        "инструмент", "игрушка", "палка", "стержень", "цилиндр", "трубка",
        "ручка", "жесткий", "твёрдый", "форма", "объём", "длина", "ось",
        "物体", "道具", "工具", "玩具", "硬", "刚性", "形状", "体积", "长度",
    },
    "quality_negative": {
        "bad", "worst", "low", "jpeg", "artifact", "artifacts", "blur", "blurry",
        "deformed", "extra", "missing", "broken", "distorted", "static", "still",
        "flicker", "jitter", "shaky", "ugly", "quality", "качество", "артефакт",
        "сломанный", "статично", "дрожание", "丑陋", "低质量", "静止", "模糊",
    },
}


_RELATION_ROLES = {
    "motion": "ObservedBehavior vector",
    "contact": "relation collision / object-object Strategy point",
    "spatial": "Outcome geometry constraint",
    "temporal": "Outcome(t-1) -> Outcome(t+1) continuity constraint",
    "identity_anchor": "SourceAnchor / OutcomePrevious preservation",
    "texture_light": "visible feature / VAE-facing quality carrier",
    "camera_style": "operator/style carrier",
    "object_topology": "ObjectTopologyCarrier / admissible form-preservation Strategy",
    "quality_negative": "negative constraint / counter-vector",
}


_OBJECT_TOPOLOGY_KEYWORDS = {
    "rigid": {
        "object", "prop", "tool", "device", "toy", "accessory", "instrument",
        "rod", "stick", "pole", "staff", "shaft", "bar", "tube", "cylinder",
        "handle", "bottle", "glass", "cup", "blade", "sword", "knife",
        "solid", "hard", "rigid", "fixed", "plastic", "metal", "wooden",
        "dildo", "vibrator", "plug", "объект", "предмет", "инструмент",
        "игрушка", "палка", "стержень", "цилиндр", "трубка", "ручка",
        "жесткий", "твёрдый", "твердый", "пластик", "металл", "дерево",
        "物体", "道具", "工具", "玩具", "硬", "刚性", "金属", "塑料",
    },
    "soft": {
        "body", "skin", "face", "hair", "hand", "hands", "finger", "fingers",
        "cloth", "fabric", "liquid", "water", "smoke", "soft", "flesh",
        "тело", "кожа", "лицо", "волосы", "рука", "руки", "пальцы",
        "ткань", "жидкость", "вода", "дым", "мягкий", "身体", "皮肤",
        "脸", "头发", "手", "布", "液体", "水", "软",
    },
    "flexible_or_morph": {
        "rubber", "rubbery", "elastic", "flexible", "bendy", "bend", "bending",
        "stretch", "stretching", "melt", "melting", "morph", "morphing",
        "warp", "warping", "deform", "deforming", "резина", "резиновый",
        "эластичный", "гибкий", "гнется", "сгибается", "растягивается",
        "растяжение", "плавится", "морфинг", "деформация", "橡胶", "弹性",
        "弯曲", "变形", "融化",
    },
    "contact_route": {
        "touch", "touching", "hold", "holding", "grab", "grabbing", "press",
        "pressing", "contact", "inside", "insert", "insertion", "enter",
        "interaction", "interact", "касается", "держит", "контакт",
        "внутри", "вставить", "вставляет", "вводит", "взаимодействие",
        "触摸", "接触", "握", "里面", "插入", "互动",
    },
}


def _clamp01(value):
    try:
        value = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _tokenize(text):
    tokens = []
    for item in _WORD_RE.findall(str(text or "").lower()):
        item = item.strip("-'_")
        if len(item) < 2 or item.isdigit():
            continue
        tokens.append(item)
    return tokens


def _signature(text):
    clean = re.sub(r"\s+", " ", str(text or "").strip())
    return hashlib.sha256(clean.encode("utf-8", errors="ignore")).hexdigest()[:16]


_GENERATED_STRATEGY_TRANSFORM_MARKERS = (
    "[SOURCE ANCHOR]",
    "[PRIMARY TERMS]",
    "[ACTIVE RELATIONS]",
    "[EVENT RULE]",
    "[MOTION DAMPING]",
    "[OBJECT TOPOLOGY]",
    "[OBJECT RELATION ONTOLOGY]",
    "source image anchored; continuous cascade route",
    "source image anchored; continuous single route",
    "prompt anchored; continuous cascade route",
    "prompt anchored; continuous single route",
    "resolve freely as one event; keep meaning, visible layout, and motion equal",
    "rigid carrier keeps coherent geometry while contact and motion evolve",
    "soft/contact regions adapt around the rigid carrier",
    "source image is the visible anchor for object identity, contact layout, and frame composition",
    "rigid physical carrier moves as one coherent object with stable silhouette",
    "contact boundary is the local Strategy point",
    "soft contact carrier adapts locally around the relation boundary",
    "motion resolves as relative translation, sliding, or rotation along the relation path",
    "temporal route keeps the same object-contact relation",
    "surface detail follows the carriers as material, lighting, and texture continuity",
)


def _strip_existing_strategy_transform_tail(text):
    raw = str(text or "")
    clean = raw.strip()
    if not clean:
        return raw, {
            "existing_strategy_transform_detected": False,
            "idempotence_action": "empty_prompt",
            "detected_marker": "",
            "raw_positive_signature": _signature(raw),
            "sanitized_positive_signature": _signature(raw),
            "stripped_character_count": 0,
        }

    lower = clean.lower()
    candidates = []
    for marker in _GENERATED_STRATEGY_TRANSFORM_MARKERS:
        marker_l = str(marker or "").lower()
        if not marker_l:
            continue
        idx = lower.find(marker_l)
        if idx >= 0:
            candidates.append((idx, marker))

    if not candidates:
        return raw, {
            "existing_strategy_transform_detected": False,
            "idempotence_action": "raw_user_prompt",
            "detected_marker": "",
            "raw_positive_signature": _signature(raw),
            "sanitized_positive_signature": _signature(raw),
            "stripped_character_count": 0,
        }

    marker_index, marker = min(candidates, key=lambda item: item[0])
    base = clean[:marker_index].rstrip(" .;\n\t")
    if len(base) >= 12:
        action = "stripped_generated_strategy_tail"
        sanitized = base
    else:
        action = "passthrough_existing_strategy_carrier"
        sanitized = clean

    return sanitized, {
        "existing_strategy_transform_detected": True,
        "idempotence_action": action,
        "detected_marker": str(marker),
        "raw_positive_signature": _signature(raw),
        "sanitized_positive_signature": _signature(sanitized),
        "stripped_character_count": max(0, len(clean) - len(sanitized)),
    }


def _keyword_hits(tokens, text, keywords):
    token_set = set(tokens or [])
    hits = []
    text_l = str(text or "").lower()
    for keyword in sorted(keywords):
        key = str(keyword).lower()
        if key in token_set or (_CJK_RE.search(key) and key in text_l):
            hits.append(keyword)
    return hits


def _top_terms(tokens, limit=12):
    useful = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    counts = Counter(useful)
    return [
        {"term": term, "count": int(count)}
        for term, count in counts.most_common(limit)
    ]


def _prompt_segments(text):
    text = str(text or "")
    markers = re.findall(r"(?:^|\n)\s*(?:#{2,}|::|\[)\s*cascade\s*(\d+)", text, flags=re.IGNORECASE)
    if not markers:
        return {"segment_marker_count": 0, "segments_detected": []}
    segments = []
    for raw in markers:
        try:
            segments.append(int(raw))
        except Exception:
            continue
    return {
        "segment_marker_count": len(segments),
        "segments_detected": sorted(set(segments)),
    }


def _term_list(top_terms, limit=8):
    out = []
    for item in top_terms or []:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "") or "").strip()
        if term:
            out.append(term)
        if len(out) >= limit:
            break
    return out


def _keyword_sample(tokens, text, group_id, limit=16):
    return _keyword_hits(tokens, text, _OBJECT_TOPOLOGY_KEYWORDS.get(group_id, set()))[:limit]


def _keyword_negated_in_text(text, keyword):
    text_l = str(text or "").lower()
    key = str(keyword or "").lower()
    if not text_l or not key or key not in text_l:
        return False
    negation_re = re.compile(
        r"(?:\bno\b|\bnot\b|\bwithout\b|\bavoid\b|\bprevent\b|\bnever\b|\bforbid\b|"
        r"\bforbidden\b|\bno\s+accidental\b|не\b|без\b|избег|запрет|нельзя|不要|禁止)"
    )
    for match in re.finditer(re.escape(key), text_l):
        window = text_l[max(0, match.start() - 80):match.start()]
        if negation_re.search(window):
            return True
    return False


def _split_requested_and_protected_hits(tokens, text, group_id):
    hits = _keyword_sample(tokens, text, group_id, limit=24)
    requested = []
    protected = []
    for hit in hits:
        if _keyword_negated_in_text(text, hit):
            protected.append(hit)
        else:
            requested.append(hit)
    return requested, protected


def _build_object_topology_map(tokens, text, relation_vectors, source_present, negative_tokens=None, negative_text=""):
    tokens = tokens or []
    text = str(text or "")
    negative_tokens = negative_tokens or []
    negative_text = str(negative_text or "")
    relations = _relation_lookup(relation_vectors)
    rigid_hits = _keyword_sample(tokens, text, "rigid")
    soft_hits = _keyword_sample(tokens, text, "soft")
    flexible_hits, positive_protection_hits = _split_requested_and_protected_hits(tokens, text, "flexible_or_morph")
    negative_flexible_hits = _keyword_sample(negative_tokens, negative_text, "flexible_or_morph")
    negative_rigid_hits = _keyword_sample(negative_tokens, negative_text, "rigid")
    topology_protection_hits = sorted(set(positive_protection_hits + negative_flexible_hits))
    contact_hits = _keyword_sample(tokens, text, "contact_route")
    depth_axis_terms = sorted(set(
        [
            item
            for item in (rigid_hits + contact_hits)
            if str(item).lower() in {
                "axis", "length", "shaft", "rod", "stick", "pole", "cylinder",
                "inside", "insert", "insertion", "enter", "depth", "ось",
                "длина", "стержень", "цилиндр", "внутри", "вставить",
                "вставляет", "вводит", "长度",
            }
        ]
    ))

    contact_score = max(
        _clamp01(len(contact_hits) / 6.0),
        _clamp01((relations.get("contact", {}) or {}).get("strength_score", 0.0)),
    )
    rigid_score = _clamp01(len(rigid_hits) / 6.0)
    soft_score = _clamp01(len(soft_hits) / 10.0)
    flexible_score = _clamp01(len(flexible_hits) / 5.0)
    topology_pressure = _clamp01(
        0.45 * rigid_score
        + 0.25 * contact_score
        + 0.20 * soft_score
        + 0.10 * (1.0 if source_present else 0.0)
    )
    rigidity_recommended = bool(rigid_hits and flexible_score < 0.50)
    morph_conflict = bool(rigid_hits and flexible_hits)
    topology_protection_detected = bool(rigid_hits and topology_protection_hits)
    contact_depth_axis_recommended = bool(
        rigid_hits
        and (contact_hits or contact_score > 0.0)
        and (depth_axis_terms or contact_score >= 0.15)
    )
    status = "active" if (rigid_hits or soft_hits or contact_hits or flexible_hits) else "absent"

    carriers = []
    if rigid_hits:
        carriers.append({
            "carrier_id": "rigid_object_carrier",
            "topology_class": "rigid" if flexible_score < 0.50 else "rigid_with_flexibility_conflict",
            "formula_role": "ObjectTopologyCarrier / rigid Strategy carrier",
            "terms_sample": rigid_hits,
            "allowed_behavior": [
                "move",
                "rotate",
                "translate",
                "partial occlusion",
                "contact",
                "temporary hidden surface",
            ],
            "forbidden_behavior": [
                "stretch",
                "melt",
                "morph",
                "change length",
                "change volume",
                "lose axis",
                "merge into nearby soft regions",
            ],
            "strategy_rule": (
                "The object may change position and visibility while preserving its local topology; "
                "shape preservation is a semantic Strategy carrier, not a hard physics solver."
            ),
        })
    if soft_hits:
        carriers.append({
            "carrier_id": "soft_region_carrier",
            "topology_class": "soft_or_articulated",
            "formula_role": "Outcome surface / deformable visible carrier",
            "terms_sample": soft_hits,
            "allowed_behavior": ["articulation", "soft deformation", "lighting change", "occlusion"],
            "forbidden_behavior": ["identity loss", "unreadable anatomy", "random melt", "texture collapse"],
        })
    if contact_hits or contact_score > 0.0:
        carriers.append({
            "carrier_id": "contact_region_carrier",
            "topology_class": "relation_boundary",
            "formula_role": "collision point where object carriers meet",
            "terms_sample": contact_hits,
            "allowed_behavior": ["consistent contact", "occlusion boundary", "depth relation", "directional motion"],
            "forbidden_behavior": ["broken relation", "detached contact", "impossible intersection", "contact reset"],
        })
    if contact_depth_axis_recommended:
        carriers.append({
            "carrier_id": "contact_depth_axis_carrier",
            "topology_class": "relative_depth_axis_continuity",
            "formula_role": "ObjectContactDepthCarrier / local Strategy collision geometry",
            "terms_sample": depth_axis_terms or (rigid_hits[:4] + contact_hits[:4]),
            "allowed_behavior": [
                "relative translation along the object axis",
                "continuous visible/occluded length split",
                "continuous depth relation",
                "partial occlusion",
                "contact boundary motion",
            ],
            "forbidden_behavior": [
                "reinterpret hidden length as changed total length",
                "reset contact depth at cascade seam",
                "duplicate depth motion after continuation",
                "treat occlusion as object deformation",
            ],
            "strategy_rule": (
                "Object total length and local axis remain a stable carrier; only the visible/occluded split "
                "and contact depth may evolve through the local relation."
            ),
        })
    if flexible_hits:
        carriers.append({
            "carrier_id": "flexibility_or_morph_carrier",
            "topology_class": "explicit_flexibility_or_morph_request",
            "formula_role": "prompt-authorized deformation carrier",
            "terms_sample": flexible_hits,
            "allowed_behavior": ["deformation only if the prompt asks for it"],
            "forbidden_behavior": ["accidental deformation when a rigid carrier is present"],
        })
    if topology_protection_detected:
        carriers.append({
            "carrier_id": "topology_protection_carrier",
            "topology_class": "negative_or_negated_deformation_boundary",
            "formula_role": "semantic corridor boundary for ObjectTopologyCarrier",
            "terms_sample": topology_protection_hits,
            "allowed_behavior": ["suppress accidental object deformation", "preserve rigid carrier form"],
            "forbidden_behavior": ["treat forbidden deformation words as requested action"],
        })

    return {
        "stage": "EventObjectTopologyCarrier",
        "status": status,
        "topology_version": "object_topology_carrier_v3_positive_admissibility",
        "formula": (
            "Object topology is read as a local Strategy point: established object form + observed interaction "
            "= admissible continuation = future contact behavior + visible object outcome."
        ),
        "model_freedom_policy": (
            "This is semantic guidance for the text-conditioning route. The video model remains free; "
            "Singularity does not replace sampler physics with a fixed external solver."
        ),
        "source_anchor_present": bool(source_present),
        "rigid_object_terms": rigid_hits,
        "negative_rigid_object_terms": negative_rigid_hits,
        "soft_region_terms": soft_hits,
        "contact_route_terms": contact_hits,
        "flexibility_terms": flexible_hits,
        "topology_protection_terms": topology_protection_hits,
        "rigid_object_count": 1 if rigid_hits else 0,
        "soft_region_count": 1 if soft_hits else 0,
        "contact_region_count": 1 if (contact_hits or contact_score > 0.0) else 0,
        "contact_depth_axis_recommended": contact_depth_axis_recommended,
        "contact_depth_axis_terms": depth_axis_terms,
        "rigidity_lock_recommended": rigidity_recommended,
        "rigidity_transform_applied": False,
        "morph_conflict_detected": morph_conflict,
        "topology_protection_detected": topology_protection_detected,
        "topology_pressure_score": topology_pressure,
        "rigidity_confidence_score": rigid_score,
        "contact_pressure_score": contact_score,
        "flexibility_pressure_score": flexible_score,
        "topology_carriers": carriers,
        "transform_hint": (
            "rigid carrier moves as one coherent object with stable shape, length, volume, and axis while soft/contact regions adapt around it"
            if rigidity_recommended else ""
        ),
        "admissible_topology_hint": (
            "rigid carrier keeps coherent geometry; soft carrier may articulate; contact boundary adapts smoothly around both carriers"
            if rigidity_recommended else ""
        ),
        "contact_depth_axis_hint": (
            "preserve total object length and local axis; let only visible/occluded length split and contact depth change continuously"
            if contact_depth_axis_recommended else ""
        ),
        "negative_hint": (
            ""
        ),
        "active_control_allowed": False,
        "control_mode": "REPORT_ONLY",
        "future_control_surface": [
            "prompt Strategy transform",
            "object-contact drift report",
            "tail candidate topology score",
            "bounded sampler research only after visual evidence",
        ],
    }


def _build_object_relation_ontology(object_topology_map, relation_vectors, source_present, cascade_plan):
    object_topology_map = object_topology_map if isinstance(object_topology_map, dict) else {}
    relations = _relation_lookup(relation_vectors)
    cascade_plan = cascade_plan if isinstance(cascade_plan, dict) else {}
    status = object_topology_map.get("status", "absent")
    rigid_active = bool(object_topology_map.get("rigidity_lock_recommended"))
    soft_active = bool(object_topology_map.get("soft_region_count", 0))
    contact_active = bool(object_topology_map.get("contact_region_count", 0) or _relation_active(relations, "contact"))
    contact_depth_axis_active = bool(object_topology_map.get("contact_depth_axis_recommended", False))
    motion_active = bool(_relation_active(relations, "motion"))
    temporal_active = bool(_relation_active(relations, "temporal") or int(cascade_plan.get("requested_segments", 1) or 1) > 1)
    texture_active = bool(_relation_active(relations, "texture_light"))
    ontology_active = bool(status != "absent" or contact_active or motion_active)

    carrier_roles = []
    positive_strategy_sentences = []
    coherence_dimensions = []

    if bool(source_present):
        positive_strategy_sentences.append(
            "source image is the visible anchor for object identity, contact layout, and frame composition"
        )

    if rigid_active:
        carrier_roles.append({
            "carrier_id": "rigid_physical_carrier",
            "formula_role": "stable object identity inside local Strategy",
            "model_language": (
                "same solid object across frames; coherent silhouette, axis, total length, cross-section, volume, "
                "material identity, and local orientation"
            ),
            "admissible_behavior": [
                "translate as one object",
                "slide along the contact path",
                "rotate as one object",
                "be partially occluded",
                "change visibility while keeping identity",
            ],
        })
        coherence_dimensions.extend(["silhouette", "axis", "total length", "cross-section", "volume", "material identity"])
        positive_strategy_sentences.append(
            "rigid physical carrier moves as one coherent object with stable silhouette, axis, total length, cross-section, volume, and material identity"
        )

    if contact_active:
        carrier_roles.append({
            "carrier_id": "contact_boundary_carrier",
            "formula_role": "object-object Strategy collision point",
            "model_language": (
                "readable interface between carriers; depth, occlusion, contact line, and relative direction remain connected"
            ),
            "admissible_behavior": [
                "maintain readable contact boundary",
                "resolve depth through occlusion continuity",
                "move along a shared relation path",
                "carry relative direction from frame to frame",
            ],
        })
        coherence_dimensions.extend(["contact line", "depth relation", "occlusion boundary", "relative direction"])
        positive_strategy_sentences.append(
            "contact boundary is the local Strategy point: depth, occlusion, contact line, and relative direction stay readable while motion continues"
        )

    if contact_depth_axis_active:
        carrier_roles.append({
            "carrier_id": "contact_depth_axis_carrier",
            "formula_role": "relative object-axis depth Strategy",
            "model_language": (
                "object total length and axis stay stable; the visible part, hidden part, and contact depth "
                "change as one continuous relation instead of becoming a new object size"
            ),
            "admissible_behavior": [
                "continuous depth change along the same axis",
                "stable total length with changing visibility",
                "occlusion-aware contact motion",
                "no cascade-seam depth reset",
            ],
        })
        coherence_dimensions.extend([
            "total length",
            "visible length",
            "occluded length",
            "contact depth",
            "axis continuity",
        ])
        positive_strategy_sentences.append(
            "object total length and local axis stay stable; only the visible/occluded length split and contact depth evolve through the same relation"
        )

    if soft_active:
        carrier_roles.append({
            "carrier_id": "soft_contact_carrier",
            "formula_role": "adaptive Outcome surface around relation boundary",
            "model_language": (
                "deformable or articulated region that adapts locally while preserving subject identity and visible structure"
            ),
            "admissible_behavior": [
                "local articulation",
                "surface response around contact",
                "lighting and texture response",
                "identity-preserving shape change",
            ],
        })
        coherence_dimensions.extend(["subject identity", "surface continuity", "local articulation"])
        positive_strategy_sentences.append(
            "soft contact carrier adapts locally around the relation boundary while preserving identity, readable structure, and surface continuity"
        )

    if motion_active or (rigid_active and contact_active):
        positive_strategy_sentences.append(
            "motion resolves as relative translation, sliding, or rotation along the relation path, with object movement and surface response belonging to one event"
        )

    if temporal_active:
        positive_strategy_sentences.append(
            "temporal route keeps the same object-contact relation from frame to frame and across selected cascade tail frames"
        )

    if texture_active:
        positive_strategy_sentences.append(
            "surface detail follows the carriers as material, lighting, and texture continuity"
        )

    if not positive_strategy_sentences and ontology_active:
        positive_strategy_sentences.append(
            "scene entities are read as carriers in one event: identity, relation, motion, and visible outcome remain connected"
        )

    relation_map = {
        "rigid_active": rigid_active,
        "soft_active": soft_active,
        "contact_active": contact_active,
        "motion_active": motion_active,
        "temporal_active": temporal_active,
            "texture_active": texture_active,
            "contact_depth_axis_active": contact_depth_axis_active,
            "active_relation_ids": [
            str(item.get("relation_id") or "")
            for item in relation_vectors or []
            if isinstance(item, dict) and str(item.get("status") or "") == "active"
        ],
    }

    return {
        "stage": "EventObjectRelationOntology",
        "status": "active" if ontology_active else "absent",
        "ontology_version": "object_relation_ontology_v1_positive_strategy",
        "formula": (
            "object identity + contact relation = Strategy(relation) = relative motion + preserved visible object outcome"
        ),
        "free_math_policy": (
            "This ontology names carriers and relation points for the positive Strategy route; it does not define a fixed solver."
        ),
        "model_freedom_policy": (
            "The model keeps its native vector-space freedom. Singularity gives it a clearer semantic carrier map."
        ),
        "source_anchor_present": bool(source_present),
        "relation_map": relation_map,
        "carrier_roles": carrier_roles,
        "coherence_dimensions": sorted(set(coherence_dimensions)),
        "positive_strategy_sentences": positive_strategy_sentences[:8],
        "motion_resolution_hint": (
            "Resolve object interaction as sliding, translation, or rotation along the contact path while preserving each carrier's identity."
            if (rigid_active and contact_active) else ""
        ),
        "strategy_point": "object_contact_strategy" if contact_active else "object_identity_strategy",
        "active_control_allowed": False,
        "control_mode": "REPORT_ONLY",
    }


def _relation_lookup(relation_vectors):
    return {
        str(item.get("relation_id") or ""): item
        for item in relation_vectors or []
        if isinstance(item, dict)
    }


def _relation_active(relations, relation_id, threshold=0.001):
    item = relations.get(relation_id, {}) if isinstance(relations, dict) else {}
    return float(item.get("strength_score", 0.0) or 0.0) >= threshold


def _add_operator_block(blocks, block_id, formula_role, operator_text, negative_text="", score=0.0, source="prompt"):
    blocks.append({
        "block_id": str(block_id),
        "formula_role": str(formula_role),
        "operator_text": str(operator_text),
        "negative_text": str(negative_text),
        "strength_score": _clamp01(score),
        "source": str(source),
        "active_control_allowed": False,
    })


def _clean_prompt_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip())


_NEGATED_CONSTRAINT_RE = re.compile(
    r"(?:^|[\s,.;:])("
    r"no|not|without|avoid|prevent|never|forbid|forbidden|"
    r"не|без|избег|запрет|нельзя|不要|禁止"
    r")(?:[\s,.;:]|$)",
    re.IGNORECASE,
)


def _split_prompt_sentences(text):
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", raw)
    return [part.strip(" \n\t.;") for part in parts if part.strip(" \n\t.;")]


def _compact_base_prompt_for_transform(text, max_chars=1200):
    """Keep scene/action content while Strategy clauses replace prohibitions."""
    sentences = _split_prompt_sentences(text)
    if not sentences:
        clean = _clean_prompt_text(text)
        return clean[:max_chars].rstrip(" ,.;")

    kept = []
    deferred_constraint_count = 0
    for sentence in sentences:
        sentence_clean = _clean_prompt_text(sentence)
        if not sentence_clean:
            continue
        is_constraint = bool(_NEGATED_CONSTRAINT_RE.search(sentence_clean))
        if is_constraint and kept:
            deferred_constraint_count += 1
            continue
        kept.append(sentence_clean)

    compact = ". ".join(kept).strip(" .")
    if not compact:
        compact = _clean_prompt_text(text)
    if len(compact) > max_chars:
        compact = compact[:max_chars].rsplit(" ", 1)[0].strip(" ,.;")
    if deferred_constraint_count:
        compact = f"{compact}. Constraint wording is resolved as positive admissible behavior through the Strategy carrier"
    return compact


def _relation_phrase(active_ids):
    labels = []
    if "motion" in active_ids:
        labels.append("one coherent action path")
    if "contact" in active_ids:
        labels.append("readable contact and interaction")
    if "spatial" in active_ids:
        labels.append("stable spatial layout")
    if "temporal" in active_ids:
        labels.append("smooth temporal continuation")
    if "identity_anchor" in active_ids:
        labels.append("same subject identity")
    if "texture_light" in active_ids:
        labels.append("stable color, line, and local detail")
    if "camera_style" in active_ids:
        labels.append("consistent camera and style")
    if "object_topology" in active_ids:
        labels.append("stable object topology")
    return labels[:5]


def _topology_transform_phrases(object_topology_map, object_relation_ontology=None):
    if not isinstance(object_topology_map, dict):
        return []
    phrases = []
    if isinstance(object_relation_ontology, dict):
        phrases.extend([
            str(item)
            for item in object_relation_ontology.get("positive_strategy_sentences", []) or []
            if str(item).strip()
        ])
    if object_topology_map.get("rigidity_lock_recommended"):
        phrases.append("rigid carrier keeps coherent geometry while contact and motion evolve")
    if object_topology_map.get("contact_depth_axis_recommended"):
        phrases.append("object total length and axis stay stable while visible/occluded length and contact depth evolve")
    if object_topology_map.get("morph_conflict_detected"):
        phrases.append("if deformation is part of the event, it stays localized, continuous, and topology-preserving")
    if object_topology_map.get("topology_protection_detected"):
        phrases.append("deformation language is resolved as admissible object behavior instead of a separate action")
    if object_topology_map.get("contact_region_count", 0):
        phrases.append("soft/contact regions adapt around the rigid carrier with readable depth continuity")
    unique = []
    seen = set()
    for phrase in phrases:
        phrase = str(phrase or "").strip()
        if not phrase or phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        unique.append(phrase)
    return unique[:5]


def _build_single_prompt_transform(
    base_text,
    active_ids,
    source_present,
    cascade_plan,
    structured=False,
    object_topology_map=None,
    object_relation_ontology=None,
):
    base_text = _clean_prompt_text(base_text)
    compact_base_text = _compact_base_prompt_for_transform(base_text)
    relation_bits = _relation_phrase(active_ids)
    topology_bits = _topology_transform_phrases(object_topology_map, object_relation_ontology)
    source_bit = "source image anchored" if source_present else "prompt anchored"
    cascade_count = int(cascade_plan.get("requested_segments", 1) or 1) if isinstance(cascade_plan, dict) else 1
    cascade_bit = "continuous cascade route" if cascade_count > 1 else "continuous single route"
    free_math_bit = (
        "resolve freely as one event; keep meaning, visible layout, and motion equal "
        "without creating a second independent action layer"
    )

    if structured:
        parts = []
        if compact_base_text:
            parts.append(compact_base_text)
        parts.extend([
            f"{source_bit}; {cascade_bit}",
            ", ".join(relation_bits) if relation_bits else "one coherent visual relation",
            ", ".join(topology_bits) if topology_bits else "",
            free_math_bit,
        ])
        return ". ".join(part.strip(". ") for part in parts if part).strip()

    extra = [source_bit, cascade_bit]
    if relation_bits:
        extra.append(", ".join(relation_bits))
    if topology_bits:
        extra.append(", ".join(topology_bits))
    extra.append(free_math_bit)
    transform_tail = "; ".join(extra)
    if not compact_base_text:
        return transform_tail
    return f"{compact_base_text}. {transform_tail}"


def _build_single_negative_transform(base_text, active_ids, source_present, cascade_plan, object_topology_map=None):
    # r67 keeps the user's negative prompt intact. Object topology control is
    # expressed as positive admissible behavior in the main Strategy carrier.
    return str(base_text or "")


def _build_model_language_transcode(
    positive_prompt_text,
    negative_prompt_text,
    positive_top_terms,
    negative_top_terms,
    relation_vectors,
    object_topology_map,
    object_relation_ontology,
    source_present,
    cascade_plan,
    scores,
    high_candidates,
    low_candidates,
):
    relations = _relation_lookup(relation_vectors)
    active_ids = [
        str(item.get("relation_id") or "")
        for item in relation_vectors or []
        if isinstance(item, dict) and str(item.get("status") or "") == "active"
    ]
    positive_terms = _term_list(positive_top_terms, limit=10)
    negative_terms = _term_list(negative_top_terms, limit=10)
    relation_complexity = _clamp01(scores.get("relation_complexity_score", 0.0))
    action_pressure = _clamp01(scores.get("action_pressure_score", 0.0))
    anchor_pressure = _clamp01(scores.get("anchor_pressure_score", 0.0))
    semantic_conflict = _clamp01(scores.get("semantic_conflict_score", 0.0))
    collapse_risk = _clamp01(scores.get("collapse_risk_score", 0.0))

    blocks = []
    if source_present:
        _add_operator_block(
            blocks,
            "source_anchor",
            "OutcomePrevious / SourceAnchor",
            "Use the source image as the main anchor for identity, silhouette, camera framing, and visible layout.",
            "Do not replace the source identity, do not drift away from the original composition.",
            max(anchor_pressure, 0.35),
            "source_image",
        )
    else:
        _add_operator_block(
            blocks,
            "scene_seed",
            "StrategyCandidate without explicit SourceAnchor",
            "Build one coherent scene from the prompt before adding motion.",
            "Avoid unrelated scene changes and contradictory visual targets.",
            0.25,
            "prompt",
        )

    if _relation_active(relations, "identity_anchor") or source_present:
        _add_operator_block(
            blocks,
            "identity_continuity",
            "OutcomePrevious preservation",
            "Keep the same primary subject identity, body proportions, face/hair/outfit cues, and source-image continuity across frames.",
            "Avoid identity swap, face drift, body drift, outfit drift, and sudden character replacement.",
            max(anchor_pressure, relations.get("identity_anchor", {}).get("strength_score", 0.0) if relations else 0.0),
            "prompt+source",
        )

    if _relation_active(relations, "motion"):
        _add_operator_block(
            blocks,
            "motion_vector",
            "ObservedBehavior vector",
            "Preserve the action already requested by the original prompt as one coherent motion route; do not add an extra independent motion layer.",
            "Avoid duplicated dynamics, local vibration, body-part shaking, random motion, reversal, jitter, and frame-to-frame snapping.",
            relations.get("motion", {}).get("strength_score", 0.0),
            "prompt",
        )

    if _relation_active(relations, "contact"):
        _add_operator_block(
            blocks,
            "relation_contact",
            "object-relation Strategy point",
            "Make the subject-action-target relation readable: contact, distance, direction, and interaction must stay geometrically consistent.",
            "Avoid broken contact, impossible intersections, detached hands, extra fingers, and unclear target relation.",
            relations.get("contact", {}).get("strength_score", 0.0),
            "prompt",
        )

    if isinstance(object_topology_map, dict) and object_topology_map.get("rigidity_lock_recommended"):
        _add_operator_block(
            blocks,
            "object_topology_rigidity",
            "ObjectTopologyCarrier / admissible form-preservation Strategy",
            (
                "Treat the rigid object as one stable topology carrier: it may move, rotate, be partly hidden, "
                "and make contact, while its shape, length, volume, and axis stay constant unless the prompt explicitly asks for deformation."
            ),
            (
                "Avoid rigid-object stretching, melting, morphing, length or volume changes, axis loss, and accidental merging with nearby soft regions."
            ),
            max(
                0.35,
                object_topology_map.get("topology_pressure_score", 0.0),
                object_topology_map.get("rigidity_confidence_score", 0.0),
            ),
            "object_topology",
        )

    if isinstance(object_topology_map, dict) and object_topology_map.get("contact_depth_axis_recommended"):
        _add_operator_block(
            blocks,
            "object_contact_depth_axis",
            "ObjectContactDepthCarrier / relative axis-depth Strategy",
            (
                "Separate the rigid object's total length from the contact-depth relation: the object keeps the same "
                "axis, total length, cross-section, and material identity, while only the visible part, hidden part, "
                "and contact depth change continuously along the same relation path."
            ),
            (
                "Avoid treating occlusion as object shortening, avoid resetting depth after a cascade seam, "
                "and avoid duplicated local motion caused by a second interpretation of contact depth."
            ),
            max(
                0.35,
                object_topology_map.get("contact_pressure_score", 0.0),
                object_topology_map.get("topology_pressure_score", 0.0),
            ),
            "object_topology",
        )

    if isinstance(object_relation_ontology, dict) and object_relation_ontology.get("status") == "active":
        ontology_sentences = [
            str(item).strip()
            for item in object_relation_ontology.get("positive_strategy_sentences", []) or []
            if str(item).strip()
        ]
        if ontology_sentences:
            _add_operator_block(
                blocks,
                "object_relation_ontology",
                "ObjectRelationOntology / carrier relation Strategy map",
                " ".join(ontology_sentences[:5]),
                "",
                max(0.35, object_topology_map.get("topology_pressure_score", 0.0) if isinstance(object_topology_map, dict) else 0.0),
                "object_relation_ontology",
            )

    if isinstance(object_topology_map, dict) and object_topology_map.get("morph_conflict_detected"):
        _add_operator_block(
            blocks,
            "object_topology_conflict",
            "prompt-authorized deformation boundary",
            (
                "If the prompt truly asks for flexible deformation, keep that deformation localized and readable; "
                "otherwise preserve the rigid object's topology."
            ),
            "Avoid accidental rubber-like drift when rigidity is the intended event carrier.",
            object_topology_map.get("flexibility_pressure_score", 0.0),
            "object_topology",
        )

    if _relation_active(relations, "spatial"):
        _add_operator_block(
            blocks,
            "spatial_geometry",
            "Outcome geometry constraint",
            "Preserve left/right/front/back/depth relations and keep object positions stable while the requested action evolves.",
            "Avoid spatial drift, flipped layout, warped depth, and inconsistent camera geometry.",
            relations.get("spatial", {}).get("strength_score", 0.0),
            "prompt",
        )

    if _relation_active(relations, "temporal") or int(cascade_plan.get("requested_segments", 1) or 1) > 1:
        _add_operator_block(
            blocks,
            "temporal_continuity",
            "Outcome(t-1) -> Outcome(t+1) continuity",
            "Make every frame and cascade segment continue the same event through one shared temporal route.",
            "Avoid reset, duplicated motion boundary, motion echo after a cascade seam, temporal jump, flicker, and unrelated continuation.",
            max(
                relations.get("temporal", {}).get("strength_score", 0.0),
                0.45 if int(cascade_plan.get("requested_segments", 1) or 1) > 1 else 0.0,
            ),
            "prompt+cascade_plan",
        )

    if _relation_active(relations, "texture_light"):
        _add_operator_block(
            blocks,
            "texture_light",
            "visible feature carrier",
            "Preserve readable lines, color, lighting, texture, and local detail without turning texture into animated noise.",
            "Avoid green/gray collapse, excessive blur, crawling texture, noisy color drift, shimmer, and localized tremor.",
            relations.get("texture_light", {}).get("strength_score", 0.0),
            "prompt",
        )

    if _relation_active(relations, "camera_style"):
        _add_operator_block(
            blocks,
            "camera_style",
            "model operator / style carrier",
            "Keep the camera, style, shot scale, and visual language stable across the generated video.",
            "Avoid unwanted style switch, lens drift, and camera inconsistency.",
            relations.get("camera_style", {}).get("strength_score", 0.0),
            "prompt",
        )

    if not active_ids:
        _add_operator_block(
            blocks,
            "basic_coherence",
            "StrategyCandidate fallback",
            "Treat the prompt as one simple coherent visual event with stable identity, readable relations, and consistent output.",
            "Avoid unrelated changes, static output, and low-quality drift.",
            0.25,
            "fallback",
        )

    positive_clauses = [block["operator_text"] for block in blocks if block.get("operator_text")]
    negative_clauses = [block["negative_text"] for block in blocks if block.get("negative_text")]
    if negative_terms:
        negative_clauses.append("Also suppress the existing negative prompt concepts without erasing required scene relations.")
    if semantic_conflict >= 0.25:
        negative_clauses.append("Do not let the negative prompt cancel the requested action or source-anchor identity.")
    if collapse_risk >= 0.45:
        negative_clauses.append("Avoid noise collapse, source-anchor loss, washed-out color, and unreadable final frames.")
    if isinstance(object_topology_map, dict) and object_topology_map.get("negative_hint"):
        negative_clauses.append(str(object_topology_map.get("negative_hint")))

    structured_lines = [
        "[SOURCE ANCHOR] " + ("source image is primary OutcomePrevious" if source_present else "no explicit source image anchor"),
        "[PRIMARY TERMS] " + (", ".join(positive_terms[:8]) if positive_terms else "not extracted"),
        "[ACTIVE RELATIONS] " + (", ".join(active_ids) if active_ids else "basic_coherence"),
        "[EVENT RULE] one coherent event; subject, action, target relation, camera, and temporal route must agree",
        "[MOTION DAMPING] preserve the original requested motion route; do not create a second independent dynamics layer",
        "[OBJECT TOPOLOGY] " + (
            object_topology_map.get("transform_hint")
            if isinstance(object_topology_map, dict) and object_topology_map.get("transform_hint")
            else "no rigid ObjectTopologyCarrier detected"
        ),
        "[OBJECT RELATION ONTOLOGY] " + (
            " ".join(object_relation_ontology.get("positive_strategy_sentences", [])[:5])
            if isinstance(object_relation_ontology, dict) and object_relation_ontology.get("positive_strategy_sentences")
            else "no active object relation ontology"
        ),
        "[CASCADE RULE] " + (
            f"{cascade_plan.get('requested_segments')} segments; pause after {cascade_plan.get('pause_after_segments')}"
            if int(cascade_plan.get("requested_segments", 1) or 1) > 1
            else "single segment or no cascade route"
        ),
    ]
    report_only_transformed_positive_prompt = _build_single_prompt_transform(
        positive_prompt_text,
        active_ids=active_ids,
        source_present=source_present,
        cascade_plan=cascade_plan,
        structured=False,
        object_topology_map=object_topology_map,
        object_relation_ontology=object_relation_ontology,
    )
    report_only_structured_prompt = _build_single_prompt_transform(
        positive_prompt_text,
        active_ids=active_ids,
        source_present=source_present,
        cascade_plan=cascade_plan,
        structured=True,
        object_topology_map=object_topology_map,
        object_relation_ontology=object_relation_ontology,
    )
    report_only_negative_prompt = _build_single_negative_transform(
        negative_prompt_text,
        active_ids=active_ids,
        source_present=source_present,
        cascade_plan=cascade_plan,
        object_topology_map=object_topology_map,
    )

    return {
        "stage": "EventPromptStrategyTranscoder",
        "status": "proposal_only",
        "legacy_stage_alias": "EventPromptModelLanguageDeconstruction",
        "transcoder_version": "prompt_strategy_transform_v5_prompt_purity_density_map",
        "formula": "The prompt is read as one Strategy carrier, but its mathematical/semantic topology is kept outside the model-facing text route.",
        "transcode_policy": "semantic_density_map_only_no_text_injection",
        "transformation_policy": "prompt_purity_lock_semantic_map_only",
        "prompt_purity_lock": True,
        "model_facing_prompt_policy": "raw_user_prompt_or_sanitized_raw_prompt_only",
        "prompt_text_injection_allowed": False,
        "semantic_math_in_prompt_allowed": False,
        "math_sorting_policy": "Sort meaning density against context density in report/control space; do not express this sorting as extra prompt words.",
        "free_math_policy": "Math, meaning, semantics, logic, and Strategy remain linked and free; this stage does not define a fixed physics solver.",
        "motion_damping_policy": "Motion topology is measured as Strategy pressure, not appended as literal text.",
        "object_topology_policy": "Object carriers are mapped as topology and density relations, not injected into the positive prompt as instructions.",
        "object_relation_ontology_policy": "Object and contact roles remain report/control-space carriers unless a future explicit bounded control proves safe.",
        "negative_transform_policy": "keep_user_negative_prompt_byte_route",
        "model_freedom_policy": "The model still solves the scene freely in its native vector space; this stage only builds the map used to decide where future math may act.",
        "original_prompt_meaning_preserved": True,
        "original_prompt_preserved": True,
        "active_control_allowed": False,
        "control_mode": "REPORT_ONLY",
        "manual_prompt_candidate_available": False,
        "auto_apply_allowed": False,
        "transformed_positive_prompt": str(positive_prompt_text or ""),
        "transformed_negative_prompt": str(negative_prompt_text or ""),
        "transformed_structured_positive_prompt": str(positive_prompt_text or ""),
        "model_facing_positive_prompt_source": "raw_user_prompt",
        "model_facing_negative_prompt_source": "raw_user_negative_prompt",
        "report_only_legacy_transformed_positive_prompt": report_only_transformed_positive_prompt,
        "report_only_legacy_transformed_negative_prompt": report_only_negative_prompt,
        "report_only_legacy_structured_positive_prompt": report_only_structured_prompt,
        "report_only_transcoded_positive_prompt": " ".join(positive_clauses),
        "report_only_transcoded_negative_prompt": " ".join(negative_clauses),
        "report_only_structured_transcode_prompt": "\n".join(structured_lines),
        "transcoded_positive_prompt": "",
        "transcoded_negative_prompt": "",
        "structured_transcode_prompt": "",
        "positive_operator_prompt": "",
        "negative_operator_prompt": "",
        "structured_operator_prompt": "",
        "operator_blocks": blocks,
        "object_topology_map": object_topology_map if isinstance(object_topology_map, dict) else {},
        "object_relation_ontology": object_relation_ontology if isinstance(object_relation_ontology, dict) else {},
        "object_relation_ontology_status": object_relation_ontology.get("status") if isinstance(object_relation_ontology, dict) else "absent",
        "relation_ontology_sentence_count": len(object_relation_ontology.get("positive_strategy_sentences", []) or []) if isinstance(object_relation_ontology, dict) else 0,
        "rigid_object_count": int(object_topology_map.get("rigid_object_count", 0)) if isinstance(object_topology_map, dict) else 0,
        "rigidity_lock_recommended": bool(object_topology_map.get("rigidity_lock_recommended", False)) if isinstance(object_topology_map, dict) else False,
        "source_terms_sample": {
            "positive_top_terms": positive_terms,
            "negative_top_terms": negative_terms,
        },
        "strategy_scores": {
            "relation_complexity_score": relation_complexity,
            "action_pressure_score": action_pressure,
            "anchor_pressure_score": anchor_pressure,
            "semantic_conflict_score": semantic_conflict,
            "collapse_risk_score": collapse_risk,
        },
        "research_candidates": {
            "high_delta_strength": high_candidates,
            "low_delta_strength": low_candidates,
        },
        "usage": {
            "safe_now": "Show this in the report as a topology/density map only.",
            "manual_test": "Compare report maps and visible output while the model-facing prompt stays clean.",
            "future_node_mode": "Future active controls should act on tensors/weights/routes, not by adding formula text to the prompt.",
        },
    }


def _build_semantic_density_context_map(
    pos_tokens,
    neg_tokens,
    active_relation_ids,
    relation_vectors,
    source_present,
    cascade_plan,
    scores,
    object_topology_map,
    object_relation_ontology,
):
    pos_tokens = pos_tokens or []
    neg_tokens = neg_tokens or []
    active_relation_ids = active_relation_ids or []
    relation_vectors = relation_vectors or []
    cascade_plan = cascade_plan if isinstance(cascade_plan, dict) else {}
    scores = scores if isinstance(scores, dict) else {}
    object_topology_map = object_topology_map if isinstance(object_topology_map, dict) else {}
    object_relation_ontology = object_relation_ontology if isinstance(object_relation_ontology, dict) else {}

    useful_positive = [item for item in pos_tokens if item not in _STOPWORDS]
    useful_negative = [item for item in neg_tokens if item not in _STOPWORDS]
    relation_strength_total = sum(
        _clamp01(item.get("strength_score", 0.0))
        for item in relation_vectors
        if isinstance(item, dict) and str(item.get("status") or "") == "active"
    )
    relation_density = _clamp01(relation_strength_total / max(1.0, float(len(_RELATION_KEYWORDS))))
    meaning_density = _clamp01(
        0.40 * (len(useful_positive) / 90.0)
        + 0.25 * (len(active_relation_ids) / max(1.0, float(len(_RELATION_KEYWORDS))))
        + 0.20 * _clamp01(scores.get("action_pressure_score", 0.0))
        + 0.15 * relation_density
    )
    context_density = _clamp01(
        0.25 * (1.0 if source_present else 0.0)
        + 0.20 * _clamp01(scores.get("anchor_pressure_score", 0.0))
        + 0.20 * _clamp01(object_topology_map.get("topology_pressure_score", 0.0))
        + 0.15 * (1.0 if object_relation_ontology.get("status") == "active" else 0.0)
        + 0.10 * _clamp01((int(cascade_plan.get("requested_segments", 1) or 1) - 1) / 4.0)
        + 0.10 * _clamp01(len(useful_negative) / 80.0)
    )
    balance_score = _clamp01(1.0 - abs(meaning_density - context_density))
    if meaning_density > context_density + 0.15:
        balance_axis = "meaning_denser_than_context"
        recommended_sort = "reduce duplicated verbal pressure; prefer context anchors and measured route controls"
    elif context_density > meaning_density + 0.15:
        balance_axis = "context_denser_than_meaning"
        recommended_sort = "increase explicit relation evidence before active math; keep prompt text unchanged"
    else:
        balance_axis = "meaning_context_balanced"
        recommended_sort = "keep prompt clean and compare route metrics before applying active math"

    return {
        "map_version": "semantic_density_context_v1_prompt_purity",
        "status": "recorded",
        "formula_role": "Strategy density sorter / report-only",
        "model_facing_prompt_policy": "unchanged_clean_prompt",
        "prompt_text_injection_allowed": False,
        "meaning_density_score": meaning_density,
        "context_density_score": context_density,
        "density_context_balance_score": balance_score,
        "balance_axis": balance_axis,
        "relation_density_score": relation_density,
        "useful_positive_token_count": len(useful_positive),
        "useful_negative_token_count": len(useful_negative),
        "active_relation_count": len(active_relation_ids),
        "source_context_present": bool(source_present),
        "object_topology_pressure_score": _clamp01(object_topology_map.get("topology_pressure_score", 0.0)),
        "object_relation_ontology_active": bool(object_relation_ontology.get("status") == "active"),
        "recommended_sort": recommended_sort,
        "next_control_surface": [
            "conditioning weight/routing map",
            "sampler seam pressure map",
            "latent delta window",
            "tail-frame StrategyCarrier choice",
            "report-only topology evidence",
        ],
    }


def build_prompt_strategy_packet(
    positive_prompt,
    negative_prompt,
    source_image_file="",
    cascade_execution_plan=None,
    math_controls=None,
):
    """Build a report-only Strategy graph from prompt text.

    This intentionally does not rewrite prompts or mutate sampler inputs. It is a
    semantic map for reports and future bounded research controls.
    """
    raw_pos_text = str(positive_prompt or "")
    pos_text, prompt_idempotence = _strip_existing_strategy_transform_tail(raw_pos_text)
    neg_text = str(negative_prompt or "")
    all_text = f"{pos_text}\n{neg_text}"
    pos_tokens = _tokenize(pos_text)
    neg_tokens = _tokenize(neg_text)
    all_tokens = pos_tokens + neg_tokens
    pos_set = {t for t in pos_tokens if t not in _STOPWORDS}
    neg_set = {t for t in neg_tokens if t not in _STOPWORDS}
    overlap = sorted(pos_set.intersection(neg_set))[:24]
    overlap_base = max(1, min(len(pos_set) or 1, len(neg_set) or 1))
    overlap_ratio = _clamp01(len(overlap) / float(overlap_base))

    relation_vectors = []
    active_relation_ids = []
    for relation_id, keywords in _RELATION_KEYWORDS.items():
        pos_hits = _keyword_hits(pos_tokens, pos_text, keywords)
        neg_hits = _keyword_hits(neg_tokens, neg_text, keywords)
        total_hits = len(pos_hits) + len(neg_hits)
        strength = _clamp01(total_hits / 8.0)
        status = "active" if total_hits else "absent"
        if status == "active":
            active_relation_ids.append(relation_id)
        relation_vectors.append({
            "relation_id": relation_id,
            "status": status,
            "formula_role": _RELATION_ROLES.get(relation_id, "Strategy relation carrier"),
            "positive_hits": pos_hits[:10],
            "negative_hits": neg_hits[:10],
            "hit_count": int(total_hits),
            "strength_score": strength,
            "collision_surface": [
                "prompt_image_anchor",
                "positive_negative_prompt_polarity",
                "model_output_scheduler_step",
                "high_low_sampler_strategy",
                "visible_video_outcome",
            ],
            "model_freedom_policy": "The sampler remains free; this vector only names the relation the model is trying to resolve.",
        })

    pos_segments = _prompt_segments(pos_text)
    neg_segments = _prompt_segments(neg_text)
    source_present = bool(str(source_image_file or "").strip() and str(source_image_file or "").strip().lower() != "none")
    cascade_plan = cascade_execution_plan if isinstance(cascade_execution_plan, dict) else {}
    controls = math_controls if isinstance(math_controls, dict) else {}

    active_count = len(active_relation_ids)
    token_pressure = _clamp01(len(pos_tokens) / 180.0)
    relation_pressure = _clamp01(active_count / max(1, len(_RELATION_KEYWORDS)))
    action_pressure = _clamp01(
        sum(v["strength_score"] for v in relation_vectors if v["relation_id"] in ("motion", "contact", "temporal"))
        / 3.0
    )
    anchor_pressure = _clamp01(
        sum(v["strength_score"] for v in relation_vectors if v["relation_id"] in ("identity_anchor", "spatial", "texture_light"))
        / 3.0
    )
    semantic_conflict = _clamp01(0.75 * overlap_ratio + 0.25 * (1.0 if not pos_tokens else 0.0))
    relation_complexity = _clamp01(0.55 * relation_pressure + 0.30 * token_pressure + 0.15 * action_pressure)
    free_model_need = _clamp01(0.50 * relation_complexity + 0.35 * action_pressure + 0.15 * anchor_pressure)

    high_candidates = [1.0]
    if source_present and action_pressure >= 0.20 and semantic_conflict < 0.35:
        high_candidates = [0.98, 0.99, 1.0]
    if relation_complexity >= 0.70:
        high_candidates = [0.99, 1.0]

    low_candidates = [1.0]
    if source_present and action_pressure >= 0.35 and semantic_conflict < 0.25:
        low_candidates = [1.0, 1.001, 1.002, 1.005]

    collapse_risk = _clamp01(
        0.30 * semantic_conflict
        + 0.25 * relation_complexity
        + 0.25 * (1.0 - anchor_pressure if source_present else 0.5)
        + 0.20 * (1.0 if float(controls.get("low_delta_strength", 1.0) or 1.0) > 1.005 else 0.0)
    )

    positive_top_terms = _top_terms(pos_tokens)
    negative_top_terms = _top_terms(neg_tokens)
    object_topology_map = _build_object_topology_map(
        tokens=pos_tokens,
        text=pos_text,
        relation_vectors=relation_vectors,
        source_present=source_present,
        negative_tokens=neg_tokens,
        negative_text=neg_text,
    )
    object_relation_ontology = _build_object_relation_ontology(
        object_topology_map=object_topology_map,
        relation_vectors=relation_vectors,
        source_present=source_present,
        cascade_plan=cascade_plan,
    )
    semantic_density_context_map = _build_semantic_density_context_map(
        pos_tokens=pos_tokens,
        neg_tokens=neg_tokens,
        active_relation_ids=active_relation_ids,
        relation_vectors=relation_vectors,
        source_present=source_present,
        cascade_plan=cascade_plan,
        scores={
            "relation_complexity_score": relation_complexity,
            "action_pressure_score": action_pressure,
            "anchor_pressure_score": anchor_pressure,
            "semantic_conflict_score": semantic_conflict,
            "collapse_risk_score": collapse_risk,
        },
        object_topology_map=object_topology_map,
        object_relation_ontology=object_relation_ontology,
    )

    model_language_transcode = _build_model_language_transcode(
        positive_prompt_text=pos_text,
        negative_prompt_text=neg_text,
        positive_top_terms=positive_top_terms,
        negative_top_terms=negative_top_terms,
        relation_vectors=relation_vectors,
        object_topology_map=object_topology_map,
        object_relation_ontology=object_relation_ontology,
        source_present=source_present,
        cascade_plan=cascade_plan,
        scores={
            "relation_complexity_score": relation_complexity,
            "action_pressure_score": action_pressure,
            "anchor_pressure_score": anchor_pressure,
            "semantic_conflict_score": semantic_conflict,
            "collapse_risk_score": collapse_risk,
        },
        high_candidates=high_candidates,
        low_candidates=low_candidates,
    )
    if isinstance(model_language_transcode, dict):
        model_language_transcode["prompt_idempotence"] = prompt_idempotence
        model_language_transcode["idempotence_policy"] = (
            "Read only the raw user prompt for the semantic map. If an older generated Strategy tail is detected, "
            "strip it before text encoding; if no raw base remains, pass the existing Strategy carrier through."
        )
        model_language_transcode["semantic_density_context_map"] = semantic_density_context_map
        if prompt_idempotence.get("idempotence_action") == "stripped_generated_strategy_tail":
            model_language_transcode["model_facing_positive_prompt_source"] = "sanitized_raw_user_prompt"
        if prompt_idempotence.get("idempotence_action") == "passthrough_existing_strategy_carrier":
            model_language_transcode["transformed_positive_prompt"] = pos_text
            model_language_transcode["transformed_structured_positive_prompt"] = pos_text
            model_language_transcode["manual_prompt_candidate_available"] = False
            model_language_transcode["transformation_policy"] = "idempotent_passthrough_existing_strategy_carrier"

    collision_points = [
        {
            "collision_id": "prompt_image_anchor",
            "why_it_exists": "Text asks for a scene while the source image already carries a visible OutcomePrevious.",
            "local_formula": "source anchor + prompt behavior = Strategy(scene) = conditioning pressure + latent seed",
            "active_control_allowed": False,
        },
        {
            "collision_id": "positive_negative_prompt_polarity",
            "why_it_exists": "Positive and negative prompts form a semantic corridor, not independent text boxes.",
            "local_formula": "positive carrier + negative counter-vector = Strategy corridor",
            "overlap_ratio": overlap_ratio,
            "overlap_terms_sample": overlap[:12],
            "active_control_allowed": False,
        },
        {
            "collision_id": "relation_action_pressure",
            "why_it_exists": "Object relations and actions become the places where the model must choose direction.",
            "local_formula": "relation carriers + previous visible state = local Strategy(action)",
            "action_pressure_score": action_pressure,
            "active_control_allowed": False,
        },
        {
            "collision_id": "high_low_sampler_strategy",
            "why_it_exists": "High sampler creates the coarse StrategyCarrier; low sampler refines it or amplifies collapse.",
            "local_formula": "OutcomeNext(high) = StrategyCarrier(low)",
            "recommended_high_delta_candidates": high_candidates,
            "recommended_low_delta_candidates": low_candidates,
            "active_control_allowed": False,
        },
        {
            "collision_id": "object_topology_contact",
            "why_it_exists": "A visible object carrier can collide with a soft/contact region while needing to stay the same object.",
            "local_formula": "object form + contact behavior = local Strategy(topology) = admissible motion + preserved object outcome",
            "object_topology_status": object_topology_map.get("status"),
            "rigidity_lock_recommended": object_topology_map.get("rigidity_lock_recommended"),
            "rigid_object_count": object_topology_map.get("rigid_object_count"),
            "topology_pressure_score": object_topology_map.get("topology_pressure_score"),
            "active_control_allowed": False,
        },
        {
            "collision_id": "object_relation_ontology",
            "why_it_exists": "The model needs to know what each carrier is and how the carriers relate before it solves motion.",
            "local_formula": "object identity + contact boundary = Strategy(relation) = relative motion + preserved carrier outcome",
            "ontology_status": object_relation_ontology.get("status"),
            "strategy_point": object_relation_ontology.get("strategy_point"),
            "motion_resolution_hint": object_relation_ontology.get("motion_resolution_hint"),
            "active_control_allowed": False,
        },
        {
            "collision_id": "tail_next_source",
            "why_it_exists": "Manual frame choice decides which visible Outcome becomes the next cascade source.",
            "local_formula": "selected tail frame = OutcomePrevious(next cascade)",
            "cascade_segments": cascade_plan.get("requested_segments"),
            "pause_after_segments": cascade_plan.get("pause_after_segments"),
            "active_control_allowed": False,
        },
    ]

    return {
        "stage": "EventPromptStrategyCompiler",
        "status": "recorded",
        "compiler_version": "prompt_strategy_compiler_v6_prompt_purity_density_map",
        "formula": "Prompt text is read into Strategy carriers, density/context balance, vector collisions, object topology carriers, object-relation ontology, and a global/local Strategy return contract. The model-facing prompt text stays clean.",
        "model_freedom_policy": "The video model is allowed to solve the scene in its own native vector space. This packet clarifies what relations should stay equal without injecting formula language into CLIP text.",
        "prompt_purity_lock": True,
        "prompt_text_injection_allowed": False,
        "semantic_math_in_prompt_allowed": False,
        "active_control_allowed": False,
        "control_mode": "REPORT_ONLY",
        "prompt_signatures": {
            "raw_positive": _signature(raw_pos_text),
            "positive": _signature(pos_text),
            "negative": _signature(neg_text),
        },
        "prompt_idempotence": prompt_idempotence,
        "token_stats": {
            "raw_positive_token_count": len(_tokenize(raw_pos_text)),
            "positive_token_count": len(pos_tokens),
            "negative_token_count": len(neg_tokens),
            "cjk_character_count": len(_CJK_RE.findall(all_text)),
            "positive_top_terms": positive_top_terms,
            "negative_top_terms": negative_top_terms,
        },
        "prompt_segments": {
            "positive": pos_segments,
            "negative": neg_segments,
        },
        "source_anchor": {
            "source_image_present": source_present,
            "source_image_file": str(source_image_file or ""),
            "formula_role": "OutcomePrevious / SourceAnchor",
        },
        "strategy_graph": {
            "carriers": [
                "positive_prompt",
                "negative_prompt",
                "source_image",
                "latent_seed",
                "noise_field",
                "model_operator",
                "sampler_route",
                "tail_frame",
                "visible_video",
                "object_topology",
                "object_relation_ontology",
            ],
            "active_relation_ids": active_relation_ids,
            "relation_complexity_score": relation_complexity,
            "action_pressure_score": action_pressure,
            "anchor_pressure_score": anchor_pressure,
            "semantic_conflict_score": semantic_conflict,
            "free_model_reasoning_need_score": free_model_need,
            "collapse_risk_score": collapse_risk,
            "object_topology_pressure_score": object_topology_map.get("topology_pressure_score"),
            "rigidity_lock_recommended": object_topology_map.get("rigidity_lock_recommended"),
            "object_relation_ontology_status": object_relation_ontology.get("status"),
            "object_relation_strategy_point": object_relation_ontology.get("strategy_point"),
            "semantic_density_context_balance_score": semantic_density_context_map.get("density_context_balance_score"),
            "semantic_density_balance_axis": semantic_density_context_map.get("balance_axis"),
        },
        "strategy_return_contract": {
            "contract_version": "global_local_strategy_return_v1_report_only",
            "global_strategy_id": "S_global_event_route",
            "global_strategy_meaning": "prompt meaning = model interpretation = sampler route = latent evolution = visible video outcome",
            "law": "Local collision formulas may unfold freely, but every local Strategy must return to the global StrategyCarrier before the next route stage receives data.",
            "main_route_order": [
                "prompt_image_anchor",
                "positive_negative_prompt_polarity",
                "image_latent_noise_seed",
                "high_low_sampler_strategy",
                "object_relation_ontology",
                "tail_next_source",
                "previous_next_frame_motion",
                "visible_video_outcome",
            ],
            "sub_strategy_parent_map": {
                "prompt_image_anchor": "S_global_prompt_source_anchor",
                "positive_negative_prompt_polarity": "S_global_prompt_corridor",
                "image_latent_noise_seed": "S_global_source_to_latent_anchor",
                "high_low_sampler_strategy": "S_global_sampler_route",
                "object_relation_ontology": "S_global_object_relation",
                "tail_next_source": "S_global_cascade_continuation",
                "previous_next_frame_motion": "S_global_visible_motion",
                "visible_video_outcome": "S_global_visible_outcome",
            },
            "model_freedom_boundary": "The model remains free to solve the scene in its native vector space; this contract names return points, not a forced solver.",
            "active_control_allowed": False,
            "control_mode": "REPORT_ONLY",
        },
        "relation_vectors": relation_vectors,
        "semantic_density_context_map": semantic_density_context_map,
        "object_topology_map": object_topology_map,
        "object_relation_ontology": object_relation_ontology,
        "collision_points": collision_points,
        "model_language_transcode": model_language_transcode,
        "model_language_deconstruction": model_language_transcode,
        "strategy_control_packet": {
            "default_policy": "observe_and_propose_only",
            "prompt_rewrite_allowed": False,
            "prompt_substitution_allowed": False,
            "prompt_transcode_available": True,
            "prompt_transcode_semantic_map_only": True,
            "prompt_text_injection_allowed": False,
            "semantic_density_context_map_available": True,
            "model_language_candidate_available": False,
            "model_language_candidate_auto_apply": False,
            "sampler_mutation_allowed": False,
            "public_safe_default": {
                "math_control_mode": "OBSERVE_ONLY",
                "high_delta_strength": 1.0,
                "low_delta_strength": 1.0,
            },
            "research_candidates": {
                "math_control_mode": "STRATEGY_PRESSURE_WINDOW for one bounded functional test after OBSERVE_ONLY baseline",
                "high_delta_strength": high_candidates,
                "low_delta_strength": low_candidates,
                "sampler_trace_mode": "SHADOW_STEP_TRACE if relation evidence is needed",
                "sampler_trace_max_steps": 24 if collapse_risk < 0.55 else 64,
            },
            "activation_rule": "Only enable active control after fixed-seed visual evidence proves that the relation improves without source-anchor collapse.",
        },
        "next_route": (
            "Use this packet to compare prompt intention against Strategy Matrix, sampler seam metrics, tail choice, "
            "and visible video review before adding bounded active math."
        ),
    }
