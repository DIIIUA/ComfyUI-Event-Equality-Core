import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const SINGULARITY_DEBUG_UI = false;
const singularityDebug = (...args) => {
    if (SINGULARITY_DEBUG_UI) console.log(...args);
};

const CLEAN_NODE_NAMES = new Set([
    "Singularity",
    "SingularityCascadeSimple",
]);
const LEGACY_NODE_NAMES = new Set([]);
const SINGULARITY_VISIBLE_NODE_TITLE = "Singularity R113";

const UI = {
    minWidth: 720,
    minHeight: 760,
    maxHeight: 1240,  // safety clamp: long prompt schedules must not stretch the node forever.
    promptHeight: 180,
    nodeBottomPad: 28,
    marginX: 6,
    bandPadY: 4,
};

const PAUSE_OVERLAY = {
    gap: 10,
    maxWidth: 1180,
    minWidth: 940,
    pad: 8,
    tileHeight: 260,
    buttonHeight: 38,
    nodeWidthScale: 1.35,
};

const COMFY_GLOBAL_BLOCKING_OVERLAY_SELECTORS = [
    ".p-dialog-mask",
    ".p-dialog",
    ".p-overlaypanel",
    ".p-popover",
    ".comfyui-popup.open",
    ".comfy-modal",
    ".comfy-modal-content",
    ".comfyui-dialog",
    ".comfyui-modal",
    ".manager-dialog",
    ".extension-manager",
    "[role='dialog']",
    "[aria-modal='true']",
];

const COMFY_OCCLUDING_SURFACE_SELECTORS = [
    ".p-sidebar",
    ".p-sidebar-content",
    ".p-drawer",
    ".p-drawer-content",
    ".p-drawer-mask",
    ".comfy-menu",
    ".comfy-menu-container",
    ".comfy-side-bar",
    ".comfy-sidebar",
    ".comfyui-sidebar",
    ".workflow-panel",
    ".workflows-panel",
    ".workflow-sidebar",
    ".workflows-sidebar",
    ".workflow-browser",
    ".workflow-explorer",
    "[class*='drawer']",
    "[class*='Drawer']",
    "[class*='sidebar']",
    "[class*='Sidebar']",
    "[class*='side-panel']",
    "[class*='SidePanel']",
];

const LATENT_PREVIEW = {
    width: 220,
    maxHeight: 132,
    rowHeight: 146,
};

const PAUSE_EVENT = "singularity_cascade_paused";
const CONTINUE_ROUTE = "/singularity/cascade/continue/";
const CANCEL_ROUTE = "/singularity/cascade/cancel/";
const CANCEL_ALL_ROUTE = "/singularity/cascade/cancel";
const STATUS_ROUTE = "/singularity/cascade/status/";
const STATUS_POLL_MS = 1200;
const PROMPT_WIDGET_NAMES = new Set(["positive_prompt", "negative_prompt"]);
const PUBLIC_HIDDEN_WIDGET_NAMES = new Set([]);
const SUPPRESSED_MEDIA_WIDGET_NAMES = new Set([
    "$$canvas-image-preview",
    "vhslatentpreview",
    "videopreview",
    "imagepreview",
]);
const SUPPRESSED_MEDIA_WIDGET_TYPES = new Set([
    "preview",
    "video",
    "IMAGE",
]);

const GROUPS = [
    { id: "SOURCE", color: "#1d4ed8", bg: "rgba(30, 64, 175, 0.12)", names: ["source_image_file"] },
    { id: "PROMPT", color: "#7c3aed", bg: "rgba(91, 33, 182, 0.10)", names: ["positive_prompt", "negative_prompt", "temporal_texture_lock"] },
    { id: "CASCADE", color: "#0891b2", bg: "rgba(8, 145, 178, 0.10)", names: ["cascade_count", "pause_after_cascade_1", "pause_after_cascade_2", "pause_after_cascade_3", "pause_after_cascade_4", "frames_per_cascade", "width", "height", "fps", "seed"] },
    { id: "SAMPLING", color: "#ca8a04", bg: "rgba(202, 138, 4, 0.09)", names: ["sampler_name", "scheduler", "global_steps", "primary_cfg", "secondary_cfg", "primary_start_step", "primary_end_step", "secondary_start_step", "secondary_end_step", "math_control_mode", "high_delta_strength", "low_delta_strength", "strategy_field_mode"] },
    { id: "DECODE", color: "#059669", bg: "rgba(5, 150, 105, 0.09)", names: ["decode_tile_size", "decode_overlap", "decode_temporal_size", "decode_temporal_overlap", "image_upscale_method", "image_crop"] },
    { id: "OUTPUT", color: "#dc2626", bg: "rgba(220, 38, 38, 0.09)", names: ["save_video", "video_format", "save_report", "save_prefix"] },
    { id: "RESEARCH", color: "#00aa00", bg: "rgba(0, 128, 0, 0.08)", names: ["sampler_trace_mode", "sampler_trace_max_steps", "use_formula_recommendation", "prompt_transcode_mode", "auto_calibration_mode", "bridge_wan_alpha", "bridge_concat_alpha", "bridge_wan_max_step", "bridge_concat_max_step", "selected_tail_index"] },
];
const TAIL_UI_SLOT_COUNT = 5;

function isSingularityNode(node) {
    const candidates = [
        node?.comfyClass,
        node?.type,
        node?.title,
        node?.constructor?.comfyClass,
        node?.constructor?.type,
        node?.constructor?.name,
        node?.properties?.["Node name for S&R"],
    ];
    return candidates.some((value) => CLEAN_NODE_NAMES.has(String(value || "")));
}

function findWidget(node, name) {
    return node?.widgets?.find((w) => w?.name === name);
}

function setWidgetValue(node, name, value) {
    const widget = findWidget(node, name);
    if (!widget) return false;
    widget.value = value;
    if (widget.callback) widget.callback(value);
    return true;
}

function applySingularityVisibleTitle(node, nodeData = null) {
    if (!node) return;
    const mappedTitle = String(nodeData?.display_name || nodeData?.displayName || "").trim();
    const title = mappedTitle && mappedTitle !== "Singularity"
        ? mappedTitle
        : SINGULARITY_VISIBLE_NODE_TITLE;
    if (title) node.title = title;
}

function normalizePromptTranscodeMode(value) {
    const text = String(value ?? "").trim().toUpperCase();
    if (text === "1" || text === "TRANSFORM_PROMPT" || text === "TRANSFORM_STRUCTURED_PROMPT" || text === "APPEND_TRANSCODE" || text === "APPEND_STRUCTURED_TRANSCODE") {
        return "TRANSFORM_PROMPT";
    }
    return "REPORT_ONLY";
}

function normalizeMathControlMode(value) {
    const text = String(value ?? "").trim().toUpperCase();
    if (text === "LATENT_DELTA_SCALE") return "LATENT_DELTA_SCALE";
    if (text === "STRATEGY_PRESSURE_WINDOW") return "STRATEGY_PRESSURE_WINDOW";
    if (text === "LATENT_MEMORY_BRIDGE") return "LATENT_MEMORY_BRIDGE";
    if (text === "DEEP_STEP_DELTA_CONTROL") return "DEEP_STEP_DELTA_CONTROL";
    return "OBSERVE_ONLY";
}

function sanitizeMathControlWidget(node) {
    const widget = findWidget(node, "math_control_mode");
    if (!widget) return;
    widget.options = widget.options || {};
    widget.options.values = [
        "OBSERVE_ONLY",
        "LATENT_DELTA_SCALE",
        "STRATEGY_PRESSURE_WINDOW",
        "LATENT_MEMORY_BRIDGE",
        "DEEP_STEP_DELTA_CONTROL",
    ];
    widget.value = normalizeMathControlMode(widget.value);
}

function sanitizePromptTranscodeWidget(node) {
    const widget = findWidget(node, "prompt_transcode_mode");
    if (!widget) return;
    widget.options = widget.options || {};
    widget.options.values = ["REPORT_ONLY", "TRANSFORM_PROMPT"];
    widget.value = normalizePromptTranscodeMode(widget.value);
}

function clampNodeSize(node) {
    if (!node || !node.setSize) return;
    const width = Math.max(Number(node.size?.[0]) || UI.minWidth, UI.minWidth);
    const visualHeight = getNodeVisualHeight(node);
    let height = Math.max(UI.minHeight, visualHeight);
    height = Math.min(height, UI.maxHeight);
    if (width !== node.size?.[0] || height !== node.size?.[1]) {
        node.setSize([width, height]);
    }
}

function stabilizePromptWidgets(node) {
    const widgets = node?.widgets || [];
    for (const widget of widgets) {
        if (!widget || !PROMPT_WIDGET_NAMES.has(widget.name)) continue;
        widget.options = widget.options || {};
        widget.options.height = UI.promptHeight;
        widget.computeSize = function(width) {
            return [Math.max(Number(width) || UI.minWidth, 320), UI.promptHeight];
        };
    }
}

function hidePublicOnlyWidgets(node) {
    const widgets = node?.widgets || [];
    for (const widget of widgets) {
        if (!widget || !PUBLIC_HIDDEN_WIDGET_NAMES.has(widget.name)) continue;
        widget.value = false;
        widget.label = "";
        widget.hidden = true;
        widget.computeSize = function() { return [0, -6]; };
    }
}

const R113_DRIFT_REPAIR_DEFAULTS = {
    cascade_count: 2,
    pause_after_cascade_1: true,
    pause_after_cascade_2: false,
    pause_after_cascade_3: false,
    pause_after_cascade_4: false,
    frames_per_cascade: 49,
    width: 416,
    height: 608,
    fps: 16,
    seed: 123,
    sampler_name: "euler",
    scheduler: "simple",
    global_steps: 4,
    primary_cfg: 1.0,
    secondary_cfg: 1.0,
    primary_start_step: 0,
    primary_end_step: 1,
    secondary_start_step: 1,
    secondary_end_step: 4,
    math_control_mode: "OBSERVE_ONLY",
    high_delta_strength: 1.0,
    low_delta_strength: 1.0,
    strategy_field_mode: "REPORT_ONLY",
    decode_tile_size: 512,
    decode_overlap: 64,
    decode_temporal_size: 32,
    decode_temporal_overlap: 12,
    image_upscale_method: "nearest-exact",
    image_crop: "disabled",
    save_video: true,
    video_format: "video/h264-mp4",
    save_report: true,
    save_prefix: "Singularity",
    sampler_trace_mode: "OFF",
    sampler_trace_max_steps: 64,
    use_formula_recommendation: false,
    prompt_transcode_mode: "REPORT_ONLY",
    auto_calibration_mode: "OFF",
    bridge_wan_alpha: 0.10,
    bridge_concat_alpha: 0.06,
    bridge_wan_max_step: 0.45,
    bridge_concat_max_step: 0.28,
};

function numericWidgetValue(node, name) {
    const widget = findWidget(node, name);
    return Number(widget?.value);
}

function textWidgetValue(node, name) {
    const widget = findWidget(node, name);
    return String(widget?.value ?? "");
}

function repairSevereWidgetDrift(node) {
    if (!node?.widgets?.length || node._singularityR113DriftRepaired) return;
    const symptoms = [];
    const cascadeCount = numericWidgetValue(node, "cascade_count");
    const frames = numericWidgetValue(node, "frames_per_cascade");
    const width = numericWidgetValue(node, "width");
    const height = numericWidgetValue(node, "height");
    const fps = numericWidgetValue(node, "fps");
    const globalSteps = numericWidgetValue(node, "global_steps");
    const decodeTile = numericWidgetValue(node, "decode_tile_size");
    const decodeOverlap = numericWidgetValue(node, "decode_overlap");
    const samplerName = textWidgetValue(node, "sampler_name").trim();
    const videoFormat = textWidgetValue(node, "video_format").trim();
    const highDelta = numericWidgetValue(node, "high_delta_strength");

    if (!Number.isFinite(cascadeCount) || cascadeCount < 1) symptoms.push("cascade_count");
    if (!Number.isFinite(frames) || frames < 1) symptoms.push("frames_per_cascade");
    if (!Number.isFinite(width) || width < 64) symptoms.push("width");
    if (!Number.isFinite(height) || height < 64) symptoms.push("height");
    if (!Number.isFinite(fps) || fps < 1 || fps > 120) symptoms.push("fps");
    if (!Number.isFinite(globalSteps) || globalSteps < 1) symptoms.push("global_steps");
    if (!Number.isFinite(decodeTile) || decodeTile < 64) symptoms.push("decode_tile_size");
    if (!Number.isFinite(decodeOverlap)) symptoms.push("decode_overlap");
    if (!samplerName || Number.isFinite(Number(samplerName))) symptoms.push("sampler_name");
    if (!videoFormat.includes("/")) symptoms.push("video_format");
    if (!Number.isFinite(highDelta)) symptoms.push("high_delta_strength");

    if (symptoms.length < 4) return;

    for (const [name, value] of Object.entries(R113_DRIFT_REPAIR_DEFAULTS)) {
        setWidgetValue(node, name, value);
    }
    node._singularityR113DriftRepaired = true;
    node.properties = node.properties || {};
    node.properties.singularity_r113_widget_drift_repaired = {
        reason: "severe_positional_widget_value_drift",
        symptoms,
        repaired_at: new Date().toISOString(),
    };
    console.warn("[Singularity UI] R113 repaired severe positional widget drift", symptoms);
}

function stabilizeNodeLayout(node) {
    repairSevereWidgetDrift(node);
    hidePublicOnlyWidgets(node);
    sanitizePromptTranscodeWidget(node);
    sanitizeMathControlWidget(node);
    stabilizePromptWidgets(node);
    clampNodeSize(node);
}

function getWidgetContentBottom(node) {
    const widgets = node?.widgets || [];
    let fallbackY = 60;
    let bottom = 0;
    for (const widget of widgets) {
        if (!widget || widget.hidden) continue;
        const rawY = Number.isFinite(Number(widget.last_y)) ? Number(widget.last_y) :
            (Number.isFinite(Number(widget.y)) ? Number(widget.y) : fallbackY);
        let height = 22;
        if (PROMPT_WIDGET_NAMES.has(widget.name)) {
            height = UI.promptHeight;
        } else if (widget.options && Number.isFinite(Number(widget.options.height))) {
            height = Math.max(height, Number(widget.options.height));
        }
        try {
            if (typeof widget.computeSize === "function") {
                const size = widget.computeSize(node.size?.[0] || UI.minWidth);
                if (Array.isArray(size) && Number.isFinite(Number(size[1]))) {
                    const computedHeight = Number(size[1]);
                    height = PROMPT_WIDGET_NAMES.has(widget.name)
                        ? UI.promptHeight
                        : Math.max(height, Math.min(computedHeight, 260));
                }
            }
        } catch (e) {}
        if (height < 0) height = 0;
        bottom = Math.max(bottom, rawY + height);
        fallbackY = rawY + height + 4;
    }
    return Math.max(100, bottom + 8);
}

function getNodeVisualHeight(node) {
    let height = Number(node?.size?.[1]) || UI.minHeight;
    try {
        height = Math.max(height, getWidgetContentBottom(node) + UI.nodeBottomPad);
    } catch (e) {}
    return Math.max(UI.minHeight, height);
}

function viewUrlFromMediaInfo(item, fallbackType = "temp") {
    if (!item?.filename) return "";
    let url = "/view?filename=" + encodeURIComponent(item.filename) +
        "&type=" + encodeURIComponent(item.type || fallbackType) +
        "&subfolder=" + encodeURIComponent(item.subfolder || "");
    if (item.format) {
        url += "&format=" + encodeURIComponent(item.format);
    }
    return api.apiURL(url);
}

function createMediaElementFromInfo(item) {
    const url = viewUrlFromMediaInfo(item, item?.type || "temp");
    if (!url) return null;
    const filename = String(item?.filename || "").toLowerCase();
    const format = String(item?.format || item?.mime || "").toLowerCase();
    const isVideo = format.includes("video") || filename.endsWith(".mp4") || filename.endsWith(".mov") || filename.endsWith(".mkv") || filename.endsWith(".webm");
    const el = isVideo ? document.createElement("video") : new Image();
    el.src = url;
    if (isVideo) {
        el.autoplay = true;
        el.loop = true;
        el.muted = true;
        el.playsInline = true;
        el.addEventListener("loadeddata", () => {
            try { el.play?.(); } catch (e) {}
        }, { once: true });
        try { el.play?.(); } catch (e) {}
    }
    el.__singularityMediaFilename = item.filename;
    return el;
}

function loadPauseFrames(node, pauseFrames) {
    node._singularityPauseImgs = [];
    for (const item of pauseFrames || []) {
        const img = new Image();
        img.src = viewUrlFromMediaInfo(item, "temp");
        img.__singularityResumeIndex = Number(item.resume_index ?? -1);
        node._singularityPauseImgs.push(img);
    }
}

function loadPausePreviewMedia(node, detail) {
    const item = detail?.preview_video || detail?.stitched_preview || detail?.stitched_preview_video;
    if (!item?.filename) {
        node._SingularityPausePreviewVideoCache = null;
        return;
    }
    const media = createMediaElementFromInfo(item);
    node._SingularityPausePreviewVideoCache = media ? {
        filename: item.filename,
        video: media,
        info: item,
    } : null;
}

function chooseDefaultResumeIndex(node, detail) {
    const explicit = Number(detail?.default_resume_frame_index ?? -1);
    if (explicit > 0) return explicit;
    const imgs = node._singularityPauseImgs || [];
    for (let i = imgs.length - 1; i >= 0; i--) {
        const idx = Number(imgs[i].__singularityResumeIndex ?? -1);
        if (idx > 0) return idx;
    }
    return -1;
}

function setElementStyle(el, styles) {
    if (!el) return;
    for (const [key, value] of Object.entries(styles)) {
        el.style[key] = value;
    }
}

function stopOverlayEvent(event) {
    event?.stopPropagation?.();
    if (event?.type === "contextmenu") {
        event.preventDefault?.();
    }
}

function inputImageUrlFromValue(value) {
    if (!value || String(value).toLowerCase() === "none") return "";
    const normalized = String(value).replace(/\\/g, "/");
    const parts = normalized.split("/");
    const filename = parts.pop() || "";
    const subfolder = parts.join("/");
    if (!filename) return "";
    return api.apiURL(
        "/view?filename=" + encodeURIComponent(filename) +
        "&type=input&subfolder=" + encodeURIComponent(subfolder)
    );
}

function getSourceImageUrl(node) {
    return inputImageUrlFromValue(findWidget(node, "source_image_file")?.value);
}

function clearRunMedia(node, keepSource = true) {
    if (!node) return;
    node._singularityPauseImgs = [];
    node._singularitySelectedTailIndex = -1;
    node._singularityResumeFrameIndex = -1;
    node._singularityPauseNodeId = "";
    node._singularityPaused = false;
    node._singularityContinuePending = false;
    node._SingularityResultVideoCache = null;
    node._SingularityPausePreviewVideoCache = null;
    node._singularityPrivateImgs = [];
    node._singularityMediaOverlayRenderedKey = "";
    node._singularityPauseStatusKey = "";
    if (!keepSource) {
        node._singularitySourceImageUrl = "";
    }
}

function resetPausedRunUi(node, keepSource = true) {
    if (!node) return;
    clearRunMedia(node, keepSource);
    cleanupStalePauseOverlays(node, node._singularityPauseOverlay || null);
    renderPauseOverlay(node, true);
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    else app.graph?.setDirtyCanvas?.(true, true);
}

function getSingularityNodes() {
    return (app.graph?._nodes || []).filter((node) => isSingularityNode(node));
}

async function postCascadeCancel(node = null) {
    try {
        const pauseNodeId = node?._singularityPauseNodeId || node?.id;
        const route = pauseNodeId
            ? CANCEL_ROUTE + encodeURIComponent(String(pauseNodeId))
            : CANCEL_ALL_ROUTE;
        await api.fetchApi(route, { method: "POST" });
    } catch (error) {
        console.warn("[Singularity UI] Cascade cancel request failed", error);
    }
}

async function requestCascadeCancel(node) {
    await postCascadeCancel(node);
    resetPausedRunUi(node, true);
}

function cancelAllPausedRunsLocally() {
    for (const node of getSingularityNodes()) {
        if (node._singularityPaused || node._singularityPauseImgs?.length || node._singularityContinuePending || node._SingularityResultVideoCache?.video || node._SingularityPausePreviewVideoCache?.video) {
            resetPausedRunUi(node, true);
        }
    }
    cleanupStalePauseOverlays(null);
}

function installApiInterruptGuard() {
    if (api._singularityInterruptGuardInstalled || typeof api.interrupt !== "function") return;
    const originalInterrupt = api.interrupt;
    api.interrupt = function() {
        postCascadeCancel(null);
        cancelAllPausedRunsLocally();
        return originalInterrupt.apply(this, arguments);
    };
    api._singularityInterruptGuardInstalled = true;
}

function installSourceWidgetWatcher(node) {
    const widget = findWidget(node, "source_image_file");
    if (!widget || widget._singularitySourceWatcherInstalled) return;
    const originalCallback = widget.callback;
    widget.callback = function(value) {
        const previousUrl = node._singularitySourceImageUrl || "";
        const nextUrl = inputImageUrlFromValue(value);
        if (previousUrl && previousUrl !== nextUrl) {
            clearRunMedia(node, true);
            cleanupStalePauseOverlays(node);
        }
        node._singularitySourceImageUrl = nextUrl;
        const r = originalCallback?.apply(this, arguments);
        renderPauseOverlay(node, true);
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
        else app.graph?.setDirtyCanvas?.(true, true);
        return r;
    };
    widget._singularitySourceWatcherInstalled = true;
    node._singularitySourceImageUrl = inputImageUrlFromValue(widget.value);
}

function installPrivatePreviewStore(node) {
    if (!node || node._singularityImgsSuppressed) return;
    const existingImgs = Array.isArray(node.imgs) ? node.imgs : [];
    node._singularityPrivateImgs = existingImgs;
    try {
        Object.defineProperty(node, "imgs", {
            get() {
                return undefined;
            },
            set(value) {
                node._singularityPrivateImgs = Array.isArray(value) ? value : (value ? [value] : []);
                node._singularityMediaOverlayRenderedKey = "";
                renderPauseOverlay(node, true);
            },
            configurable: true,
            enumerable: true,
        });
        Object.defineProperty(node, "videoContainer", {
            get() {
                return undefined;
            },
            set(value) {
                node._singularityPrivateVideoContainer = value;
            },
            configurable: true,
            enumerable: true,
        });
        node.imgs = existingImgs;
    } catch (e) {}
    node.imageIndex = null;
    node.overIndex = null;
    node.videoContainer = undefined;
    node._singularityImgsSuppressed = true;
}

function isSuppressedMediaWidget(widget) {
    if (!widget) return false;
    if (isCompactLatentPreviewWidget(widget)) return false;
    return SUPPRESSED_MEDIA_WIDGET_NAMES.has(widget.name) ||
        SUPPRESSED_MEDIA_WIDGET_TYPES.has(widget.type) ||
        widget.name?.toLowerCase?.().includes("preview") ||
        widget.type?.toLowerCase?.().includes("preview");
}

function isCompactLatentPreviewWidget(widget) {
    if (!widget) return false;
    return widget.name === "vhslatentpreview" || widget.type === "vhscanvas";
}

function compactLatentPreviewWidget(widget) {
    if (!widget) return;
    widget.hidden = false;
    widget.serialize = false;
    widget.computeSize = function(width) {
        return [Number(width) || UI.minWidth, LATENT_PREVIEW.rowHeight];
    };
    widget._singularityCompactLatentPreview = true;
    const el = widget.element;
    if (!el?.style) return;
    const aspect = Number(widget.aspectRatio || (el.width && el.height ? el.width / el.height : 0));
    const height = aspect > 0
        ? Math.min(LATENT_PREVIEW.maxHeight, Math.max(72, Math.round(LATENT_PREVIEW.width / aspect)))
        : LATENT_PREVIEW.maxHeight;
    el.hidden = false;
    setElementStyle(el, {
        display: "block",
        width: `${LATENT_PREVIEW.width}px`,
        height: `${height}px`,
        maxWidth: `${LATENT_PREVIEW.width}px`,
        maxHeight: `${LATENT_PREVIEW.maxHeight}px`,
        margin: "6px auto 8px auto",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "4px",
        background: "#101018",
        objectFit: "contain",
        overflow: "hidden",
    });
}

function suppressMediaWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.serialize = false;
    widget.computeSize = function() { return [0, -4]; };
    widget._singularitySuppressed = true;
    const elements = [
        widget.element,
        widget.parentEl,
        widget.videoEl,
        widget.imgEl,
    ];
    for (const el of elements) {
        if (!el?.style) continue;
        el.hidden = true;
        el.style.display = "none";
        el.style.width = "0px";
        el.style.height = "0px";
        el.style.maxHeight = "0px";
        el.style.overflow = "hidden";
    }
}

function suppressNativeMediaWidgets(node) {
    const widgets = node?.widgets || [];
    for (let i = widgets.length - 1; i >= 0; i--) {
        const widget = widgets[i];
        if (isCompactLatentPreviewWidget(widget)) {
            compactLatentPreviewWidget(widget);
            continue;
        }
        if (isSuppressedMediaWidget(widget)) {
            suppressMediaWidget(widget);
        }
    }
}

function installMediaWidgetGuard(node) {
    if (!node || node._singularityMediaWidgetGuardInstalled || typeof node.addDOMWidget !== "function") return;
    const originalAddDOMWidget = node.addDOMWidget;
    node.addDOMWidget = function(name, type, element, options) {
        const widget = originalAddDOMWidget.apply(this, arguments);
        const added = { name, type, element };
        if (isSingularityNode(this) && isCompactLatentPreviewWidget(added)) {
            compactLatentPreviewWidget(widget);
            this._singularityMediaOverlayRenderedKey = "";
            renderPauseOverlay(this, true);
            stabilizeNodeLayout(this);
        } else if (isSingularityNode(this) && isSuppressedMediaWidget(added)) {
            suppressMediaWidget(widget);
            this._singularityMediaOverlayRenderedKey = "";
            renderPauseOverlay(this, true);
            stabilizeNodeLayout(this);
        }
        return widget;
    };
    node._singularityMediaWidgetGuardInstalled = true;
    suppressNativeMediaWidgets(node);
}

function removePauseOverlay(node) {
    const overlay = node?._singularityPauseOverlay;
    if (node?._singularityPauseOverlayTicker) {
        cancelAnimationFrame(node._singularityPauseOverlayTicker);
        node._singularityPauseOverlayTicker = 0;
    }
    if (overlay?.parentElement) {
        overlay.parentElement.removeChild(overlay);
    }
    if (node) {
        node._singularityPauseOverlay = null;
        node._singularityPauseOverlayRenderedKey = "";
        node._singularityMediaOverlayRenderedKey = "";
        cleanupStalePauseOverlays(node);
    }
}

function getActiveSingularityNodeIds() {
    const ids = new Set();
    const nodes = app.graph?._nodes || [];
    for (const graphNode of nodes) {
        if (!isSingularityNode(graphNode)) continue;
        if (graphNode.id !== undefined && graphNode.id !== null) {
            ids.add(String(graphNode.id));
        }
    }
    return ids;
}

function cleanupStalePauseOverlays(node, keepOverlay = null) {
    if (typeof document === "undefined") return;
    const overlays = Array.from(document.querySelectorAll(".singularity-media-overlay"));
    if (!overlays.length) return;
    const currentId = node?.id !== undefined && node?.id !== null ? String(node.id) : "";
    const activeIds = getActiveSingularityNodeIds();
    for (const overlay of overlays) {
        if (keepOverlay && overlay === keepOverlay) continue;
        const overlayNodeId = String(overlay.dataset?.nodeId || "");
        const duplicateForCurrentNode = currentId && overlayNodeId === currentId;
        const orphanOverlay = !overlayNodeId || (activeIds.size > 0 && !activeIds.has(overlayNodeId));
        if (duplicateForCurrentNode || orphanOverlay) {
            overlay.remove();
        }
    }
}

function getNodeScreenPosition(node, graphX, graphY) {
    const canvas = app.canvas?.canvas;
    const ds = app.canvas?.ds;
    if (!canvas || !ds) return null;
    const rect = canvas.getBoundingClientRect();
    const scale = Number(ds.scale) || 1;
    const offset = ds.offset || [0, 0];
    return {
        x: rect.left + (graphX + Number(offset[0] || 0)) * scale,
        y: rect.top + (graphY + Number(offset[1] || 0)) * scale,
        scale,
    };
}

function rectsIntersect(a, b) {
    if (!a || !b) return false;
    return !(
        a.right <= b.left ||
        a.left >= b.right ||
        a.bottom <= b.top ||
        a.top >= b.bottom
    );
}

function isIgnorableBlockingElement(el, overlay) {
    if (!el || el === overlay || overlay?.contains?.(el) || el.closest?.(".singularity-media-overlay")) return true;
    const tagName = String(el.tagName || "").toUpperCase();
    if (tagName === "HTML" || tagName === "BODY" || tagName === "CANVAS") return true;
    if (el.id === "graph-canvas") return true;
    if (el.classList?.contains?.("litegraph") || el.classList?.contains?.("litegraphcanvas")) return true;
    return false;
}

function isVisibleBlockingElement(el, overlay, overlayRect = null, requireIntersection = false) {
    if (isIgnorableBlockingElement(el, overlay)) return false;
    const style = window.getComputedStyle ? window.getComputedStyle(el) : null;
    if (style) {
        if (style.display === "none" || style.visibility === "hidden") return false;
        if (Number(style.opacity) === 0) return false;
    }
    const rect = el.getBoundingClientRect?.();
    if (!rect || rect.width <= 1 || rect.height <= 1) return false;
    if (requireIntersection && overlayRect && !rectsIntersect(rect, overlayRect)) return false;
    return true;
}

function hasBlockingComfyOverlay(overlay) {
    if (typeof document === "undefined") return false;
    const overlayRect = overlay?.getBoundingClientRect?.();
    for (const selector of COMFY_GLOBAL_BLOCKING_OVERLAY_SELECTORS) {
        const elements = document.querySelectorAll(selector);
        for (const el of elements) {
            if (isVisibleBlockingElement(el, overlay, overlayRect, true)) return true;
        }
    }
    for (const selector of COMFY_OCCLUDING_SURFACE_SELECTORS) {
        const elements = document.querySelectorAll(selector);
        for (const el of elements) {
            if (isVisibleBlockingElement(el, overlay, overlayRect, true)) return true;
        }
    }
    return false;
}

function updatePauseOverlayVisibility(node) {
    const overlay = node?._singularityPauseOverlay;
    if (!overlay) return false;
    overlay.dataset.comfyBlocked = "false";
    overlay.style.visibility = "visible";
    overlay.style.pointerEvents = "auto";
    overlay.style.zIndex = "10000";
    return false;
}

function positionPauseOverlay(node) {
    const overlay = node?._singularityPauseOverlay;
    if (!overlay || !node?.pos || !node?.size) return;
    if (updatePauseOverlayVisibility(node)) return;
    const nodeWidth = Math.max(Number(node.size[0]) || UI.minWidth, UI.minWidth);
    const visualHeight = getNodeVisualHeight(node);
    const baseWidth = Math.max(
        PAUSE_OVERLAY.minWidth,
        Math.min(Math.round(nodeWidth * PAUSE_OVERLAY.nodeWidthScale), PAUSE_OVERLAY.maxWidth)
    );
    const graphX = Number(node.pos[0] || 0) + (nodeWidth - baseWidth) / 2;
    const graphY = Number(node.pos[1] || 0) + visualHeight + PAUSE_OVERLAY.gap;
    const screen = getNodeScreenPosition(node, graphX, graphY);
    if (!screen) return;
    overlay.style.left = `${screen.x}px`;
    overlay.style.top = `${screen.y}px`;
    overlay.style.width = `${baseWidth}px`;
    overlay.style.transform = `scale(${screen.scale})`;
    overlay.style.transformOrigin = "0 0";
}

function startPauseOverlayTicker(node) {
    if (!node || node._singularityPauseOverlayTicker) return;
    const tick = () => {
        const active = Boolean(node);
        if (!active || !node._singularityPauseOverlay) {
            removePauseOverlay(node);
            return;
        }
        positionPauseOverlay(node);
        node._singularityPauseOverlayTicker = requestAnimationFrame(tick);
    };
    node._singularityPauseOverlayTicker = requestAnimationFrame(tick);
}

function selectPauseFrame(node, index) {
    if (!node?._singularityPaused) return;
    node._singularitySelectedTailIndex = index;
    const img = node._singularityPauseImgs?.[index];
    const resumeIndex = Number(img?.__singularityResumeIndex ?? -1);
    if (resumeIndex > 0) {
        node._singularityResumeFrameIndex = resumeIndex;
    }
    setWidgetValue(node, "selected_tail_index", index);
    renderPauseOverlay(node, true);
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    else app.graph?.setDirtyCanvas?.(true, true);
}

async function requestCascadeContinue(node) {
    if (!node?._singularityPaused) return;
    const resumeFrameIndex = Number(node._singularityResumeFrameIndex ?? -1);
    if (resumeFrameIndex < 1) {
        alert("Select a resume frame before continuing.");
        return;
    }
    node._singularityContinuePending = true;
    renderPauseOverlay(node, true);
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    try {
        const positivePromptWidget = findWidget(node, "positive_prompt");
        const negativePromptWidget = findWidget(node, "negative_prompt");
        const transcodeModeWidget = findWidget(node, "prompt_transcode_mode");
        const promptTranscodeMode = normalizePromptTranscodeMode(transcodeModeWidget?.value);
        if (transcodeModeWidget) {
            transcodeModeWidget.value = promptTranscodeMode;
        }
        await api.fetchApi(CONTINUE_ROUTE + encodeURIComponent(String(node._singularityPauseNodeId || node.id)), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                resume_frame_index: resumeFrameIndex,
                prompt_payload_version: "cascade_continue_prompt_v1",
                prompt_source: "node_widgets_at_continue_click",
                positive_prompt: String(positivePromptWidget?.value ?? ""),
                negative_prompt: String(negativePromptWidget?.value ?? ""),
                prompt_transcode_mode: promptTranscodeMode,
            }),
        });
    } catch (error) {
        console.error("[Singularity UI] Continue request failed", error);
        node._singularityContinuePending = false;
        renderPauseOverlay(node, true);
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
        alert("Continue request failed. Check the ComfyUI console.");
    }
}

function getPauseOverlayRenderKey(node) {
    const imgs = node?._singularityPauseImgs || [];
    const parts = [
        String(getSourceImageUrl(node) || ""),
        String(node?._SingularityResultVideoCache?.filename || ""),
        String(node?._SingularityPausePreviewVideoCache?.filename || ""),
        String(Boolean(node?._singularityPaused)),
        String(Boolean(node?._singularityContinuePending)),
        String(node?._singularitySelectedTailIndex ?? -1),
        String(node?._singularityResumeFrameIndex ?? -1),
        String(findWidget(node, "use_formula_recommendation")?.value ?? false),
    ];
    for (const img of imgs) {
        parts.push(String(img?.src || ""));
        parts.push(String(img?.__singularityResumeIndex ?? -1));
    }
    return parts.join("|");
}

function ensurePauseOverlay(node) {
    if (!node) return null;
    let overlay = node._singularityPauseOverlay;
    if (overlay?.parentElement) {
        overlay.dataset.nodeId = String(node.id ?? "");
        cleanupStalePauseOverlays(node, overlay);
        return overlay;
    }

    cleanupStalePauseOverlays(node);
    overlay = document.createElement("div");
    overlay.className = "singularity-media-overlay";
    overlay.dataset.nodeId = String(node.id ?? "");
    setElementStyle(overlay, {
        position: "fixed",
        zIndex: "10000",
        pointerEvents: "auto",
        boxSizing: "border-box",
        padding: `${PAUSE_OVERLAY.pad}px`,
        background: "rgba(4, 4, 10, 0.96)",
        border: "1px solid #00aa00",
        borderRadius: "6px",
        boxShadow: "0 10px 28px rgba(0,0,0,0.45)",
        color: "#ddd",
        fontFamily: "Arial, sans-serif",
        userSelect: "none",
    });
    for (const eventName of ["pointerdown", "mousedown", "mouseup", "click", "dblclick", "contextmenu"]) {
        overlay.addEventListener(eventName, stopOverlayEvent);
    }
    document.body.appendChild(overlay);
    node._singularityPauseOverlay = overlay;
    return overlay;
}

function appendMediaTile(grid, options) {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.title = options.title || options.label || "";
    setElementStyle(tile, {
        height: `${PAUSE_OVERLAY.tileHeight}px`,
        minWidth: "0",
        border: options.selected ? "3px solid #00ff66" : (options.advised ? "2px solid #ffdd66" : "1px solid rgba(255,255,255,0.18)"),
        borderRadius: "4px",
        background: options.advised ? "rgba(255, 215, 0, 0.16)" : "rgba(255,255,255,0.04)",
        padding: "4px",
        cursor: options.onClick ? "pointer" : "default",
        overflow: "hidden",
        position: "relative",
    });
    if (options.onClick) {
        tile.addEventListener("click", (event) => {
            stopOverlayEvent(event);
            options.onClick();
        });
    }

    const media = options.media;
    if (media) {
        let el = null;
        if (typeof media === "string") {
            el = document.createElement("img");
            el.src = media;
            el.alt = options.label || "";
        } else if (media instanceof HTMLVideoElement) {
            el = media;
            el.autoplay = true;
            el.loop = true;
            el.muted = true;
            el.playsInline = true;
            try { el.play?.(); } catch (e) {}
        } else if (media instanceof HTMLImageElement) {
            el = media;
        }
        if (el) {
            setElementStyle(el, {
                width: "100%",
                height: "100%",
                objectFit: "contain",
                display: "block",
                background: "#fff",
                pointerEvents: "none",
            });
            tile.appendChild(el);
        }
    } else if (options.placeholder) {
        const placeholder = document.createElement("div");
        setElementStyle(placeholder, {
            width: "100%",
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "rgba(255,255,255,0.24)",
            background: "rgba(255,255,255,0.035)",
            fontSize: "11px",
            pointerEvents: "none",
        });
        placeholder.textContent = options.placeholder;
        tile.appendChild(placeholder);
    }

    const label = document.createElement("div");
    label.textContent = options.label || "";
    setElementStyle(label, {
        position: "absolute",
        left: "4px",
        right: "4px",
        bottom: "3px",
        height: "14px",
        lineHeight: "14px",
        fontSize: "10px",
        textAlign: "center",
        color: options.selected ? "#00ff66" : (options.advised ? "#ffdd66" : "#ddd"),
        background: "rgba(0,0,0,0.58)",
        borderRadius: "2px",
        pointerEvents: "none",
    });
    tile.appendChild(label);
    grid.appendChild(tile);
}

function renderPauseOverlay(node, force = false) {
    const sourceUrl = getSourceImageUrl(node);
    const active = Boolean(node);
    if (!active) {
        removePauseOverlay(node);
        return;
    }

    const overlay = ensurePauseOverlay(node);
    if (!overlay) return;
    const renderKey = getPauseOverlayRenderKey(node);
    if (!force && node._singularityMediaOverlayRenderedKey === renderKey) {
        positionPauseOverlay(node);
        startPauseOverlayTicker(node);
        return;
    }
    node._singularityMediaOverlayRenderedKey = renderKey;

    overlay.innerHTML = "";

    const grid = document.createElement("div");
    setElementStyle(grid, {
        display: "grid",
        gridTemplateColumns: "repeat(5, minmax(0, 1fr))",
        gap: "8px",
        marginBottom: "8px",
    });

    const useFormula = Boolean(findWidget(node, "use_formula_recommendation")?.value ?? false);
    const selected = Number(node._singularitySelectedTailIndex ?? -1);
    const formulaBest = Number(node._singularityFormulaBestIndex ?? 0);
    const imgs = node._singularityPauseImgs || [];
    const tailCount = TAIL_UI_SLOT_COUNT;

    appendMediaTile(grid, {
        label: "Source",
        title: "Current source image",
        media: sourceUrl,
    });

    grid.style.gridTemplateColumns = `repeat(${tailCount + 2}, minmax(0, 1fr))`;

    for (let i = 0; i < tailCount; i++) {
        const img = imgs[i];
        const resumeIndex = Number(img?.__singularityResumeIndex ?? -1);
        const selectedSlot = node._singularityPaused && i === selected;
        const formulaSlot = useFormula && i === formulaBest;
        appendMediaTile(grid, {
            label: resumeIndex > 0 ? `Frame ${resumeIndex}` : `Tail ${i + 1}`,
            title: resumeIndex > 0 ? `Tail candidate frame ${resumeIndex}` : `Tail candidate ${i + 1}`,
            media: img,
            placeholder: img ? "" : "Waiting",
            selected: selectedSlot,
            advised: formulaSlot,
            onClick: node._singularityPaused && img ? () => selectPauseFrame(node, i) : null,
        });
    }

    const resultMedia = node._SingularityResultVideoCache?.video || node._SingularityPausePreviewVideoCache?.video || null;
    const hasFinalResult = Boolean(node._SingularityResultVideoCache?.video);
    appendMediaTile(grid, {
        label: hasFinalResult ? "Result" : (node._SingularityPausePreviewVideoCache?.video ? "Preview" : "Result"),
        title: hasFinalResult ? "Saved result video" : "Stitched preview from start to this cascade boundary",
        media: resultMedia,
    });

    const buttonRow = document.createElement("div");
    setElementStyle(buttonRow, {
        display: "grid",
        gridTemplateColumns: "1fr 180px",
        gap: "8px",
    });

    const continueButton = document.createElement("button");
    continueButton.type = "button";
    continueButton.textContent = node._singularityContinuePending ? "Continuing..." : "Resume Cascade / Continue";
    continueButton.disabled = !node._singularityPaused || Boolean(node._singularityContinuePending);
    setElementStyle(continueButton, {
        width: "100%",
        height: `${PAUSE_OVERLAY.buttonHeight}px`,
        border: "1px solid #00aa00",
        borderRadius: "4px",
        background: node._singularityContinuePending ? "rgba(52,52,52,0.95)" : "rgba(16, 35, 21, 0.96)",
        color: "#00ff99",
        fontWeight: "700",
        fontSize: "12px",
        cursor: continueButton.disabled ? "default" : "pointer",
    });
    continueButton.addEventListener("click", (event) => {
        stopOverlayEvent(event);
        requestCascadeContinue(node);
    });

    const cancelButton = document.createElement("button");
    cancelButton.type = "button";
    cancelButton.textContent = "Cancel Pause";
    cancelButton.disabled = !node._singularityPaused && !node._singularityContinuePending;
    setElementStyle(cancelButton, {
        width: "100%",
        height: `${PAUSE_OVERLAY.buttonHeight}px`,
        border: "1px solid #aa3333",
        borderRadius: "4px",
        background: "rgba(48, 18, 18, 0.96)",
        color: "#ff9999",
        fontWeight: "700",
        fontSize: "12px",
        cursor: cancelButton.disabled ? "default" : "pointer",
    });
    cancelButton.addEventListener("click", (event) => {
        stopOverlayEvent(event);
        requestCascadeCancel(node);
    });
    buttonRow.appendChild(continueButton);
    buttonRow.appendChild(cancelButton);

    overlay.appendChild(grid);
    if (node._singularityPaused) {
        overlay.appendChild(buttonRow);
    }
    positionPauseOverlay(node);
    startPauseOverlayTicker(node);
}

function showPauseState(node, detail) {
    if (!node || !isSingularityNode(node)) return;
    cleanupStalePauseOverlays(node, node._singularityPauseOverlay || null);
    node._singularityPaused = true;
    node._singularityContinuePending = false;
    node._SingularityResultVideoCache = null;
    node._singularityPauseNodeId = String(detail?.node_id ?? node.id);
    node._singularitySelectedTailIndex = -1;
    loadPauseFrames(node, detail?.pause_frames || []);
    loadPausePreviewMedia(node, detail || {});

    const defaultResumeIndex = chooseDefaultResumeIndex(node, detail);
    node._singularityResumeFrameIndex = defaultResumeIndex;
    const imgs = node._singularityPauseImgs || [];
    const defaultSlot = imgs.findIndex((img) => Number(img.__singularityResumeIndex) === defaultResumeIndex);
    if (defaultSlot >= 0) {
        node._singularitySelectedTailIndex = defaultSlot;
        setWidgetValue(node, "selected_tail_index", defaultSlot);
    }

    stabilizeNodeLayout(node);
    renderPauseOverlay(node, true);
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    else app.graph?.setDirtyCanvas?.(true, true);
}

function pauseStatusKey(state) {
    if (!state || typeof state !== "object") return "";
    return [
        String(state.status || ""),
        String(state.segment_index ?? ""),
        String(state.updated_at ?? ""),
        String(state.default_resume_frame_index ?? state.resume_frame_index ?? ""),
        JSON.stringify(state.resume_candidates || []),
        JSON.stringify((state.pause_frames || []).map((item) => [
            item?.filename || "",
            item?.subfolder || "",
            item?.type || "",
            item?.resume_index ?? "",
        ])),
        String(state.preview_video?.filename || state.stitched_preview?.filename || ""),
    ].join("|");
}

function pauseDetailFromStatusPayload(payload, node) {
    const state = payload?.state || {};
    if (!state || typeof state !== "object") return null;
    const status = String(state.status || "");
    if (status !== "paused") return { status };
    return {
        status,
        node_id: String(payload?.node_id ?? node?.id ?? ""),
        segment_index: Number(state.segment_index ?? 1),
        pause_frames: state.pause_frames || [],
        resume_candidates: state.resume_candidates || [],
        default_resume_frame_index: Number(state.resume_frame_index ?? state.default_resume_frame_index ?? -1),
        preview_video: state.preview_video || state.stitched_preview || null,
        stitched_preview: state.preview_video || state.stitched_preview || null,
        __status_key: pauseStatusKey(state),
    };
}

async function pollPauseStatusForNode(node) {
    if (!node || !isSingularityNode(node) || node.id === undefined || node.id === null) return;
    if (node._singularityContinuePending) return;
    try {
        const response = await api.fetchApi(STATUS_ROUTE + encodeURIComponent(String(node.id)), {
            method: "GET",
            cache: "no-store",
        });
        if (!response?.ok) return;
        const payload = await response.json();
        const detail = pauseDetailFromStatusPayload(payload, node);
        if (!detail) return;
        if (detail.status === "paused") {
            if (!node._singularityPaused || node._singularityPauseStatusKey !== detail.__status_key) {
                node._singularityPauseStatusKey = detail.__status_key;
                singularityDebug("[Singularity UI] pause state recovered by status polling", detail);
                showPauseState(node, detail);
            } else {
                positionPauseOverlay(node);
            }
        } else if (detail.status === "cancelled") {
            resetPausedRunUi(node, true);
        }
    } catch (error) {
        console.warn("[Singularity UI] pause status polling failed", error);
    }
}

function pollPauseStatusesOnce() {
    for (const node of getSingularityNodes()) {
        pollPauseStatusForNode(node);
    }
}

function installPauseStatusPoller() {
    if (api._singularityPauseStatusPollerInstalled) return;
    api._singularityPauseStatusPollerInstalled = true;
    window.setInterval(pollPauseStatusesOnce, STATUS_POLL_MS);
    pollPauseStatusesOnce();
}

function clearPauseState(node) {
    if (!node || !isSingularityNode(node)) return;
    const keepTailFrames = Boolean(node._SingularityResultVideoCache?.video);
    node._singularityPaused = false;
    node._singularityContinuePending = false;
    node._singularityPauseStatusKey = "";
    if (node._SingularityResultVideoCache?.video) {
        node._SingularityPausePreviewVideoCache = null;
    }
    if (!keepTailFrames) {
        node._singularityPauseImgs = [];
    }
    node._singularityResumeFrameIndex = -1;
    renderPauseOverlay(node, true);
    if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    else app.graph?.setDirtyCanvas?.(true, true);
}

function groupForWidget(widget) {
    const name = widget?.name || "";
    for (const g of GROUPS) {
        if (g.names.includes(name)) return g;
    }
    return null;
}

function drawRoundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

function drawGroupBackgrounds(ctx, node) {
    if (!node.widgets || !node.widgets.length) return;

    ctx.save();
    const oldComposite = ctx.globalCompositeOperation;
    ctx.globalCompositeOperation = "destination-over";

    // Use inset width like the Tail 5 bar to reduce full-width colored group background noise.
    const fullW = node.size?.[0] || UI.minWidth;
    const contentMargin = 20;
    const width = Math.max(100, fullW - contentMargin * 2);
    const groupX = contentMargin;
    const widgets = node.widgets || [];
    let current = null;
    const runs = [];
    let fallbackY = 60;

    for (let i = 0; i < widgets.length; i++) {
        const w = widgets[i];
        if (!w || w.hidden) continue;
        const g = groupForWidget(w);
        if (!g) {
            if (current) { runs.push(current); current = null; }
            continue;
        }
        const y = Number(w.last_y ?? w.y ?? fallbackY);
        const h = 22;
        fallbackY = y + h + 2;

        if (!current || current.id !== g.id) {
            if (current) runs.push(current);
            current = { id: g.id, group: g, y1: y, y2: y + h };
        } else {
            current.y1 = Math.min(current.y1, y);
            current.y2 = Math.max(current.y2, y + h);
        }
    }
    if (current) runs.push(current);

    for (const run of runs) {
        const g = run.group;
        const y = Math.max(24, run.y1 - UI.bandPadY);
        const h = Math.max(18, (run.y2 - run.y1) + UI.bandPadY * 2);

        ctx.fillStyle = g.bg;
        drawRoundRect(ctx, groupX, y, width, h, 5);
        ctx.fill();

        ctx.strokeStyle = g.color;
        ctx.lineWidth = 1.2;
        drawRoundRect(ctx, groupX, y, width, h, 5);
        ctx.stroke();

        ctx.fillStyle = g.color;
        drawRoundRect(ctx, groupX + 2, y + 3, 3, Math.max(5, h - 6), 1);
        ctx.fill();

        ctx.fillStyle = "#ddd";
        ctx.font = "bold 8px sans-serif";
        ctx.fillText(g.id, groupX + 8, y + 11);
    }

    ctx.globalCompositeOperation = oldComposite;
    ctx.restore();
}

app.registerExtension({
    name: "event_horizon.ui.clean",

    setup() {
        installApiInterruptGuard();
        installPauseStatusPoller();
        api.addEventListener(PAUSE_EVENT, ({ detail }) => {
            const nodeId = String(detail?.node_id ?? "");
            const node = app.graph?.getNodeById?.(nodeId) || app.graph?.getNodeById?.(Number(nodeId));
            showPauseState(node, detail || {});
        });
        api.addEventListener("execution_start", () => {
            for (const node of getSingularityNodes()) {
                if (node._singularityPaused || node._singularityPauseImgs?.length || node._SingularityResultVideoCache?.video || node._SingularityPausePreviewVideoCache?.video) {
                    postCascadeCancel(node);
                    clearRunMedia(node, true);
                    cleanupStalePauseOverlays(node, node._singularityPauseOverlay || null);
                    renderPauseOverlay(node, true);
                }
            }
            cleanupStalePauseOverlays(null);
            pollPauseStatusesOnce();
        });
        for (const eventName of ["execution_interrupted", "execution_error"]) {
            api.addEventListener(eventName, () => {
                postCascadeCancel(null);
                cancelAllPausedRunsLocally();
            });
        }
    },

    loadedGraphNode(node) {
        if (!isSingularityNode(node)) return;
        installPrivatePreviewStore(node);
        installMediaWidgetGuard(node);
        installSourceWidgetWatcher(node);
        suppressNativeMediaWidgets(node);
        stabilizeNodeLayout(node);
        removePauseOverlay(node);
        for (let i = node.widgets?.length - 1 || -1; i >= 0; i--) {
            if (node.widgets[i] && node.widgets[i].name === "continue_cascade_btn") {
                node.widgets.splice(i, 1);
            }
        }
        node._singularityPaused = false;
        node._singularityContinuePending = false;
        node._singularityPauseImgs = [];
        node._singularityResumeFrameIndex = -1;
        renderPauseOverlay(node, true);
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const isClean = CLEAN_NODE_NAMES.has(nodeData?.name);
        const isLegacy = LEGACY_NODE_NAMES.has(nodeData?.name);
        singularityDebug(`[Singularity UI] beforeRegisterNodeDef for name=${nodeData?.name}, isClean=${isClean}`);
        if (!isClean && !isLegacy) return;

        // Custom File Upload Button ONLY
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = originalOnNodeCreated?.apply(this, arguments);
            singularityDebug("[Singularity UI] onNodeCreated called for node", this);
            applySingularityVisibleTitle(this, nodeData);
            
            // Stable min size for multiline prompts and native preview. The pause controls live in a DOM overlay below the node.
            if (!this.size) this.size = [720, UI.minHeight];
            if (this.size[0] < UI.minWidth) this.size[0] = UI.minWidth;
            if (this.size[1] < UI.minHeight) this.size[1] = UI.minHeight;
            installPrivatePreviewStore(this);
            installMediaWidgetGuard(this);
            installSourceWidgetWatcher(this);
            suppressNativeMediaWidgets(this);
            stabilizeNodeLayout(this);

            // Keep the native ComfyUI image upload button from source_image_file.
            // Only remove the old custom duplicate from earlier JS builds.
            // Cleanup for old node instances in the workflow that may still have the widget from previous JS.
            const oldUpload = this.widgets.findIndex(w => w.name === "event_horizon_upload_btn");
            if (oldUpload !== -1) {
                this.widgets.splice(oldUpload, 1);
            }

            // Cleanup for continue button widgets (we now use a DOM overlay when paused, no widget).
            // Old persisted "continue_cascade_btn" from previous versions duplicate and multiply on each run/restore.
            // Remove all of them aggressively.
            for (let i = this.widgets.length - 1; i >= 0; i--) {
                if (this.widgets[i] && this.widgets[i].name === "continue_cascade_btn") {
                    this.widgets.splice(i, 1);
                }
            }

            // Selected tail frame index (synced with green outline clicks)
            // The widget may be auto-created by Comfy (because declared in optional) or added here.
            // We collapse its size so it takes no visual space (bar is the real Tail 5 UI), but .value is still saved/serialized for workflow persistence.
            let tailIdxWidget = this.widgets.find(w => w.name === "selected_tail_index");
            if (!tailIdxWidget) {
                tailIdxWidget = this.addWidget("number", "tail#", "selected_tail_index", -1, {
                    min: -1, max: 4, step: 1
                });
            }
            if (tailIdxWidget) {
                tailIdxWidget.options = tailIdxWidget.options || {};
                tailIdxWidget.options.min = -1;
                tailIdxWidget.options.max = 4;
                tailIdxWidget.options.step = 1;
                tailIdxWidget.min = -1;
                tailIdxWidget.max = 4;
                // Old workflows may persist 0 from earlier builds. Before a live pause/click,
                // selection must be neutral so the run button starts a new route, not a stale continue.
                if (!this._singularityPaused) {
                    tailIdxWidget.value = -1;
                }
                // Hide visually (prevents extra spinner/widget row under TAIL 5 group), keep for data + callback sync from bar clicks.
                tailIdxWidget.label = "";
                tailIdxWidget.hidden = true;
                tailIdxWidget.computeSize = function() { return [0, -6]; };
            }
            stabilizeNodeLayout(this);

            // Pause-aware Tail selection: no pre-chosen green before pause. Only during pause user clicks to choose, green appears then.
            removePauseOverlay(this);
            this._singularityPaused = false;
            this._singularitySelectedTailIndex = -1;  // none selected until explicit click *during* the pause
            this._singularityFormulaBestIndex = 0;
            this._singularityTailScores = [0, 0, 0, 0, 0];

            // Force bounded size and prevent the native preview from stretching the node indefinitely.
            this.setSize([Math.max(this.size ? this.size[0] : 0, UI.minWidth), Math.max(UI.minHeight, Math.min(this.size ? this.size[1] : UI.minHeight, UI.maxHeight))]);
            renderPauseOverlay(this, true);
            if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
            return r;
        };

        // onConfigure to re-enforce min size after loading workflow (prevents UI collapse after node size changes or reloads)
        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = originalOnConfigure ? originalOnConfigure.apply(this, arguments) : undefined;
            applySingularityVisibleTitle(this, nodeData);
            if (!this.size) this.size = [UI.minWidth, UI.minHeight];
            if (this.size[0] < UI.minWidth) this.size[0] = UI.minWidth;
            if (this.size[1] < UI.minHeight) this.size[1] = UI.minHeight;
            installPrivatePreviewStore(this);
            installMediaWidgetGuard(this);
            installSourceWidgetWatcher(this);
            suppressNativeMediaWidgets(this);
            stabilizeNodeLayout(this);

            // Reset interactive state on load; will be set by onExecuted when a real pause happens
            removePauseOverlay(this);
            this._singularityPaused = false;
            this._singularitySelectedTailIndex = -1;

            // Cleanup for continue button widgets (we now use a DOM overlay when paused, no widget).
            // Old persisted "continue_cascade_btn" from previous versions duplicate and multiply on each run/restore.
            for (let i = this.widgets.length - 1; i >= 0; i--) {
                if (this.widgets[i] && this.widgets[i].name === "continue_cascade_btn") {
                    this.widgets.splice(i, 1);
                }
            }

            // Force size on configure too (min safety for bottom bar)
            this.setSize([Math.max(this.size[0], UI.minWidth), Math.max(UI.minHeight, Math.min(this.size[1], UI.maxHeight))]);
            renderPauseOverlay(this, true);
            if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            installMediaWidgetGuard(this);
            suppressNativeMediaWidgets(this);

            if (message && message.gifs && message.gifs.length > 0) {
                const videoData = message.gifs[0];
                const videoUrl = api.apiURL("/view?filename=" + encodeURIComponent(videoData.filename) + "&type=" + videoData.type + "&subfolder=" + videoData.subfolder + "&format=" + videoData.format);
                
                if (!this._SingularityResultVideoCache || this._SingularityResultVideoCache.filename !== videoData.filename) {
                    const vid = document.createElement("video");
                    vid.src = videoUrl;
                    vid.autoplay = true;
                    vid.loop = true;
                    vid.muted = true;
                    vid.play();
                    this._SingularityResultVideoCache = { filename: videoData.filename, video: vid };
                }
                clearPauseState(this);
            }

            // Detect PAUSED from the status output (python sets result_status="PAUSED" after first body + pause_after_cascade_1).
            // This makes the resume button and interactive Tail selection appear only after pause.
            // Remove the widget-based continue button; it was hidden by oversized group backgrounds and polluted the widget list.
            // We draw a custom bottom resume strip instead (see onDraw + onMouse).
            let isPaused = false;
            try {
                if (message) {
                    const s = message.status || (Array.isArray(message.outputs) && message.outputs[0]) || (message.ui && message.ui[0]) || '';
                    if (String(s).indexOf('PAUSED') !== -1) isPaused = true;
                }
            } catch (e) {}
            const wasPaused = Boolean(this._singularityPaused);
            this._singularityPaused = isPaused || wasPaused;
            if (isPaused) {
                this._singularitySelectedTailIndex = -1;  // clear any pre-choice; user must click during this pause to pick green frame
                renderPauseOverlay(this, true);
                if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
            } else if (wasPaused && message && !String(message.status || "").includes("PAUSED")) {
                clearPauseState(this);
            }

            // Always clean any continue widgets on execution (in case old persisted ones or any stray add).
            // We use a custom drawn button now (see onDrawForeground and onMouseDown) to avoid duplication and preview interference.
            for (let i = this.widgets.length - 1; i >= 0; i--) {
                if (this.widgets[i] && this.widgets[i].name === "continue_cascade_btn") {
                    this.widgets.splice(i, 1);
                }
            }
            return;
        };

        // Green outline tail frame selection (as per Gemini version)
        // User can click the tail slots for visual green border selection.
        // Default -1 (no frame chosen) until a pause execution + explicit user click during pause.
        nodeType.prototype._singularitySelectedTailIndex = -1;
        nodeType.prototype._singularityPaused = false;

        const originalOnMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function(e, local_pos, canvas) {
            if (originalOnMouseDown) {
                const res = originalOnMouseDown.apply(this, arguments);
                if (res) return res;
            }

            // Pause controls are now DOM overlay controls outside the node frame.
            // Keep the overlay positioned if a mouse interaction wakes this node.
            if (this._singularityPaused || this._singularityPauseImgs?.length) {
                renderPauseOverlay(this);
            }
            return false;
        };

        // === Stable background groups + lightweight Tail 5 panel ===
        const originalOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            installPrivatePreviewStore(this);
            installMediaWidgetGuard(this);
            installSourceWidgetWatcher(this);
            suppressNativeMediaWidgets(this);
            const r = originalOnDrawForeground ? originalOnDrawForeground.apply(this, arguments) : undefined;

            try {
                suppressNativeMediaWidgets(this);
                stabilizeNodeLayout(this);
                // 1. Colored background bands behind widgets (Gemini's stable r29+ method)
                // Groups stay within node bounds. Using destination-over to place behind widgets.
                drawGroupBackgrounds(ctx, this);

                // 2. Pause Tail/MirrorCut panel + resume strip.
                // Draw only while a pause exists. The actual controls live in a DOM overlay
                // below the node frame so the native preview/noise cannot hide them.
                renderPauseOverlay(this);
            } catch (e) {}

            return r;
        };

    },
});


