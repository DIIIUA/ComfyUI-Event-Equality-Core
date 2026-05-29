// Event Horizon UI helper r59 public clean
// WIDGET STABILITY HOTFIX.
// Important: this file must NEVER sort or reorder node.widgets.
// ComfyUI widget_values are positional, so sorting widgets can visually/semantically mix settings.

import { app } from "/scripts/app.js";

const CLEAN_NODE_NAMES = new Set([
    "EventHorizon",
]);
const LEGACY_NODE_NAMES = new Set([]);

const UI = {
    // Make prompt lanes comfortably readable by default.
    minWidth: 1024,
    minHeight: 860,
    marginX: 7,
    bandPadY: 5,
    labelX: 18,
    promptHeight: 180,
};

const PROMPT_WIDGET_NAMES = new Set(["positive_prompt", "negative_prompt"]);

const GROUPS = [
    {
        id: "SOURCE",
        color: "#1d4ed8",
        bg: "rgba(30, 64, 175, 0.14)",
        names: ["source_image_file"],
    },
    {
        id: "PROMPT",
        color: "#7c3aed",
        bg: "rgba(91, 33, 182, 0.13)",
        names: ["positive_prompt", "negative_prompt", "temporal_texture_lock"],
    },
    {
        id: "CASCADE",
        color: "#0891b2",
        bg: "rgba(8, 145, 178, 0.12)",
        names: ["cascade_count", "frames_per_cascade", "width", "height", "fps", "seed", "pause_after_cascade_1", "pause_after_cascade_2", "pause_after_cascade_3", "pause_after_cascade_4", "resume_frame_index"],
    },
    {
        id: "SAMPLING",
        color: "#ca8a04",
        bg: "rgba(202, 138, 4, 0.11)",
        names: ["sampler_name", "scheduler", "global_steps", "primary_cfg", "secondary_cfg", "primary_start_step", "primary_end_step", "secondary_start_step", "secondary_end_step", "math_control_mode", "high_delta_strength", "low_delta_strength"],
    },
    {
        id: "DECODE",
        color: "#059669",
        bg: "rgba(5, 150, 105, 0.11)",
        names: ["decode_tile_size", "decode_overlap", "decode_temporal_size", "decode_temporal_overlap", "image_upscale_method", "image_crop", "cleanup_timing"],
    },
    {
        id: "POST GENERATION",
        color: "#dc2626",
        bg: "rgba(220, 38, 38, 0.11)",
        names: ["save_video", "video_format", "save_report", "save_prefix"],
    },
];

function nodeTypeName(node) {
    return node?.comfyClass || node?.type || node?.constructor?.type || node?.title || "";
}

function isCascadeNode(node) {
    const type = nodeTypeName(node);
    return CLEAN_NODE_NAMES.has(type) || LEGACY_NODE_NAMES.has(type) || String(node?.title || "").includes("Event Horizon");
}

function isLegacyNode(node) {
    const type = nodeTypeName(node);
    return LEGACY_NODE_NAMES.has(type) || String(node?.title || "").includes("Legacy") || String(node?.title || "").includes("Advanced");
}

function groupForWidget(widget) {
    const name = widget?.name || "";
    for (const g of GROUPS) {
        if (g.names.includes(name)) return g;
    }
    return null;
}

function expandPromptWidget(widget) {
    if (!widget || !PROMPT_WIDGET_NAMES.has(widget.name)) return;

    // R55 prompt lock:
    // Fixed height, no manual textarea resize, no growing/shrinking.
    // This avoids positive/negative prompt overlap caused by user-resized multiline widgets.
    widget.options = widget.options || {};
    widget.options.height = UI.promptHeight;
    widget.options.resizable = false;
    widget.options.no_resize = true;
    widget.options.disable_resize = true;
    widget.minHeight = UI.promptHeight;
    widget.maxHeight = UI.promptHeight;
    widget.computedHeight = UI.promptHeight;
    widget.h = UI.promptHeight;
    widget.height = UI.promptHeight;

    if (widget.inputEl?.style) {
        widget.inputEl.style.minHeight = `${UI.promptHeight}px`;
        widget.inputEl.style.height = `${UI.promptHeight}px`;
        widget.inputEl.style.maxHeight = `${UI.promptHeight}px`;
        widget.inputEl.style.boxSizing = "border-box";
        widget.inputEl.style.width = "100%";
        widget.inputEl.style.resize = "none";
        widget.inputEl.style.overflowY = "auto";
        widget.inputEl.style.overflowX = "hidden";
    }

    if (typeof widget.computeSize === "function" && !widget.__eventHorizonPromptComputePatched) {
        const originalComputeSize = widget.computeSize.bind(widget);
        widget.computeSize = function(width) {
            const size = originalComputeSize(width) || [width, UI.promptHeight];
            size[1] = UI.promptHeight;
            return size;
        };
        widget.__eventHorizonPromptComputePatched = true;
    }
}


function enforceLayoutStability(node) {
    if (!node || !Array.isArray(node.widgets) || !node.widgets.length) return;

    // Preserve native widget order. Do not sort.
    // Only lock prompt widget geometry and ensure the node is tall enough for Comfy's own layout.
    let fallbackY = 86;
    let maxBottom = 0;
    let previousPromptBottom = -1;

    for (const w of node.widgets) {
        const isPrompt = PROMPT_WIDGET_NAMES.has(w?.name || "");
        if (isPrompt) {
            expandPromptWidget(w);
        }

        let y = widgetY(w, fallbackY);
        const h = isPrompt ? UI.promptHeight : widgetHeight(w, 22);

        if (isPrompt) {
            const minPromptY = Math.max(fallbackY, previousPromptBottom > -1 ? previousPromptBottom + 10 : fallbackY);
            if (!Number.isFinite(y) || y < minPromptY) {
                y = minPromptY;
                w.last_y = y;
                w.y = y;
            }
            previousPromptBottom = y + h;
        }

        fallbackY = Math.max(fallbackY, y + h + 8);
        maxBottom = Math.max(maxBottom, y + h);
    }

    const requiredHeight = Math.max(UI.minHeight, Math.ceil(maxBottom + 48));
    if (!node.size) node.size = [UI.minWidth, requiredHeight];
    if (node.size[0] < UI.minWidth) node.size[0] = UI.minWidth;
    if (node.size[1] < requiredHeight) node.size[1] = requiredHeight;
    node.__eventHorizonRequiredHeight = requiredHeight;
}


function applyBaseShape(node) {
    const legacy = isLegacyNode(node);
    node.properties = node.properties || {};
    node.properties.event_horizon_ui = legacy ? "advanced_legacy" : "r58_input_integrity_prompt_nonoverlap";
    node.properties.event_horizon_widget_order_policy = "native_order_preserved";
    node.properties.event_horizon_background_tracks_widgets = true;

    // Only set initial minimum. Do not fight user resizing.
    if (!node.size) node.size = [UI.minWidth, UI.minHeight];
    if (node.size[0] < UI.minWidth) node.size[0] = UI.minWidth;
    if (node.size[1] < UI.minHeight) node.size[1] = UI.minHeight;

    node.color = legacy ? "#3f2f1f" : "#1e3a5f";
    node.bgcolor = legacy ? "#17120c" : "#07111f";

    // Safe label-only changes. Do not reorder widgets.
    for (const w of node.widgets || []) {
        if (w.name === "source_image_file") w.label = "source image";
        expandPromptWidget(w);
    }
    enforceLayoutStability(node);
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

function widgetY(widget, fallback) {
    const y = Number(widget?.last_y ?? widget?.y ?? NaN);
    if (Number.isFinite(y)) return y;
    return fallback;
}

function widgetHeight(widget, fallback = 22) {
    const h = Number(widget?.last_h ?? widget?.h ?? widget?.height ?? NaN);
    if (Number.isFinite(h) && h > 4) return h;
    const ch = Number(widget?.computedHeight ?? widget?.options?.height ?? NaN);
    if (Number.isFinite(ch) && ch > 4) return ch;
    return fallback;
}

function computeDynamicRuns(node) {
    const widgets = node.widgets || [];
    const runs = [];
    let current = null;
    let fallbackY = 86;

    // Preserve native ComfyUI widget order exactly.
    for (let i = 0; i < widgets.length; i++) {
        const w = widgets[i];
        const g = groupForWidget(w);
        const id = g?.id || "OTHER";

        const y = widgetY(w, fallbackY);
        const h = widgetHeight(w, 22);
        fallbackY = y + h + 2;

        if (!g) {
            if (current) {
                runs.push(current);
                current = null;
            }
            continue;
        }

        if (!current || current.id !== id) {
            if (current) runs.push(current);
            current = {
                id,
                group: g,
                y1: y,
                y2: y + h,
            };
        } else {
            current.y1 = Math.min(current.y1, y);
            current.y2 = Math.max(current.y2, y + h);
        }
    }
    if (current) runs.push(current);
    return runs.filter((r) => r.group);
}

function drawDynamicBackground(ctx, node) {
    if (!node.widgets || !node.widgets.length) return;

    const runs = computeDynamicRuns(node);
    if (!runs.length) return;

    ctx.save();
    const oldComposite = ctx.globalCompositeOperation;
    ctx.globalCompositeOperation = "destination-over";

    const width = Math.max(100, (node.size?.[0] || UI.minWidth) - UI.marginX * 2);

    for (const run of runs) {
        const g = run.group;
        const y = Math.max(28, run.y1 - UI.bandPadY);
        const h = Math.max(22, (run.y2 - run.y1) + UI.bandPadY * 2);

        ctx.fillStyle = g.bg;
        drawRoundRect(ctx, UI.marginX, y, width, h, 8);
        ctx.fill();

        ctx.strokeStyle = g.color;
        ctx.globalAlpha = 0.9;
        ctx.lineWidth = 1.4;
        drawRoundRect(ctx, UI.marginX, y, width, h, 8);
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        ctx.fillStyle = g.color;
        drawRoundRect(ctx, UI.marginX + 3, y + 5, 5, Math.max(8, h - 10), 3);
        ctx.fill();

        ctx.font = "bold 10px sans-serif";
        ctx.fillStyle = "#ffffff";
        ctx.fillText(g.id, UI.labelX, y + 14);
    }

    ctx.globalCompositeOperation = oldComposite;
    ctx.restore();
}


function injectPromptLockCSS() {
    if (document.getElementById("event-horizon-r58-prompt-lock-css")) return;
    const style = document.createElement("style");
    style.id = "event-horizon-r58-prompt-lock-css";
    style.textContent = `
        textarea {
            scrollbar-gutter: stable;
        }
    `;
    document.head.appendChild(style);
}


app.registerExtension({
    name: "event_horizon.ui.r58",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const isClean = CLEAN_NODE_NAMES.has(nodeData?.name);
        const isLegacy = LEGACY_NODE_NAMES.has(nodeData?.name);
        if (!isClean && !isLegacy) return;
        try { injectPromptLockCSS(); } catch (e) {}

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = originalOnNodeCreated?.apply(this, arguments);
            try { applyBaseShape(this); } catch (e) {
                console.warn("[Event Horizon UI] onNodeCreated failed", e);
            }
            return r;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = originalOnConfigure?.apply(this, arguments);
            try { applyBaseShape(this); } catch (e) {
                console.warn("[Event Horizon UI] onConfigure failed", e);
            }
            return r;
        };

        const originalOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            const r = originalOnDrawForeground?.apply(this, arguments);
            try {
                enforceLayoutStability(this);
                drawDynamicBackground(ctx, this);
            } catch (e) {}
            return r;
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const r = originalOnExecuted?.apply(this, arguments);
            if (message && message.images) {
                let continueBtn = this.widgets?.find(w => w.name === "continue_cascade_btn");
                if (!continueBtn) {
                    this.addWidget("button", "▶ Resume Cascade / Continue", "continue_cascade_btn", () => {
                        app.queuePrompt(0);
                    });
                }
            }
            return r;
        };

        const originalOnDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function(ctx) {
            if (this.imgs && this.imgs.length > 0) {
                // Prevent infinite vertical expansion by drawing our own constrained filmstrip
                ctx.save();
                const nodeWidth = this.size[0];
                const filmstripHeight = 160; 
                const y = this.size[1] - filmstripHeight - 10;
                
                ctx.fillStyle = "#111";
                ctx.fillRect(10, y, nodeWidth - 20, filmstripHeight);
                
                let x = 15;
                for (let img of this.imgs) {
                    if (img.complete && img.naturalWidth) {
                        const aspect = img.naturalWidth / img.naturalHeight;
                        const drawWidth = (filmstripHeight - 10) * aspect;
                        ctx.drawImage(img, x, y + 5, drawWidth, filmstripHeight - 10);
                        x += drawWidth + 10;
                        if (x > nodeWidth) break;
                    }
                }
                ctx.restore();
                
                // Do not call original if we are drawing it ourselves to prevent huge node!
                return;
            }
            if (originalOnDrawBackground) {
                originalOnDrawBackground.apply(this, arguments);
            }
        };

    },
});
