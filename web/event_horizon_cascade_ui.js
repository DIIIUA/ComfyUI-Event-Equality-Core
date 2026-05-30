// Event Horizon UI helper r59 public clean
// WIDGET STABILITY HOTFIX.
// Important: this file must NEVER sort or reorder node.widgets.
// ComfyUI widget_values are positional, so sorting widgets can visually/semantically mix settings.

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

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
        names: ["source_image_file", "event_horizon_upload_btn"],
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
        names: ["cascade_count", "frames_per_cascade", "width", "height", "fps", "seed", "pause_after_cascade_1", "pause_after_cascade_2", "pause_after_cascade_3", "pause_after_cascade_4"],
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

    let fallbackY = 50;

    // We no longer rely on ComfyUI's native image rendering to push widgets down,
    // because we have a dedicated bottom panel for images. So we ignore this.imgs
    // when calculating fallbackY.

    let maxBottom = fallbackY;

    for (const w of node.widgets) {
        if (w.type === "hidden" || w.name === "resume_frame_index") {
            continue; // Skip layout for hidden widgets
        }

        const isPrompt = PROMPT_WIDGET_NAMES.has(w?.name || "");
        if (isPrompt) {
            expandPromptWidget(w);
        }

        let h = isPrompt ? UI.promptHeight : widgetHeight(w, 22);
        if (w.computeSize) {
            let ch = w.computeSize(node.size[0])[1];
            if (ch && ch > h) h = ch;
        }

        // Force strict sequential layout to prevent overlap
        w.y = fallbackY;
        w.last_y = fallbackY;

        fallbackY += h + 8;
        maxBottom = Math.max(maxBottom, fallbackY);
    }

    // Dynamic height based on node width (3 square panels)
    const panelWidth = (node.size[0] - 20 - 40) / 3;
    let requiredHeight = Math.max(UI.minHeight, Math.ceil(maxBottom + 48));
    
    // Reserve space for the 3-Panel Media UI
    requiredHeight += panelWidth + 50;

    // Reserve additional space for the Filmstrip if we are paused
    if (node.eventHorizonPauseImgs && node.eventHorizonPauseImgs.length > 0) {
        requiredHeight += 180;
    }

    if (!node.size) node.size = [UI.minWidth, requiredHeight];
    if (node.size[0] < UI.minWidth) node.size[0] = UI.minWidth;
    
    // Always force the height to prevent ComfyUI from randomly collapsing or expanding it
    if (node.size[1] !== requiredHeight) {
        node.size[1] = requiredHeight;
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    }
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
        if (w.name === "source_image_file") {
            w.label = "source image";
            // No need to patch w.draw since it's a standard COMBO now, not an imageUpload widget!
        }
        if (w.name === "resume_frame_index") {
            w.type = "hidden";
            w.hidden = true;
            w.disabled = true;
            w.computeSize = () => [0, 0];
            w.draw = () => {};
        }
        expandPromptWidget(w);
    }
    
    // Hook instance computeSize to override any ComfyUI instance-level patching
    if (!node.__eventHorizonInstanceComputePatched) {
        const originalInstanceComputeSize = node.computeSize;
        node.computeSize = function(out) {
            let size = originalInstanceComputeSize ? originalInstanceComputeSize.apply(this, arguments) : 
                       (Object.getPrototypeOf(this).computeSize ? Object.getPrototypeOf(this).computeSize.apply(this, arguments) : [UI.minWidth, UI.minHeight]);
            
            // Always reserve +180 for the filmstrip at the bottom
            if (size && size[1] !== undefined && !this.__eventHorizonComputeSizeReentered) {
                this.__eventHorizonComputeSizeReentered = true;
                size[1] += 180;
                this.__eventHorizonComputeSizeReentered = false;
            }
            return size;
        };
        node.__eventHorizonInstanceComputePatched = true;
    }

    // --- Custom File Upload Button ---
    if (!node.widgets.find(w => w.name === "event_horizon_upload_btn")) {
        node.addWidget("button", "📤 Загрузить фото", "event_horizon_upload_btn", () => {
            const input = document.createElement("input");
            input.type = "file";
            input.accept = "image/*";
            input.onchange = async (e) => {
                const file = e.target.files[0];
                if (!file) return;
                const body = new FormData();
                body.append("image", file);
                body.append("subfolder", "");
                body.append("type", "input");
                try {
                    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                    if (resp.status === 200) {
                        const data = await resp.json();
                        const filename = data.name;
                        const sourceWidget = node.widgets.find(w => w.name === "source_image_file");
                        if (sourceWidget) {
                            if (!sourceWidget.options.values.includes(filename)) {
                                sourceWidget.options.values.push(filename);
                            }
                            sourceWidget.value = filename;
                            if (sourceWidget.callback) sourceWidget.callback(filename);
                            if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
                        }
                    }
                } catch(err) {
                    console.error("[EventHorizon] File upload failed", err);
                }
            };
            input.click();
        });
    }

    // Force upload button to be right after source_image_file visually (without breaking python positional args)
    let uploadBtnIndex = node.widgets.findIndex(w => w.name === "event_horizon_upload_btn");
    let sourceIndex = node.widgets.findIndex(w => w.name === "source_image_file");
    if (uploadBtnIndex !== -1 && sourceIndex !== -1 && uploadBtnIndex !== sourceIndex + 1) {
        const btn = node.widgets.splice(uploadBtnIndex, 1)[0];
        node.widgets.splice(sourceIndex + 1, 0, btn);
    }

    // --- GROK FIX: Completely suppress ComfyUI's core image preview rendering ---
    if (!node._imgsSuppressed) {
        Object.defineProperty(node, 'imgs', {
            get() {
                return undefined; // Hide from core renderer
            },
            set(v) {
                node._eventHorizonPrivateImgs = v; // Keep for our dashboard
            },
            configurable: true,
            enumerable: true
        });

        node.imgs = undefined;
        node.imageIndex = null;
        node.videoContainer = undefined; // Suppress video preview just in case
        node._imgsSuppressed = true;
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

    // Use actual widget y positions computed in enforceLayoutStability
    for (let i = 0; i < widgets.length; i++) {
        const w = widgets[i];
        if (w.type === "hidden" || w.name === "resume_frame_index") continue;

        const g = groupForWidget(w);
        const id = g?.id || "OTHER";

        const y = w.y || 86;
        let h = widgetHeight(w, 22);
        if (w.computeSize) {
            let ch = w.computeSize(node.size[0])[1];
            if (ch && ch > h) h = ch;
        }

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
    name: "event_horizon.ui.r62",

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
            let isPauseFilmstrip = false;
            if (this._eventHorizonPrivateImgs && this._eventHorizonPrivateImgs.length > 0 && this._eventHorizonPrivateImgs[0].src && this._eventHorizonPrivateImgs[0].src.includes("cascade_preview")) {
                isPauseFilmstrip = true;
            }

            // We no longer need to backup this.imgs because it is permanently suppressed
            const r = originalOnDrawForeground?.apply(this, arguments);

            try {
                enforceLayoutStability(this);
                drawDynamicBackground(ctx, this);
            } catch (e) {}
            return r;
        };

        const originalComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function(out) {
            let size = originalComputeSize ? originalComputeSize.apply(this, arguments) : [UI.minWidth, UI.minHeight];
            // Always reserve +180 for the filmstrip at the bottom
            if (size[1]) {
                size[1] += 180;
            }
            return size;
        };

        const originalSetSize = nodeType.prototype.setSize;
        nodeType.prototype.setSize = function(size) {
            if (size[0] < UI.minWidth) size[0] = UI.minWidth;
            // Let the computed size (which now includes +180) take effect.
            if (this.__eventHorizonRequiredHeight && size[1] < this.__eventHorizonRequiredHeight) {
                size[1] = this.__eventHorizonRequiredHeight;
            }
            if (originalSetSize) {
                originalSetSize.call(this, size);
            } else {
                this.size = size;
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) onExecuted.apply(this, arguments);

            if (message && message.pause_frames) {
                this.eventHorizonPauseImgs = [];
                for (let i = 0; i < message.pause_frames.length; i++) {
                    const img = new Image();
                    img.src = api.apiURL("/view?filename=" + encodeURIComponent(message.pause_frames[i].filename) + "&type=temp&subfolder=cascade_preview");
                    img.__eventHorizonResumeIndex = message.pause_frames[i].resume_index;
                    this.eventHorizonPauseImgs.push(img);
                }
            }
            if (message && message.gifs && message.gifs.length > 0) {
                const videoData = message.gifs[0];
                const videoUrl = api.apiURL("/view?filename=" + encodeURIComponent(videoData.filename) + "&type=" + videoData.type + "&subfolder=" + videoData.subfolder + "&format=" + videoData.format);
                
                if (!this._eventHorizonResultVideoCache || this._eventHorizonResultVideoCache.filename !== videoData.filename) {
                    const vid = document.createElement("video");
                    vid.src = videoUrl;
                    vid.autoplay = true;
                    vid.loop = true;
                    vid.muted = true;
                    vid.play();
                    this._eventHorizonResultVideoCache = { filename: videoData.filename, video: vid };
                    
                    if (!this._eventHorizonVideoInterval) {
                        this._eventHorizonVideoInterval = setInterval(() => { 
                            if (this._eventHorizonResultVideoCache && this.setDirtyCanvas) {
                                this.setDirtyCanvas(true, false);
                            }
                        }, 33);
                    }
                }
            }
                // PERMANENTLY hide the resume_frame_index widget so user doesn't see or type into it
                const resumeWidget = this.widgets?.find(w => w.name === "resume_frame_index");
                if (resumeWidget) {
                    resumeWidget.type = "hidden";
                    resumeWidget.computeSize = () => [0, 0];
                }

                if (this.widgets) {
                    for (let w of this.widgets) {
                        if (w.name === "videopreview" || w.name === "imagepreview" || w.type === "video" || w.type === "IMAGE") {
                            if (w.element) {
                                w.element.style.display = "none";
                                w.element.hidden = true;
                            }
                            w.computeSize = () => [0, 0];
                        }
                    }
                }
                
                this.setSize(this.computeSize());

                let continueBtn = this.widgets?.find(w => w.name === "continue_cascade_btn");
                if (!continueBtn) {
                    this.addWidget("button", "▶ Resume Cascade / Continue", "continue_cascade_btn", () => {
                        const resumeWidget = this.widgets?.find(w => w.name === "resume_frame_index");
                        if (resumeWidget && resumeWidget.value === -1) {
                            alert("Please click one of the frames to select the resume point before continuing!");
                            return;
                        }
                        app.queuePrompt(0);
                    });
                }
            }
            return;
        };

        const originalOnMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function(e, local_pos, canvas) {
            if (this.filmstripGeometry) {
                const {x, y, width, height} = this.filmstripGeometry;
                if (local_pos[0] >= x && local_pos[0] <= x + width && local_pos[1] >= y && local_pos[1] <= y + height) {
                    const imgCount = this.eventHorizonPauseImgs.length;
                    const thumbW = width / imgCount;
                    const clickedIndex = Math.floor((local_pos[0] - x) / thumbW);
                    
                    if (clickedIndex >= 0 && clickedIndex < imgCount) {
                        this.selectedPauseFrame = clickedIndex;
                        const resumeWidget = this.widgets?.find(w => w.name === "resume_frame_index");
                        if (resumeWidget) {
                            resumeWidget.value = this.eventHorizonPauseImgs[clickedIndex].__eventHorizonResumeIndex;
                        }
                        if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
                        return true;
                    }
                }
            }
            if (originalOnMouseDown) {
                return originalOnMouseDown.apply(this, arguments);
            }
            return false;
        };

        const originalOnDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function(ctx) {
            // Clear pause frames if we are actively generating
            if (app.runningNodeId === this.id) {
                this.eventHorizonPauseImgs = null;
                this.selectedPauseFrame = -1;
            }

            // We no longer need to backup this.imgs because it is permanently suppressed
            if (originalOnDrawBackground) {
                originalOnDrawBackground.apply(this, arguments);
            }

            ctx.save();
            const nodeWidth = this.size[0];
            
            // BULLETPROOF: Dynamically compute exactly where the widgets end!
            let widgetEnd = 30; // LiteGraph default start
            if (this.widgets) {
                for (const w of this.widgets) {
                    if (w.type === "hidden" || w.name === "resume_frame_index") continue;
                    let h = 22;
                    if (w.computeSize) {
                        let ch = w.computeSize(nodeWidth)[1];
                        if (ch && ch > h) h = ch;
                    } else if (w.name === "positive_prompt" || w.name === "negative_prompt") {
                        h = 180;
                    }
                    widgetEnd += h + 4; // LiteGraph widget spacing
                }
            }
            
            let filmstripHeight = 160;
            
            // Recompute mediaAreaHeight correctly since it depends on nodeWidth
            const panelWidth = (nodeWidth - 20 - (10 * 4)) / 3;
            let mediaAreaHeight = panelWidth + 50;

            let totalBottomSpace = mediaAreaHeight;
            if (this.eventHorizonPauseImgs && this.eventHorizonPauseImgs.length > 0) {
                totalBottomSpace += filmstripHeight + 10;
            }

            // Ensure the node is tall enough to fit everything!
            let absoluteRequiredHeight = widgetEnd + totalBottomSpace + 40;
            if (this.size[1] < absoluteRequiredHeight) {
                this.size[1] = absoluteRequiredHeight;
            }
            
            // Start the media dashboard exactly after the widgets (plus some padding)
            // But also respect the bottom of the node if it was dragged to be huge.
            const startY = Math.max(this.size[1] - totalBottomSpace, widgetEnd + 20);
            const mediaStartY = startY;

            // --- 3-PANEL MEDIA DASHBOARD ---
            let mediaAreaHeightDraw = panelWidth + 50; 
            
            ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
            ctx.fillRect(10, mediaStartY, nodeWidth - 20, mediaAreaHeightDraw);

            const labels = ["Фото (Source)", "Шум (Latent)", "Видео (Result)"];

            // Load source image manually to bypass ComfyUI image renderer completely
            let sourceImage = null;
            const sourceWidget = this.widgets?.find(w => w.name === "source_image_file");
            if (sourceWidget && sourceWidget.value) {
                if (!this._eventHorizonSourceImgCache || this._eventHorizonSourceImgCache.filename !== sourceWidget.value) {
                    const img = new Image();
                    img.src = api.apiURL("/view?filename=" + encodeURIComponent(sourceWidget.value) + "&type=input&subfolder=");
                    this._eventHorizonSourceImgCache = { filename: sourceWidget.value, img: img };
                }
                sourceImage = this._eventHorizonSourceImgCache.img;
            }

            for (let i = 0; i < 3; i++) {
                let x = 10 + 10 + (i * (panelWidth + 10));
                
                ctx.fillStyle = "#888";
                ctx.font = "bold 12px Arial";
                ctx.textAlign = "center";
                ctx.fillText(labels[i], x + panelWidth / 2, mediaStartY + 20);

                ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
                ctx.lineWidth = 1;
                // Inner box is exactly panelWidth x panelWidth (SQUARE)
                ctx.strokeRect(x, mediaStartY + 30, panelWidth, panelWidth);

                let imgToDraw = null;
                const privImgs = this.eventHorizonPauseImgs; // <--- Changed to eventHorizonPauseImgs to show something!
                
                if (i === 0) {
                    imgToDraw = sourceImage; // Left panel: Always Source Photo
                } else if (this._eventHorizonResultVideoCache && i === 2) {
                    imgToDraw = this._eventHorizonResultVideoCache.video; // Right panel: Final Video
                } else if (privImgs && privImgs.length > 0) {
                    if (i === 1) {
                        imgToDraw = privImgs[0]; // Middle panel: First frame of noise/pause
                    } else if (i === 2) {
                        imgToDraw = privImgs[privImgs.length - 1]; // Right panel: Last frame of pause
                    }
                }

                if (imgToDraw && (imgToDraw.complete || imgToDraw.readyState >= 2)) {
                    let naturalWidth = imgToDraw.naturalWidth || imgToDraw.videoWidth;
                    let naturalHeight = imgToDraw.naturalHeight || imgToDraw.videoHeight;
                    if (naturalWidth && naturalHeight) {
                        const aspect = naturalWidth / naturalHeight;
                    const maxH = panelWidth - 4; // slight padding
                    const maxW = panelWidth - 4;
                    
                    let drawW = maxW;
                    let drawH = drawW / aspect;
                    
                    if (drawH > maxH) {
                        drawH = maxH;
                        drawW = drawH * aspect;
                    }
                    
                    let offsetX = (panelWidth - drawW) / 2;
                    let offsetY = 32 + (maxH - drawH) / 2;
                    
                    ctx.drawImage(imgToDraw, x + offsetX, mediaStartY + offsetY, drawW, drawH);
                    }
                } else if (i === 1 || i === 2) {
                    ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
                    ctx.fillText("Ожидание...", x + panelWidth / 2, mediaStartY + 30 + panelWidth / 2 + 5);
                }
            }

            // --- PAUSE FILMSTRIP ---
            if (this.eventHorizonPauseImgs && this.eventHorizonPauseImgs.length > 0) {
                const filmstripStartY = mediaStartY + mediaAreaHeightDraw + 10;
                
                ctx.fillStyle = "rgba(10, 20, 50, 0.6)";
                ctx.fillRect(10, filmstripStartY, nodeWidth - 20, filmstripHeight);
                ctx.strokeStyle = "#4466ff";
                ctx.strokeRect(10, filmstripStartY, nodeWidth - 20, filmstripHeight);
                
                this.filmstripGeometry = {
                    x: 10,
                    y: filmstripStartY,
                    width: nodeWidth - 20,
                    height: filmstripHeight
                };

                const imgCount = this.eventHorizonPauseImgs.length;
                const margin = 10;
                const availableWidth = nodeWidth - 20 - (margin * 2);
                let thumbW = availableWidth / imgCount;
                let thumbH = filmstripHeight - (margin * 2);

                for (let i = 0; i < imgCount; i++) {
                    const img = this.eventHorizonPauseImgs[i];
                    if (!img.complete) continue;

                    const aspect = img.naturalWidth / img.naturalHeight;
                    let drawW = thumbW - 10;
                    let drawH = drawW / aspect;

                    if (drawH > thumbH) {
                        drawH = thumbH;
                        drawW = drawH * aspect;
                    }

                    const x = 10 + margin + (i * thumbW) + (thumbW - drawW) / 2;
                    const y = filmstripStartY + margin + (thumbH - drawH) / 2;

                    ctx.drawImage(img, x, y, drawW, drawH);

                    ctx.fillStyle = "rgba(0,0,0,0.7)";
                    ctx.fillRect(x, y, 30, 24);
                    ctx.fillStyle = "#FFF";
                    ctx.font = "bold 16px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText(String(i + 1), x + 15, y + 18);

                    if (this.selectedPauseFrame === i) {
                        ctx.strokeStyle = "#00FF00";
                        ctx.lineWidth = 4;
                        ctx.strokeRect(x, y, drawW, drawH);
                    }
                }
            } else {
                this.filmstripGeometry = null;
            }

            ctx.restore();
        };

    },
});
