console.log("EVENT HORIZON UI CLEAN (V3) JS FILE LOADED!");

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const CLEAN_NODE_NAMES = new Set([
    "EventHorizon",
    "EventHorizonCascadeSimple",
]);
const LEGACY_NODE_NAMES = new Set([]);

app.registerExtension({
    name: "event_horizon.ui.clean",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const isClean = CLEAN_NODE_NAMES.has(nodeData?.name);
        const isLegacy = LEGACY_NODE_NAMES.has(nodeData?.name);
        if (!isClean && !isLegacy) return;

        // Custom File Upload Button ONLY
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = originalOnNodeCreated?.apply(this, arguments);
            
            if (!this.widgets.find(w => w.name === "event_horizon_upload_btn")) {
                this.addWidget("button", "📤 Загрузить фото", "event_horizon_upload_btn", () => {
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
                                const sourceWidget = this.widgets.find(w => w.name === "source_image_file");
                                if (sourceWidget) {
                                    if (!sourceWidget.options.values.includes(filename)) {
                                        sourceWidget.options.values.push(filename);
                                    }
                                    sourceWidget.value = filename;
                                    if (sourceWidget.callback) sourceWidget.callback(filename);
                                    if (this.setDirtyCanvas) this.setDirtyCanvas(true, true);
                                }
                            }
                        } catch(err) {
                            console.error("[EventHorizon] File upload failed", err);
                        }
                    };
                    input.click();
                });
            }
            return r;
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
                }
            }

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

        const originalOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            if (originalOnDrawForeground) originalOnDrawForeground.apply(this, arguments);

            // Draw completely BELOW the node. We do not fight ComfyUI sizing.
            ctx.save();
            const nodeWidth = this.size[0];
            const startY = this.size[1] + 10; // Draw under the node boundary

            // --- 3-PANEL MEDIA DASHBOARD ---
            const panelWidth = (nodeWidth - 20 - (10 * 4)) / 3;
            let mediaAreaHeightDraw = panelWidth + 50; 
            
            ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
            ctx.fillRect(10, startY, nodeWidth - 20, mediaAreaHeightDraw);

            const labels = ["Фото (Source)", "Шум (Latent)", "Видео (Result)"];

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
                ctx.fillText(labels[i], x + panelWidth / 2, startY + 20);

                ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
                ctx.lineWidth = 1;
                ctx.strokeRect(x, startY + 30, panelWidth, panelWidth);

                let imgToDraw = null;
                const privImgs = this.eventHorizonPauseImgs;
                
                if (i === 0) {
                    imgToDraw = sourceImage;
                } else if (this._eventHorizonResultVideoCache && i === 2) {
                    imgToDraw = this._eventHorizonResultVideoCache.video;
                } else if (privImgs && privImgs.length > 0) {
                    if (i === 1) {
                        imgToDraw = privImgs[0];
                    } else if (i === 2) {
                        imgToDraw = privImgs[privImgs.length - 1];
                    }
                }

                if (imgToDraw && (imgToDraw.complete || imgToDraw.readyState >= 2)) {
                    let naturalWidth = imgToDraw.naturalWidth || imgToDraw.videoWidth;
                    let naturalHeight = imgToDraw.naturalHeight || imgToDraw.videoHeight;
                    if (naturalWidth && naturalHeight) {
                        const aspect = naturalWidth / naturalHeight;
                        const maxH = panelWidth - 4;
                        const maxW = panelWidth - 4;
                        let drawW = maxW;
                        let drawH = drawW / aspect;
                        if (drawH > maxH) { drawH = maxH; drawW = drawH * aspect; }
                        let offsetX = (panelWidth - drawW) / 2;
                        let offsetY = 32 + (maxH - drawH) / 2;
                        ctx.drawImage(imgToDraw, x + offsetX, startY + offsetY, drawW, drawH);
                    }
                }
            }

            // --- PAUSE FILMSTRIP ---
            let filmstripHeight = 160;
            if (this.eventHorizonPauseImgs && this.eventHorizonPauseImgs.length > 0) {
                const filmstripStartY = startY + mediaAreaHeightDraw + 10;
                
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
