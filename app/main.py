"""
Main FastAPI Application Entrypoint.
Provides Swagger UI at /docs, ReDoc at /redoc, static file mounting,
and an interactive visual Metrology Dashboard at /.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router as api_router
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
## COCO Animal Segmentation & Visual Metrology API (YOLO & ViTPose)

### Features:
1. **Animal Instance Segmentation**: YOLOv8-seg extracts full animal body contour masks and bounding polygons.
2. **Eye Contour Segmentation**: High-precision Left Eye and Right Eye contour segmentation with sub-pixel spatial moments.
3. **Intra-animal Metrology**: Measures inter-pupillary eye distance (每隻動物雙眼距離) in pixels and calibrated millimeters.
4. **Inter-animal Metrology**: Measures right-eye distance between any pair of animals (任意兩隻動物右眼距離).
5. **CSV Reporting & Ground Truth Benchmarking**: Ground Truth evaluation (MAE, RMSE, MAPE) and downloadable CSV results.

### Default Test Accounts:
- **Admin**: `admin` / `AdminPassword123!`
- **Tester**: `metrology_tester` / `TesterPassword123!`
""",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory for viewing annotated images
outputs_path = Path("outputs")
outputs_path.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_path)), name="outputs")

# Mount data/coco_samples directory for sample access
coco_path = Path("data/coco_samples")
coco_path.mkdir(parents=True, exist_ok=True)
app.mount("/samples", StaticFiles(directory=str(coco_path)), name="samples")

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    """Interactive visual dashboard for testing metrology in browser."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-TW">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{settings.APP_NAME}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{
                background-color: #0f172a;
                color: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .card {{
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
            }}
            .badge-metric {{
                font-size: 1.1rem;
                padding: 0.5rem 1rem;
                border-radius: 8px;
            }}
            .upload-box {{
                border: 2px dashed #475569;
                border-radius: 10px;
                padding: 25px;
                text-align: center;
                background-color: #0b1329;
                cursor: pointer;
                transition: border-color 0.2s, background-color 0.2s;
            }}
            .upload-box:hover {{
                border-color: #38bdf8;
                background-color: #132142;
            }}
            .sample-btn {{
                background-color: #334155;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 0.85rem;
                margin-right: 6px;
                margin-bottom: 6px;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .sample-btn:hover {{
                background-color: #0284c7;
                color: #ffffff;
                border-color: #38bdf8;
            }}
            pre {{
                background-color: #020617;
                color: #38bdf8;
                padding: 12px;
                border-radius: 8px;
                font-size: 0.85rem;
            }}
            .spinner-overlay {{
                display: none;
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                background: rgba(15, 23, 42, 0.75);
                border-radius: 12px;
                z-index: 10;
                align-items: center;
                justify-content: center;
            }}
        </style>
    </head>
    <body class="p-4">
        <div class="container-fluid" style="max-width: 1400px;">
            <!-- Header -->
            <header class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary">
                <div>
                    <h2 class="fw-bold text-primary mb-1">COCO 動物分割與視覺幾何計量系統</h2>
                    <p class="text-secondary mb-0">Powered by YOLOv8-seg (實例切割) + ViTPose AP-10K (跨物種眼睛姿態)</p>
                </div>
                <div>
                    <a href="/docs" target="_blank" class="btn btn-outline-info me-2">Swagger API 文件</a>
                    <a href="/redoc" target="_blank" class="btn btn-outline-light me-2">ReDoc 規範</a>
                    <a href="/api/v1/metrology/sample-measurements-csv" class="btn btn-success">下載計量 CSV</a>
                </div>
            </header>

            <!-- Control & Upload Panel -->
            <div class="card p-4 mb-4">
                <div class="row g-3 align-items-end">
                    <div class="col-md-5">
                        <label class="form-label text-info fw-bold">1. 上傳測試照片（或點擊下方快速載入 COCO 範例）：</label>
                        <input type="file" id="imageInput" class="form-control bg-dark text-light border-secondary" accept="image/*">
                        <div class="mt-2">
                            <span class="text-secondary small me-2">快速範例：</span>
                            <button type="button" class="sample-btn" onclick="loadSample('coco_000000555705_two_cats_shoes.jpg')">雙貓 (鞋盒)</button>
                            <button type="button" class="sample-btn" onclick="loadSample('coco_000000020247_two_bears.jpg')">雙熊 (野外)</button>
                            <button type="button" class="sample-btn" onclick="loadSample('coco_000000236784_two_dogs_couch.jpg')">雙狗 (沙發)</button>
                            <button type="button" class="sample-btn" onclick="loadSample('coco_000000387383_three_cats.jpg')">三貓 (毯子)</button>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label text-info fw-bold">2. 選擇眼睛分割演算法：</label>
                        <select id="methodSelect" class="form-select bg-dark text-light border-secondary">
                            <option value="vitpose" selected>ViTPose (跨54種動物姿態 - 推薦)</option>
                            <option value="traditional">Traditional (光學暗區+空間對稱矩)</option>
                        </select>
                    </div>
                    <div class="col-md-2">
                        <label class="form-label text-info fw-bold">3. 比例 (mm/px)：</label>
                        <input type="number" id="scaleInput" class="form-control bg-dark text-light border-secondary" value="0.25" step="0.05">
                    </div>
                    <div class="col-md-2">
                        <button id="analyzeBtn" class="btn btn-primary w-100 fw-bold py-2" onclick="runAnalysis()">
                            開始分析量測
                        </button>
                    </div>
                </div>
            </div>

            <!-- Analysis Results Grid -->
            <div class="row g-4 position-relative">
                <div id="spinner" class="spinner-overlay">
                    <div class="text-center text-light">
                        <div class="spinner-border text-info" style="width: 3.5rem; height: 3.5rem;" role="status"></div>
                        <h5 class="mt-3">YOLO & ViTPose 模型正在推論與測量中，請稍候...</h5>
                    </div>
                </div>

                <!-- Left: Visual Image -->
                <div class="col-lg-6">
                    <div class="card p-3 h-100">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h5 class="card-title text-info mb-0">視覺測量標註圖 (Visual Output)</h5>
                            <span id="currentImgLabel" class="badge bg-secondary">coco_000000555705_two_cats_shoes.jpg</span>
                        </div>
                        <div class="text-center my-auto p-2 bg-black rounded border border-secondary" style="min-height: 420px; display: flex; align-items: center; justify-content: center;">
                            <img id="resultImage" src="/outputs/result_coco_000000555705_two_cats_shoes.jpg" 
                                 class="img-fluid rounded shadow" 
                                 alt="Metrology Annotated Output" 
                                 style="max-height: 520px; width: 100%; object-fit: contain;">
                        </div>
                        <div class="small text-muted mt-3">
                            <span class="badge bg-primary me-1">個體遮罩: YOLOv8-seg</span>
                            <span class="badge bg-info text-dark me-1">左眼質心: 青色</span>
                            <span class="badge bg-warning text-dark me-1">右眼質心: 橘色</span>
                            <span class="badge bg-success me-1">個體雙眼瞳距: 綠色連線</span>
                            <span class="badge bg-warning text-dark">跨動物右眼距離: 金色連線</span>
                        </div>
                    </div>
                </div>

                <!-- Right: Quantitative Data Table & Metrics -->
                <div class="col-lg-6">
                    <div class="card p-3 mb-4">
                        <h5 class="card-title text-warning mb-3">實時幾何測量數據 (Real-time Metrology)</h5>
                        
                        <div class="row text-center g-2 mb-3">
                            <div class="col-4">
                                <div class="p-2 bg-dark rounded border border-secondary">
                                    <small class="text-muted d-block">偵測動物數量</small>
                                    <span id="statAnimals" class="h4 text-info">2 隻</span>
                                </div>
                            </div>
                            <div class="col-4">
                                <div class="p-2 bg-dark rounded border border-secondary">
                                    <small class="text-muted d-block">使用演算法</small>
                                    <span id="statMethod" class="h4 text-success">ViTPose</span>
                                </div>
                            </div>
                            <div class="col-4">
                                <div class="p-2 bg-dark rounded border border-secondary">
                                    <small class="text-muted d-block">平均瞳距</small>
                                    <span id="statMeanIntra" class="h4 text-warning">46.08 px</span>
                                </div>
                            </div>
                        </div>

                        <div class="table-responsive">
                            <table class="table table-dark table-sm table-striped table-bordered align-middle mb-0">
                                <thead class="table-secondary text-dark">
                                    <tr>
                                        <th>#</th>
                                        <th>物種</th>
                                        <th>左眼質心 (x, y)</th>
                                        <th>右眼質心 (x, y)</th>
                                        <th>雙眼瞳距 (px / mm)</th>
                                    </tr>
                                </thead>
                                <tbody id="animalTableBody">
                                    <tr>
                                        <td>1</td>
                                        <td><span class="badge bg-success">CAT</span></td>
                                        <td>(242.87, 140.69)</td>
                                        <td>(286.27, 143.06)</td>
                                        <td><strong class="text-success">43.46 px</strong> (10.87 mm)</td>
                                    </tr>
                                    <tr>
                                        <td>2</td>
                                        <td><span class="badge bg-success">CAT</span></td>
                                        <td>(350.56, 111.25)</td>
                                        <td>(399.00, 116.29)</td>
                                        <td><strong class="text-success">48.70 px</strong> (12.18 mm)</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>

                        <div class="mt-3">
                            <h6 class="text-info mb-1">跨動物右眼空間距離 (Inter-animal Right-Eye Distance)：</h6>
                            <div id="interDistList" class="p-2 bg-dark rounded border border-secondary small text-warning">
                                • 動物 #1 到 動物 #2 右眼距離: <strong>115.86 px</strong> (28.97 mm)
                            </div>
                        </div>
                    </div>

                    <!-- Test Accounts & Equations -->
                    <div class="card p-3">
                        <div class="d-flex justify-content-between align-items-center">
                            <h6 class="text-secondary mb-0">內建測試帳號（JWT 認證）：</h6>
                            <span class="badge bg-success">已自動為您授權 (Admin Token Connected)</span>
                        </div>
                        <table class="table table-dark table-sm table-bordered mt-2 mb-0" style="font-size: 0.85rem;">
                            <thead><tr><th>角色</th><th>帳號</th><th>密碼</th><th>權限說明</th></tr></thead>
                            <tbody>
                                <tr><td><span class="badge bg-danger">Admin</span></td><td><code>admin</code></td><td><code>AdminPassword123!</code></td><td>最高權限管理員</td></tr>
                                <tr><td><span class="badge bg-primary">Tester</span></td><td><code>metrology_tester</code></td><td><code>TesterPassword123!</code></td><td>計量測試員</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentToken = null;
            let currentBlob = null;

            // 1. 自動登入取得 JWT Token
            async function ensureToken() {{
                if (currentToken) return currentToken;
                try {{
                    const res = await fetch("/api/v1/auth/login", {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/x-www-form-urlencoded" }},
                        body: "username=admin&password=AdminPassword123!"
                    }});
                    const data = await res.json();
                    currentToken = data.access_token;
                    return currentToken;
                }} catch (e) {{
                    console.error("Auto login failed:", e);
                }}
            }}

            // 2. 快速點選範例圖片
            async function loadSample(fileName) {{
                document.getElementById("currentImgLabel").innerText = fileName;
                const sampleUrl = "/samples/" + fileName;
                try {{
                    const resp = await fetch(sampleUrl);
                    currentBlob = await resp.blob();
                    currentBlob.name = fileName;
                    // 自動觸發分析
                    runAnalysis(currentBlob);
                }} catch (err) {{
                    alert("載入範例圖失敗: " + err);
                }}
            }}

            // 3. 上傳檔案事件
            document.getElementById("imageInput").addEventListener("change", function(e) {{
                if (e.target.files && e.target.files[0]) {{
                    currentBlob = e.target.files[0];
                    document.getElementById("currentImgLabel").innerText = currentBlob.name;
                }}
            }});

            // 4. 執行分析
            async function runAnalysis(blobOverride) {{
                const fileToUpload = blobOverride || currentBlob;
                if (!fileToUpload) {{
                    alert("請先選擇一張照片或點擊範例按鈕！");
                    return;
                }}

                const token = await ensureToken();
                const method = document.getElementById("methodSelect").value;
                const scale = document.getElementById("scaleInput").value || "0.25";

                const formData = new FormData();
                formData.append("file", fileToUpload, fileToUpload.name || "input.jpg");
                formData.append("eye_method", method);
                formData.append("scale_mm_per_px", scale);

                document.getElementById("spinner").style.display = "flex";
                document.getElementById("analyzeBtn").disabled = true;

                try {{
                    const response = await fetch("/api/v1/metrology/analyze-image", {{
                        method: "POST",
                        headers: {{ "Authorization": "Bearer " + token }},
                        body: formData
                    }});

                    if (!response.ok) {{
                        const err = await response.json();
                        throw new Error(err.detail || "分析失敗");
                    }}

                    const result = await response.json();
                    renderResults(result, method);
                }} catch (error) {{
                    alert("分析過程發生錯誤: " + error.message);
                }} finally {{
                    document.getElementById("spinner").style.display = "none";
                    document.getElementById("analyzeBtn").disabled = false;
                }}
            }}

            // 5. 渲染結果到畫面
            function renderResults(data, method) {{
                // 更新圖片 (加上 timestamp 避免快取)
                const imgEl = document.getElementById("resultImage");
                imgEl.src = data.annotated_image_url + "?t=" + new Date().getTime();

                // 更新頂部統計數據
                document.getElementById("statAnimals").innerText = data.total_animals_detected + " 隻";
                document.getElementById("statMethod").innerText = method === "vitpose" ? "ViTPose" : "Traditional";

                // 計算平均雙眼距離
                let sumIntra = 0;
                data.animals.forEach(a => sumIntra += a.intra_eye_distance_px);
                const avgIntra = data.animals.length > 0 ? (sumIntra / data.animals.length).toFixed(2) : "0.00";
                document.getElementById("statMeanIntra").innerText = avgIntra + " px";

                // 更新動物表格
                const tbody = document.getElementById("animalTableBody");
                tbody.innerHTML = "";
                data.animals.forEach(a => {{
                    const tr = document.createElement("tr");
                    const le = a.left_eye.centroid;
                    const re = a.right_eye.centroid;
                    const mmStr = a.intra_eye_distance_mm ? ` (${{a.intra_eye_distance_mm}} mm)` : "";
                    tr.innerHTML = `
                        <td>${{a.animal_id}}</td>
                        <td><span class="badge bg-success">${{a.species.toUpperCase()}}</span></td>
                        <td>(${{le[0]}}, ${{le[1]}})</td>
                        <td>(${{re[0]}}, ${{re[1]}})</td>
                        <td><strong class="text-success">${{a.intra_eye_distance_px}} px</strong>${{mmStr}}</td>
                    `;
                    tbody.appendChild(tr);
                }});

                // 更新跨動物距離清單
                const interDiv = document.getElementById("interDistList");
                if (data.inter_animal_right_eye_distances && data.inter_animal_right_eye_distances.length > 0) {{
                    interDiv.innerHTML = data.inter_animal_right_eye_distances.map(p => {{
                        const mmStr = p.distance_mm ? ` (${{p.distance_mm}} mm)` : "";
                        return `• 動物 #${{p.animal_1_id}} 到 動物 #${{p.animal_2_id}} 右眼距離: <strong>${{p.distance_px}} px</strong>${{mmStr}}`;
                    }}).join("<br>");
                }} else {{
                    interDiv.innerHTML = "無其他配對動物（畫面上動物少於 2 隻）。";
                }}
            }}

            // 頁面初次載入自動初始化 Token
            window.addEventListener("DOMContentLoaded", () => {{
                ensureToken();
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
