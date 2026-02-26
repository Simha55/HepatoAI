# app/ui.py
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

INDEX_HTML = r"""<!doctype html>
<html>HepatoAI
<head>
  <meta charset="utf-8"/>
  <title>HepatoAI — API Playground</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 1100px; margin: 24px auto; }
    h2 { margin-bottom: 6px; }
    .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; padding: 12px; border-radius: 12px; margin-top: 12px; }
    .muted { color: #666; }
    canvas { max-width: 100%; border: 1px solid #333; margin-top: 12px; border-radius: 8px; }
    pre { background: #0b1020; color: #cfe3ff; padding: 12px; border-radius: 10px; overflow: auto; }
    button { padding: 8px 12px; border-radius: 10px; border: 1px solid #bbb; cursor: pointer; background: #fafafa; }
    button:hover { background: #f0f0f0; }
    input[type="text"] { padding: 8px 10px; border-radius: 10px; border: 1px solid #bbb; min-width: 280px; }
    .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; background:#eef; font-size:12px; }
    .split { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    @media (max-width: 900px) { .split { grid-template-columns: 1fr; } }
    .ok { color: #0a7a2f; }
    .bad { color: #b00020; }
  </style>
</head>
<body>
  <h2>HepatoAI — FastAPI Endpoint Tester</h2>
  <div class="muted">
    Use this page to test: <span class="pill">/health</span> <span class="pill">/metrics</span>
    <span class="pill">/detect</span> <span class="pill">/v1/assets</span> <span class="pill">/v1/assets/{id}</span>
    <span class="pill">/report</span> <span class="pill">/export</span>
  </div>

  <div class="card">
    <div class="row">
      <button id="btnHealth">GET /health</button>
      <button id="btnMetrics">GET /metrics</button>
      <span id="status" class="muted">Ready.</span>
    </div>
  </div>

  <div class="split">
    <!-- DETECT -->
    <div class="card">
      <h3 style="margin-top:0;">Single Image Detect</h3>
      <div class="row">
        <input id="fileSingle" type="file" accept="image/*" />
        <button id="btnDetect">POST /detect</button>
      </div>
      <canvas id="canvas"></canvas>
      <div id="detectInfo" class="muted" style="margin-top:10px;">Pick an image and click Detect.</div>
    </div>

    <!-- ASSETS -->
    <div class="card">
      <h3 style="margin-top:0;">Asset Pipeline (Multi-image)</h3>
      <div class="row">
        <input id="filesMulti" type="file" accept="image/*" multiple />
      </div>
      <div class="row" style="margin-top:10px;">
        <input id="idemKey" type="text" placeholder="Optional Idempotency-Key (e.g., run-123)" />
        <button id="btnCreateAsset">POST /v1/assets</button>
      </div>

      <div class="row" style="margin-top:10px;">
        <input id="assetId" type="text" placeholder="asset_id (auto-filled after create)" />
      </div>
      <div class="row" style="margin-top:10px;">
        <button id="btnGetAsset">GET /v1/assets/{asset_id}</button>
        <button id="btnReport">GET /v1/assets/{asset_id}/report</button>
        <button id="btnExportJson">GET /v1/assets/{asset_id}/export?format=json</button>
        <button id="btnExportCsv">GET /v1/assets/{asset_id}/export?format=csv</button>
      </div>

      <div id="assetInfo" class="muted" style="margin-top:10px;">Upload multiple images to create an asset, then fetch status/report/export.</div>
    </div>
  </div>

  <div class="card">
    <h3 style="margin-top:0;">Response Viewer</h3>
    <pre id="out">(responses will appear here)</pre>
  </div>

<script>
  const statusEl = document.getElementById('status');
  const outEl = document.getElementById('out');

  function setStatus(msg, ok=true) {
    statusEl.className = ok ? "ok" : "bad";
    statusEl.textContent = msg;
  }

  function show(obj) {
    if (typeof obj === "string") outEl.textContent = obj;
    else outEl.textContent = JSON.stringify(obj, null, 2);
  }

  async function safeFetch(url, opts={}) {
    const t0 = performance.now();
    try {
      const res = await fetch(url, opts);
      const t1 = performance.now();
      const ct = res.headers.get("content-type") || "";
      let body;
      if (ct.includes("application/json")) body = await res.json();
      else body = await res.text();

      if (!res.ok) {
        setStatus(`${opts.method || "GET"} ${url} -> ${res.status}`, false);
        show(body);
        return { ok:false, status:res.status, body, ms:(t1-t0) };
      }

      setStatus(`${opts.method || "GET"} ${url} -> ${res.status} (${(t1-t0).toFixed(1)}ms)`, true);
      show(body);
      return { ok:true, status:res.status, body, ms:(t1-t0) };
    } catch (e) {
      setStatus(`Fetch failed: ${e}`, false);
      show(String(e));
      return { ok:false, status:0, body:String(e), ms:0 };
    }
  }

  // --- health / metrics ---
  document.getElementById('btnHealth').onclick = () => safeFetch("/health");
  document.getElementById('btnMetrics').onclick = () => safeFetch("/metrics");

  // --- single detect + draw boxes ---
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const detectInfo = document.getElementById('detectInfo');
  const fileSingle = document.getElementById('fileSingle');

  let img = new Image();
  let fileObj = null;

  fileSingle.addEventListener('change', () => {
    const f = fileSingle.files && fileSingle.files[0];
    if (!f) return;
    fileObj = f;
    const url = URL.createObjectURL(f);
    img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(img, 0, 0);
    };
    img.src = url;
    detectInfo.textContent = `Loaded: ${f.name} (${f.size} bytes)`;
  });

  function draw(dets) {
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = 3;
    ctx.font = "16px sans-serif";
    dets.forEach(d => {
      const [x1,y1,x2,y2] = d.box;
      ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      const name = d.label_name ?? d.label;
      const text = `${name} ${(d.score*100).toFixed(1)}%`;
      const ty = Math.max(18, y1 - 6);
      ctx.fillText(text, x1 + 4, ty);
    });
  }

  document.getElementById('btnDetect').onclick = async () => {
    if (!fileObj) { alert("Pick an image first"); return; }

    const fd = new FormData();
    fd.append("file", fileObj);

    const res = await safeFetch("/detect", { method: "POST", body: fd });
    if (res.ok && res.body && res.body.detections) {
      draw(res.body.detections);
      detectInfo.innerHTML =
        `<div><b>Backend:</b> ${res.body.backend}</div>` +
        `<div><b>Total latency:</b> ${res.body.total_latency_ms?.toFixed?.(1) ?? "?"} ms</div>` +
        `<div><b>Batch latency:</b> ${res.body.batch_latency_ms?.toFixed?.(1) ?? "?"} ms</div>` +
        `<div><b>Detections:</b> ${res.body.detections.length}</div>`;
    }
  };

  // --- assets ---
  const filesMulti = document.getElementById('filesMulti');
  const idemKey = document.getElementById('idemKey');
  const assetId = document.getElementById('assetId');
  const assetInfo = document.getElementById('assetInfo');

  document.getElementById('btnCreateAsset').onclick = async () => {
    const files = filesMulti.files;
    if (!files || files.length === 0) { alert("Pick 1+ images"); return; }

    const fd = new FormData();
    for (const f of files) fd.append("files", f);

    const headers = {};
    if (idemKey.value.trim().length > 0) headers["Idempotency-Key"] = idemKey.value.trim();

    const res = await safeFetch("/v1/assets", { method: "POST", body: fd, headers });
    if (res.ok && res.body && res.body.asset_id) {
      assetId.value = res.body.asset_id;
      assetInfo.innerHTML = `Created asset_id: <b>${res.body.asset_id}</b> (status: ${res.body.status})`;
    }
  };

  function requireAssetId() {
    const id = assetId.value.trim();
    if (!id) { alert("asset_id is empty"); return null; }
    return id;
  }

  document.getElementById('btnGetAsset').onclick = async () => {
    const id = requireAssetId(); if (!id) return;
    await safeFetch(`/v1/assets/${encodeURIComponent(id)}`);
  };

  document.getElementById('btnReport').onclick = async () => {
    const id = requireAssetId(); if (!id) return;
    await safeFetch(`/v1/assets/${encodeURIComponent(id)}/report`);
  };

  document.getElementById('btnExportJson').onclick = async () => {
    const id = requireAssetId(); if (!id) return;
    await safeFetch(`/v1/assets/${encodeURIComponent(id)}/export?format=json`);
  };

  document.getElementById('btnExportCsv').onclick = async () => {
    const id = requireAssetId(); if (!id) return;
    const res = await safeFetch(`/v1/assets/${encodeURIComponent(id)}/export?format=csv`);
    // nice-to-have: download csv as file
    if (res.ok && typeof res.body === "string") {
      const blob = new Blob([res.body], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `asset_${id}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };
</script>

</body>
</html>
"""

@router.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML