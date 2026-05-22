/** Shared client for Ventilator FIWARE dashboards (Handloom-style relay pattern) */
const RELAY = window.VENT_RELAY || "http://127.0.0.1:5050";
const API = window.VENT_API || "http://127.0.0.1:8000";

const fmt = (v, d = 1) =>
  v === null || v === undefined || Number.isNaN(Number(v)) ? "--" : Number(v).toFixed(d);

async function relay(path, options = {}) {
  const res = await fetch(`${RELAY}${path}`, options);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

async function checkHealth() {
  try {
    const h = await relay("/health");
    return {
      relay: true,
      api: h.digital_twin_api === "ok",
      fiware: h.fiware?.reachable === true,
      demo: h.demo_mode,
    };
  } catch {
    return { relay: false, api: false, fiware: false, demo: true };
  }
}

function bindClock(el) {
  const tick = () => {
    el.textContent = new Date().toLocaleTimeString();
  };
  tick();
  setInterval(tick, 1000);
}

function renderEventLog(container, events) {
  if (!container) return;
  container.innerHTML = (events || [])
    .slice(0, 40)
    .map((e) => `<div class="logLine"><span>${e.ts}</span> [${e.kind}] ${e.message}</div>`)
    .join("");
}

function drawSparkline(canvas, values, color = "#0984f8") {
  if (!canvas || !values?.length) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.shadowColor = color;
  ctx.shadowBlur = 8;
  ctx.beginPath();
  values.forEach((v, i) => {
    const x = (i / (values.length - 1 || 1)) * (w - 8) + 4;
    const y = h - 4 - ((v - min) / span) * (h - 8);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;
}

window.VentFiware = {
  RELAY,
  API,
  fmt,
  relay,
  checkHealth,
  bindClock,
  renderEventLog,
  drawSparkline,
};
