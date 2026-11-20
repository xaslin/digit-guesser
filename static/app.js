const pad = document.getElementById("pad");
const ctx = pad.getContext("2d");
const result = document.getElementById("result");
const guessBtn = document.getElementById("guess");

function reset() {
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, pad.width, pad.height);
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  result.textContent = "";
}
reset();

let drawing = false;
let last = null;

function pos(e) {
  const r = pad.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) * (pad.width / r.width),
    y: (e.clientY - r.top) * (pad.height / r.height),
  };
}

pad.addEventListener("pointerdown", (e) => {
  drawing = true;
  pad.setPointerCapture(e.pointerId);
  last = pos(e);
  // draw a dot so a single tap leaves a mark
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.lineTo(last.x + 0.01, last.y);
  ctx.stroke();
});

pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = pos(e);
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  last = p;
});

const stop = () => { drawing = false; };
pad.addEventListener("pointerup", stop);
pad.addEventListener("pointercancel", stop);

document.getElementById("clear").addEventListener("click", reset);

guessBtn.addEventListener("click", async () => {
  guessBtn.disabled = true;
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: pad.toDataURL("image/png") }),
    });
    const data = await res.json();
    result.replaceChildren();
    if (!res.ok) {
      const p = document.createElement("div");
      p.className = "error";
      p.textContent = data.error || "Something went wrong.";
      result.append(p);
    } else if (data.uncertain) {
      const p = document.createElement("div");
      p.className = "error";
      p.textContent = "I'm not sure what that is. Please draw another number.";
      result.append(p);
    } else {
      const d = document.createElement("div");
      d.className = "digit";
      d.textContent = data.digit;
      const c = document.createElement("div");
      c.className = "conf";
      c.textContent = `I think it's a ${data.digit}`;
      const pct = Math.round(data.confidence * 100);
      const level = pct >= 80 ? "high" : pct >= 50 ? "mid" : "low";
      const meter = document.createElement("div");
      meter.className = "meter";
      const label = document.createElement("div");
      label.className = "meter-label";
      const name = document.createElement("span");
      name.textContent = "Confidence";
      const value = document.createElement("span");
      value.textContent = `${pct}%`;
      label.append(name, value);
      const track = document.createElement("div");
      track.className = "meter-track";
      track.setAttribute("role", "meter");
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", "100");
      track.setAttribute("aria-valuenow", pct);
      const fill = document.createElement("div");
      fill.className = `meter-fill ${level}`;
      track.append(fill);
      meter.append(label, track);
      result.append(d, c, meter);
      requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${pct}%`; }));
    }
  } catch {
    result.textContent = "Could not reach the server.";
  } finally {
    guessBtn.disabled = false;
  }
});
