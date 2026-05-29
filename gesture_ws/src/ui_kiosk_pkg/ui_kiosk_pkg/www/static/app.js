const el = id => document.getElementById(id);
let st = null;
let last_session = null;

// Simple toast notification
function toast(msg) {
  const t = el("toast");
  if (!t) return;
  t.textContent = msg || "";
  t.style.display = "block";
  setTimeout(() => {
    t.style.display = "none";
  }, 1200);
}

// Video area logic: show idle text when no clip
function setVideo(src) {
  const clip = el("clip");
  const idle = el("idle");
  if (!clip || !idle) return;

  if (src) {
    if (clip.dataset.currentSrc !== src) {
      clip.dataset.currentSrc = src;
      clip.src = src + "?t=" + Date.now();
      clip.load();
      clip.play().catch(e => console.log("Video play failed:", e));
    }
    clip.style.display = "block";
    idle.style.display = "none";
  } else {
    clip.pause();
    clip.removeAttribute("src");
    clip.dataset.currentSrc = "";
    clip.style.display = "none";
    idle.style.display = "block";
  }
}

// History sidebar
function renderHist(history) {
  const box = el("hist");
  if (!box) return;
  box.innerHTML = "";
  (history || []).forEach(it => {
    const d = document.createElement("div");
    d.className = "status";
    const ts = it.ts ? new Date(it.ts).toLocaleTimeString() : "";
    const lbl = it.final_label || it.candidate_label || "";
    d.textContent = `${it.outcome || ""} · ${lbl} · ${ts}`;
    box.appendChild(d);
  });
}

// HUD: direction arrow + text
function updateHud(cmd, ts) {
  const hud = el("hud");
  const arrow = el("hud-arrow");
  const text = el("hud-text");
  if (!hud || !arrow || !text) return;

  if (!cmd || (ts && Date.now() - ts > 3000)) {
    hud.classList.remove("active");
    return;
  }

  let icon = "";
  let cls = "";
  const label = cmd.replace(/_/g, " ").toUpperCase();

  const lc = cmd.toLowerCase();
  if (lc.includes("forward")) { icon = "▲"; cls = "anim-up"; }
  else if (lc.includes("backward")) { icon = "▼"; cls = "anim-down"; }
  else if (lc.includes("left")) { icon = "◀"; cls = "anim-left"; }
  else if (lc.includes("right")) { icon = "▶"; cls = "anim-right"; }

  arrow.textContent = icon;
  arrow.className = "hud-arrow " + cls;
  text.textContent = label;
  hud.classList.add("active");
}

// Fullscreen button
document.addEventListener("DOMContentLoaded", () => {
  const fsBtn = el("fullscreen-btn");
  if (!fsBtn) return;
  fsBtn.addEventListener("click", () => {
    console.log("Fullscreen button clicked");
    if (!document.fullscreenElement) {
      console.log("Requesting fullscreen...");
      document.documentElement.requestFullscreen().catch(err => {
        console.warn("Fullscreen request failed:", err);
      });
    } else {
      console.log("Exiting fullscreen...");
      document.exitFullscreen();
    }
  });

  // Debug Toggle
  const dbgToggle = el("debug-toggle");
  const dbgBox = el("debug-box");
  const dbgImg = el("debug-img");

  if (dbgToggle && dbgBox && dbgImg) {
    dbgToggle.addEventListener("change", () => {
      if (dbgToggle.checked) {
        dbgBox.style.display = "block";
        dbgImg.src = "/stream?" + Date.now(); // bust cache
      } else {
        dbgBox.style.display = "none";
        dbgImg.removeAttribute("src");
      }
    });
  }
});

// Poll /state
async function pull() {
  try {
    const r = await fetch("/state?t=" + Date.now(), {
      headers: { "Cache-Control": "no-cache" }
    });
    if (!r.ok) throw new Error("Response not OK");
    const s = await r.json();
    st = s;

    // Main label + confidence + hint
    if (el("label")) el("label").value = s.label || "";
    if (el("conf")) {
      const c = typeof s.confidence === "number" ? s.confidence : 0;
      el("conf").textContent = `conf ${c.toFixed(2)}`;
    }
    if (el("hint")) {
      el("hint").textContent = s.hint || "";
    }

    // Auto-approve checkbox
    if (el("auto")) {
      el("auto").checked = !!s.auto_approve;
    }

    // Video
    setVideo(s.media_src || null);

    // HUD from last command
    if (s.last_command && s.last_command_ts) {
      updateHud(s.last_command, s.last_command_ts);
    } else {
      updateHud("", 0);
    }

    // Progress bar / status
    const timeout = s.timeout_ms || 0;
    const timeLeft = s.time_left_ms || 0;
    const fill = el("fill");
    const status = el("status");
    if (fill && status) {
      if (timeout > 0) {
        const pct = Math.max(0, Math.min(100, 100 * (timeLeft / timeout)));
        fill.style.width = pct + "%";
        status.textContent = s.label
          ? `${s.label} (${(s.confidence ?? 0).toFixed(2)})`
          : "";
      } else {
        fill.style.width = "0%";
        status.textContent = "";
      }
    }

    // Reset focus when session changes
    if (last_session !== s.session_id) {
      last_session = s.session_id;
      if (el("label")) el("label").focus();
    }

    // History
    renderHist(s.history || []);
  } catch (e) {
    console.error("[PULL] Error:", e);
  }

  setTimeout(pull, 1000);
}

// Send approve/reject
async function send(approved) {
  if (!st || !st.session_id) return;
  const body = {
    session_id: st.session_id,
    approved: !!approved,
    final_label: (el("label") && el("label").value) || ""
  };
  try {
    const r = await fetch("/confirm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    if (r.ok) toast(approved ? "Approved" : "Rejected");
  } catch (e) {
    console.error("[SEND] Error:", e);
  }
}

if (el("approve")) el("approve").onclick = () => send(true);
if (el("reject")) el("reject").onclick = () => send(false);

console.log("[INIT] Starting initial pull");
pull();
