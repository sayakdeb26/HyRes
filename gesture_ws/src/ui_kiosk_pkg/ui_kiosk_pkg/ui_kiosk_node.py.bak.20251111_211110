#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json, threading, time
from urllib.parse import urlparse, parse_qs

import rclpy
from rclpy.node import Node

from vlm_interfaces.msg import ConfirmRequest, ConfirmReply

def now_s():
    return time.time()

class KioskState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {"label":"", "confidence":0.0, "hint":""}
        self.auto_approve = False
        self.deadline_s = 20.0
        self.active_until = 0.0  # epoch seconds when this request expires

    def update_from_req(self, req: ConfirmRequest, deadline_s: float):
        with self.lock:
            label = getattr(req, "candidate_label", None) or getattr(req, "label", "")
            conf = getattr(req, "candidate_conf", None)
            if conf is None:
                conf = getattr(req, "confidence", 0.0)
            hint = getattr(req, "hint", "")
            self.latest.update({"label":label, "confidence":float(conf), "hint":hint})
            self.deadline_s = float(deadline_s)
            self.active_until = now_s() + self.deadline_s

    def set_auto(self, v: bool):
        with self.lock:
            self.auto_approve = bool(v)

    def clear(self):
        with self.lock:
            self.latest = {"label":"", "confidence":0.0, "hint":""}
            self.active_until = 0.0

    def get_state(self):
        with self.lock:
            tl = max(0.0, self.active_until - now_s())
            active = tl > 0.0
            return {
                "latest": dict(self.latest),
                "auto_approve": self.auto_approve,
                "deadline_s": self.deadline_s,
                "time_left": round(tl, 1),
                "active": active
            }

class UIHTTP(BaseHTTPRequestHandler):
    kiosk = None  # set by node

    def _reply(self, code, body, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        try:
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def log_message(self, fmt, *args):  # quiet logs
        return

    def do_GET(self):
        k = self.kiosk
        path = urlparse(self.path).path
        if path == "/":
            html = """<!doctype html>
<meta charset="utf-8"/>
<title>AMR UI Kiosk</title>
<style>
  body{font-family:system-ui,Arial;margin:24px}
  .row{margin:8px 0}
  button{padding:8px 12px}
  .dim{opacity:0.6}
</style>
<h2>KOLAMeRo UI</h2>
<div class="row">Label: <span id="label"></span></div>
<div class="row">Confidence: <span id="conf"></span></div>
<div class="row">Hint: <span id="hint"></span></div>
<div class="row">Time left: <span id="tleft">0.0</span> s</div>
<div class="row">
  <button id="approve">Approve</button>
  <button id="reject">Reject</button>
  <label style="margin-left:16px;">
    <input type="checkbox" id="auto"> Auto-approve
  </label>
</div>
<script>
async function refresh(){
  const s = await fetch('/state').then(r=>r.json());
  const has = !!s.active;
  document.getElementById('label').textContent = has ? (s.latest.label||'') : '';
  document.getElementById('conf').textContent = has ? Number(s.latest.confidence||0).toFixed(2) : '0.00';
  document.getElementById('hint').textContent = has ? (s.latest.hint||'') : '';
  document.getElementById('tleft').textContent = has ? s.time_left.toFixed(1) : '0.0';
  document.getElementById('auto').checked = !!s.auto_approve;
  document.getElementById('approve').disabled = !has;
  document.getElementById('reject').disabled = !has;
}
async function post(url){ try{ await fetch(url,{method:'POST'});}catch(e){} }
document.addEventListener('DOMContentLoaded', ()=>{
  document.getElementById('approve').addEventListener('click', async ()=>{
    await post('/confirm?approved=1');
    await refresh(); // instant refresh on click
  });
  document.getElementById('reject').addEventListener('click', async ()=>{
    await post('/confirm?approved=0');
    await refresh(); // instant refresh on click
  });
  document.getElementById('auto').addEventListener('change', async (e)=>{
    await post('/toggle_auto?value='+(e.target.checked?1:0));
  });
  setInterval(refresh, 1000); // keep polling every second
  refresh();
});
</script>
"""
            self._reply(200, html, "text/html; charset=utf-8")
            return
        if path == "/state":
            self._reply(200, json.dumps(k.state.get_state()))
            return
        self._reply(404, json.dumps({"error":"not found"}))

    def do_POST(self):
        k = self.kiosk
        u = urlparse(self.path)
        if u.path == "/confirm":
            q = parse_qs(u.query)
            approved = bool(int(q.get("approved", ["0"])[0]))
            k.publish_confirm(approved)
            # Clear current candidate immediately after a decision
            k.state.clear()
            self._reply(200, json.dumps({"ok": True}))
            return
        if u.path == "/toggle_auto":
            q = parse_qs(u.query)
            v = bool(int(q.get("value", ["0"])[0]))
            k.state.set_auto(v)
            self._reply(200, json.dumps({"auto_approve": v}))
            return
        self._reply(404, json.dumps({"error":"not found"}))

class KioskNode(Node):
    def __init__(self):
        super().__init__("ui_kiosk_node")
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8008)
        self.declare_parameter("auto_approve", False)
        self.declare_parameter("deadline_s", 20.0)

        self.host = self.get_parameter("host").get_parameter_value().string_value
        self.port = int(self.get_parameter("port").get_parameter_value().integer_value)

        self.state = KioskState()
        self.state.set_auto(self.get_parameter("auto_approve").get_parameter_value().bool_value)
        self.state.deadline_s = float(self.get_parameter("deadline_s").get_parameter_value().double_value)

        self.pub_reply = self.create_publisher(ConfirmReply, "/ui/confirm_reply", 10)
        self.sub_req = self.create_subscription(ConfirmRequest, "/vlm/confirm_request", self.on_confirm_request, 10)

        UIHTTP.kiosk = self
        self.http = ThreadingHTTPServer((self.host, self.port), UIHTTP)
        import threading
        self.http_thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.http_thread.start()
        self.get_logger().info(f"UI kiosk serving on http://{self.host}:{self.port}")

    def on_confirm_request(self, req: ConfirmRequest):
        self.state.update_from_req(req, self.state.deadline_s)
        # Optional: auto-approve if enabled
        if self.state.auto_approve:
            self.publish_confirm(True)
            self.state.clear()

    def publish_confirm(self, approved: bool):
        msg = ConfirmReply()
        if hasattr(msg, "approved"):
            msg.approved = bool(approved)
        # send back the chosen label
        label = self.state.latest.get("label","")
        if hasattr(msg, "final_label"):
            msg.final_label = label
        elif hasattr(msg, "label"):
            msg.label = label
        self.pub_reply.publish(msg)
        self.get_logger().info(f"ConfirmReply sent (approved={approved}, label={label})")

def main():
    rclpy.init()
    node = KioskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, "http"):
            node.http.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
