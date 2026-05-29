#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import os, sys, json, time, shutil, threading, mimetypes, urllib.parse, socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from amr_interfaces.msg import ConfirmRequest, ConfirmReply, TelemetryCommand
import cv2
import base64
from cv_bridge import CvBridge

def utc_ts():
    return int(time.time()*1000)

def safe_basename(p:str)->str:
    return os.path.basename(p).replace('\\','/')

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.processing = False  # VLM is currently processing
        self.session_id = ""
        self.candidate_label = ""
        self.candidate_conf = 0.0
        self.hint = ""                  # optional (leave blank if none)
        self.clip_src = ""              # browser path like /media/<file>.mp4
        self.clip_abspath = ""          # on-disk file in media_dir
        self.deadline_at_ms = 0
        self.auto_approve = False
        self.history = []               # list of dicts (max history_size)
        self.time_recv_ms = 0
        self.last_command = ""
        self.last_command_ts = 0
        self.latest_debug_jpg = None  # For MJPEG stream
        


class UiKiosk(Node):
    def __init__(self):
        super().__init__('ui_kiosk_node')
        p = self.declare_parameter
        self.http_host           = p('http_host', '0.0.0.0').value
        self.http_port           = int(p('http_port', 8008).value)
        self.media_dir           = p('media_dir', '/home/sayak/amr_kiosk_media').value
        self.history_size        = int(p('history_size', 5).value)
        self.video_w             = int(p('video_width', 960).value)
        self.video_h             = int(p('video_height', 720).value)
        self.decision_timeout_s  = float(p('decision_timeout_s', 20.0).value)
        self.keep_days_approved  = int(p('keep_days_approved', 7).value)
        self.janitor_interval_s  = int(p('janitor_interval_s', 3600).value)
        self.auto_approve        = p('auto_approve_default', False).value

        self.state = State()
        self.state.auto_approve = self.auto_approve
        
        self.bridge = CvBridge()

        # Ensure dirs
        os.makedirs(self.media_dir, exist_ok=True)
        os.makedirs(os.path.join(self.media_dir, "approved"), exist_ok=True)

        # HTTP Server in thread
        self.server = HTTPServer((self.http_host, self.http_port), KioskHandler)
        # inject node ref into handler class
        global node_ref
        node_ref = self
        
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.get_logger().info(f"UI Kiosk running at http://{self.http_host}:{self.http_port}")

        # Janitor thread
        self.janitor_thread = threading.Thread(target=self._janitor_loop)
        self.janitor_thread.daemon = True
        self.janitor_thread.start()

        # Timer for auto-approve check
        self.create_timer(1.0, self._tick)

        # Subscribers
        self.sub_req = self.create_subscription(ConfirmRequest, "/vlm/confirm_request", self.on_confirm_request, 10)
        self.sub_telemetry = self.create_subscription(TelemetryCommand, "/telemetry/command", self.on_telemetry, 10)
        self.sub_debug = self.create_subscription(Image, "/lstm/debug_feed", self.on_debug_feed, 10)

        # Publisher for reply
        self.pub_reply = self.create_publisher(ConfirmReply, "/ui/confirm_reply", 10)

    def on_telemetry(self, msg: TelemetryCommand):
        with self.state.lock:
            self.state.last_command = msg.command_text
            self.state.last_command_ts = utc_ts()

    def on_debug_feed(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Compress to JPEG
            _, jpg = cv2.imencode('.jpg', cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            with self.state.lock:
                self.state.latest_debug_jpg = jpg.tobytes()
        except Exception:
            pass

    def _html_index(self):
        # We serve the static index.html from disk
        path = os.path.join(os.path.dirname(__file__), "www", "index.html")
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error loading index.html: {e}"

    def _serve_static(self, handler:BaseHTTPRequestHandler):
        # serve from www/static
        base = Path(__file__).parent / "www"
        base_resolved = base.resolve()
        
        # sanitize
        req_path = handler.path.split('?')[0]
        # remove leading /
        if req_path.startswith('/'): req_path = req_path[1:]
        
        # resolve
        fpath = (base / req_path).resolve()
        
        # security check: must be inside www
        if not str(fpath).startswith(str(base_resolved)):
            return handler._send(403, {}, b"Forbidden")
            
        if not fpath.exists():
            return handler._send(404, {}, b"Not Found")
            
        # guess mime
        ctype, _ = mimetypes.guess_type(fpath)
        if not ctype: ctype = "application/octet-stream"
        
        try:
            with open(fpath, "rb") as f:
                content = f.read()
            handler._send(200, {"Content-Type":ctype}, content)
        except Exception as e:
            handler._send(500, {}, str(e).encode("utf-8"))

    def _serve_media(self, handler:BaseHTTPRequestHandler):
        # serve from self.media_dir
        # path is /media/filename
        fname = handler.path.split('/')[-1].split('?')[0]
        fpath = os.path.join(self.media_dir, fname)
        
        if not os.path.exists(fpath):
            return handler._send(404, {}, b"Not Found")
            
        ctype, _ = mimetypes.guess_type(fpath)
        if not ctype: ctype = "application/octet-stream"
        
        try:
            with open(fpath, "rb") as f:
                content = f.read()
            handler._send(200, {"Content-Type":ctype}, content)
        except Exception as e:
            handler._send(500, {}, str(e).encode("utf-8"))

    def _serve_stream(self, handler:BaseHTTPRequestHandler):
        # MJPEG stream
        handler.send_response(200)
        handler.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        handler.end_headers()
        
        try:
            while True:
                jpg_data = None
                with self.state.lock:
                    jpg_data = self.state.latest_debug_jpg
                
                if jpg_data:
                    handler.wfile.write(b'--frame\r\n')
                    handler.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    handler.wfile.write(jpg_data)
                    handler.wfile.write(b'\r\n')
                else:
                    # If no data yet, send a 1x1 black pixel or wait
                    # Minimal black pixel jpeg
                    handler.wfile.write(b'--frame\r\n')
                    handler.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    # 1x1 black
                    # handler.wfile.write(b'\xFF\xD8...') 
                    # Just wait a bit instead of spamming junk
                    pass

                time.sleep(0.1) # Max 10 FPS
        except Exception:
            pass # Client disconnected


    def _serve_state(self, handler:BaseHTTPRequestHandler):
        # self.get_logger().info("[STATE ENDPOINT CALLED]")
        with self.state.lock:
            now = utc_ts()
            # Only calculate time left if deadline is set (not processing)
            time_left = 0
            if self.state.active and self.state.deadline_at_ms > 0:
                time_left = max(0, self.state.deadline_at_ms - now)
            
            active = self.state.active
            label = self.state.candidate_label
            conf = self.state.candidate_conf
            clip = self.state.clip_src
            
        # self.get_logger().info(f"Serving /state: active={active}, label={label}, conf={conf}, clip={clip}")
        
        with self.state.lock:
            payload = {
                "session_id": self.state.session_id if self.state.active else "",
                "label": self.state.candidate_label if self.state.active else "",
                "confidence": float(self.state.candidate_conf) if self.state.active else 0.0,
                "hint": self.state.hint if self.state.active else "",
                "media_src": self.state.clip_src if self.state.active else "",
                "auto_approve": self.state.auto_approve,
                "timeout_ms": int(self.decision_timeout_s * 1000) if self.state.deadline_at_ms > 0 else 0,
                "time_left_ms": time_left,
                "history": self.state.history[-self.history_size:],
                "last_command": self.state.last_command,
                "last_command_ts": self.state.last_command_ts
            }
        body = json.dumps(payload).encode("utf-8")
        handler._send(200, {"Content-Type":"application/json"}, body)

    def _handle_confirm(self, handler:BaseHTTPRequestHandler, data:dict):
        sid = data.get("session_id", "")
        approved = data.get("approved", False)
        final_lbl = data.get("final_label", "")
        
        self.get_logger().info(f"Received /confirm: sid={sid}, approved={approved}, label={final_lbl}")
        
        with self.state.lock:
            if not self.state.active or self.state.session_id != sid:
                return handler._send(400, {}, b"Session mismatch or inactive")
                
            # Publish reply
            msg = ConfirmReply()
            msg.session_id = sid
            msg.approved = bool(approved)
            msg.final_label = final_lbl
            self.pub_reply.publish(msg)
            
            # Log history
            outcome = "approved" if approved else "rejected"
            self._push_history({"ts":utc_ts(), "session_id":sid, "outcome":outcome,
                 "candidate_label":self.state.candidate_label, "final_label":final_lbl,
                 "candidate_conf":self.state.candidate_conf,
                 "latency_ms": self.decision_timeout_s*1000 - (self.state.deadline_at_ms - utc_ts()),
                 "clip_name": safe_basename(self.state.clip_abspath)})
                 
            # Handle file retention
            try:
                clip_abs = self.state.clip_abspath
                if approved:
                    # move to approved/
                    dst = os.path.join(self.media_dir, "approved", safe_basename(clip_abs))
                    if clip_abs and os.path.exists(clip_abs):
                        if os.path.abspath(clip_abs) != os.path.abspath(dst):
                            shutil.move(clip_abs, dst)
                else:
                    # delete immediately
                    if clip_abs and os.path.exists(clip_abs):
                        os.remove(clip_abs)
            except Exception as e:
                self.get_logger().warn(f"retention op failed: {e}")

            # Clear state
            self._clear_active()
            
        handler._send(200, {"Content-Type":"application/json"}, json.dumps({"ok":True}).encode("utf-8"))

    def _push_history(self, item):
        self.state.history.append(item)
        if len(self.state.history) > self.history_size:
            self.state.history = self.state.history[-self.history_size:]

    def _clear_active(self):
        self.state.active = False
        self.state.session_id = ""
        self.state.candidate_label = ""
        self.state.candidate_conf = 0.0
        self.state.hint = ""
        self.state.clip_src = ""
        self.state.clip_abspath = ""
        self.state.deadline_at_ms = 0
        self.state.time_recv_ms = 0

    # -------- ROS callbacks --------


    def on_confirm_request(self, msg: ConfirmRequest):
        session_id = msg.session_id
        label = msg.candidate_label
        conf = msg.candidate_conf
        
        # Try to find the clip
        recorder_dir = "/home/sayak/amr_gesture_ws/data/runtime_clips"
        src_path = ""
        
        try:
            for f in os.listdir(recorder_dir):
                if session_id in f and f.endswith(".mp4"):
                    src_path = os.path.join(recorder_dir, f)
                    break
        except Exception:
            pass
            
        if not src_path:
            self.get_logger().warn(f"Could not find clip for session {session_id}")
            return

        try:
            # 1) Copy the MP4 into media_dir (for retention / training)
            base = safe_basename(src_path)
            dst_mp4 = os.path.join(self.media_dir, base)
            if os.path.abspath(src_path) != os.path.abspath(dst_mp4):
                shutil.copy2(src_path, dst_mp4)

            # 2) Extract middle frame as JPEG preview
            preview_name = base.rsplit(".", 1)[0] + "_preview.jpg"
            preview_path = os.path.join(self.media_dir, preview_name)

            try:
                cap = cv2.VideoCapture(dst_mp4)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                mid = total // 2 if total > 0 else 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ok, frame = cap.read()
                cap.release()
                if ok:
                    cv2.imwrite(preview_path, frame)
                else:
                    preview_path = ""  # fallback to MP4 thumbnail if needed
            except Exception as e:
                self.get_logger().warn(f"preview extract failed: {e}")
                preview_path = ""

            # 3) Update kiosk state (point UI at the preview frame)
            with self.state.lock:
                self.state.active = True
                self.state.session_id = session_id
                self.state.candidate_label = label
                self.state.candidate_conf = conf
                self.state.hint = msg.hint
                self.state.clip_abspath = dst_mp4
                # If preview exists, serve that; otherwise, still show the MP4
                # Always use MP4 for video playback
                self.state.clip_src = "/media/" + safe_basename(dst_mp4)
                self.state.time_recv_ms = utc_ts()
                
                # TIMER LOGIC:
                # If label is "PROCESSING...", do NOT start timer yet.
                if label == "PROCESSING...":
                    self.state.deadline_at_ms = 0 # No timeout
                else:
                    # Real label arrived, start 30s countdown
                    self.state.deadline_at_ms = self.state.time_recv_ms + int(self.decision_timeout_s * 1000)

            self.get_logger().info(
                f"STATE SET: active=True, label={label}, conf={conf:.2f}, clip_src={self.state.clip_src}"
            )
            self.get_logger().info(
                f"ConfirmRequest {session_id} → {base} (conf={conf:.2f})"
            )

        except Exception as e:
            self.get_logger().error(f"on_confirm_request error: {e}")

    def _tick(self):
        # auto-approve on timeout (if auto_approve ON)
        with self.state.lock:
            if not self.state.active: return
            
            # If deadline is 0, we are waiting for VLM processing, so do nothing
            if self.state.deadline_at_ms == 0:
                return

            now = utc_ts()
            if now >= self.state.deadline_at_ms:
                if self.state.auto_approve:
                    # synthesize approve with current label
                    msg = ConfirmReply()
                    msg.session_id = self.state.session_id
                    msg.approved = True
                    msg.final_label = self.state.candidate_label
                    self.pub_reply.publish(msg)
                    self._push_history({"ts":utc_ts(), "session_id":self.state.session_id,
                        "outcome":"auto-approved", "candidate_label":self.state.candidate_label,
                        "final_label":self.state.candidate_label, "candidate_conf":self.state.candidate_conf,
                        "latency_ms": self.decision_timeout_s*1000, "clip_name": safe_basename(self.state.clip_abspath)})
                    # keep for 1 day; move to approved/
                    try:
                        dst = os.path.join(self.media_dir, "approved", safe_basename(self.state.clip_abspath))
                        if self.state.clip_abspath and os.path.exists(self.state.clip_abspath):
                            if os.path.abspath(self.state.clip_abspath) != os.path.abspath(dst):
                                shutil.move(self.state.clip_abspath, dst)
                    except Exception as e:
                        self.get_logger().warn(f"auto-approve retention move failed: {e}")
                    self._clear_active()
                else:
                    # timeout → delete immediately
                    # FIX: Publish reply so bridge knows session ended
                    msg = ConfirmReply()
                    msg.session_id = self.state.session_id
                    msg.approved = False
                    msg.final_label = "" # or "TIMEOUT"
                    self.pub_reply.publish(msg)
                    
                    try:
                        if self.state.clip_abspath and os.path.exists(self.state.clip_abspath):
                            os.remove(self.state.clip_abspath)
                    except Exception as e:
                        self.get_logger().warn(f"timeout delete failed: {e}")
                    self._push_history({"ts":utc_ts(), "session_id":self.state.session_id,
                        "outcome":"timed-out", "candidate_label":self.state.candidate_label,
                        "final_label":"", "candidate_conf":self.state.candidate_conf,
                        "latency_ms": self.decision_timeout_s*1000, "clip_name": safe_basename(self.state.clip_abspath)})
                    self._clear_active()

    def _janitor_loop(self):
        while rclpy.ok():
            try:
                # delete approved older than keep_days_approved
                cutoff = time.time() - self.keep_days_approved*86400
                for root, dirs, files in os.walk(self.media_dir):
                    for f in files:
                        path = os.path.join(root, f)
                        if "approved/keep" in path:  # preserved by training
                            continue
                        try:
                            st = os.stat(path)
                            if st.st_mtime < cutoff:
                                os.remove(path)
                        except Exception:
                            pass
            except Exception as e:
                self.get_logger().warn(f"janitor: {e}")
            time.sleep(self.janitor_interval_s)

class KioskHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index.html"):
            body = node_ref._html_index().encode("utf-8")
            return self._send(200, {"Content-Type":"text/html; charset=utf-8"}, body)
        elif self.path.startswith("/static/"):
            return node_ref._serve_static(self)
        elif self.path.startswith("/state"):
            return node_ref._serve_state(self)
        elif self.path.startswith("/media/"):
            return node_ref._serve_media(self)
        elif self.path.startswith("/stream"):
            return node_ref._serve_stream(self)
        else:
            return self._send(404, {"Content-Type":"text/plain"}, b"not found")

    def do_POST(self):
        if self.path.startswith("/confirm"):
            length = int(self.headers.get('Content-Length','0'))
            raw = self.rfile.read(length) if length>0 else b""
            try:
                data = json.loads(raw.decode("utf-8") if raw else "{}")
            except Exception:
                data = {}
            return node_ref._handle_confirm(self, data)
        elif self.path.startswith("/toggle_auto"):
            with node_ref.state.lock:
                node_ref.state.auto_approve = not node_ref.state.auto_approve
            return self._send(200, {"Content-Type":"application/json"}, b"{\"ok\":true}")
        else:
            return self._send(404, {}, b"not found")

    def _send(self, code, headers, body):
        self.send_response(code)
        for k,v in headers.items():
            self.send_header(k,v)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quieter
        node_ref.get_logger().debug("HTTP: " + fmt % args)

def main():
    rclpy.init()
    node = UiKiosk()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == "__main__":
    main()
