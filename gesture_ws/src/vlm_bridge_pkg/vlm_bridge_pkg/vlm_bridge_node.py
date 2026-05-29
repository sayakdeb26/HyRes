#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
VLM Bridge (session-locked):
- On /lstm/unknown: freeze pipeline, compute ROI from /lstm/keypoints_window,
  request recorder [t-7s, t+7s], wait for clip, call VLM once, show UI, resume.
- Publishes /pipeline/hold (Bool, transient_local).
- Backward compatible with legacy JSON /recorder/clip_ready.
"""

import json
import time
import threading
import uuid
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    DurabilityPolicy,
    ReliabilityPolicy,
    HistoryPolicy,
)

import base64
import cv2

from std_msgs.msg import Bool, String
from builtin_interfaces.msg import Time as RosTime

from vlm_interfaces.msg import (
    UnknownGesture,
    KeypointsWindow,
    RecorderRequest,
    ClipReady,
    ConfirmRequest,
    ConfirmReply,
)
from vlm_interfaces.srv import InferClip

# 7 seconds before and after the center timestamp
WINDOW_PRE = 7.0
WINDOW_POST = 7.0


class VlmBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("bridge_node")

        # ---- params ----
        self.declare_parameter("confirm_timeout_s", 15.0)
        self.declare_parameter("wait_clip_timeout_s", 5.0)
        self.confirm_timeout_s = float(
            self.get_parameter("confirm_timeout_s").value
        )
        self.wait_clip_timeout_s = float(
            self.get_parameter("wait_clip_timeout_s").value
        )

        # ---- QoS ----
        qos_sub = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        qos_pub = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        qos_hold = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # ---- state ----
        #self.session_id: Optional[str] = None
        #self.latest_kp: Optional[KeypointsWindow] = None
        #self.clip_for_session: Optional[str] = None
        #self.latest_reply: Optional[ConfirmReply] = None
        self.session_active: bool = False           # are we currently handling one unknown?
        self.session_id: str = ""

        # ---- subscribers ----
        self.sub_unknown = self.create_subscription(
            UnknownGesture, "/lstm/unknown", self.on_unknown, qos_sub
        )
        self.sub_kp = self.create_subscription(
            KeypointsWindow, "/lstm/keypoints_window", self.on_kp, qos_sub
        )
        self.sub_ready_msg = self.create_subscription(
            ClipReady, "/recorder/clip_ready_msg", self.on_clip_ready_msg, qos_sub
        )
        # legacy JSON path
        self.sub_ready_json = self.create_subscription(
            String, "/recorder/clip_ready", self.on_clip_ready_json, qos_sub
        )
        self.sub_reply = self.create_subscription(
            ConfirmReply, "/vlm/confirm_reply", self.on_confirm_reply, qos_sub
        )

        # ---- publishers ----
        self.pub_hold = self.create_publisher(Bool, "/pipeline/hold", qos_hold)
        self.pub_req = self.create_publisher(
            RecorderRequest, "/recorder/request", qos_pub
        )
        self.pub_confirm = self.create_publisher(
            ConfirmRequest, "/vlm/confirm_request", qos_pub
        )

        # ---- VLM client ----
        self.vlm_cli = self.create_client(InferClip, "/vlm/infer")

        self.get_logger().info("VLM Bridge (session-locked) ready.")

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def on_kp(self, msg: KeypointsWindow) -> None:
        self.latest_kp = msg

    def on_unknown(self, msg: UnknownGesture) -> None:
        if self.session_id is not None:
            self.get_logger().warn(
                "Session already active; ignoring new unknown."
            )
            return

        # Start a new session
        self.session_id = str(uuid.uuid4())
        self.clip_for_session = None
        self.latest_reply = None

        self._set_hold(True)

        # Compute ROI from latest keypoints (or full-frame fallback)
        x0, y0, x1, y1 = self._roi_from_keypoints(self.latest_kp)

        # Choose t_center: prefer latest_kp.stamp (camera time),
        # fall back to msg.stamp if needed.
        t_center = RosTime()
        if (
            self.latest_kp is not None
            and isinstance(getattr(self.latest_kp, "stamp", None), RosTime)
        ):
            t_center = self.latest_kp.stamp
        elif isinstance(getattr(msg, "stamp", None), RosTime):
            t_center = msg.stamp

        # Build RecorderRequest
        req = RecorderRequest()
        req.session_id = self.session_id
        # window_id mostly for logging
        kp_frames = getattr(self.latest_kp, "frames", 0) or 0
        req.window_id = f"win_{kp_frames}"
        req.t_center = t_center
        req.pre_secs = float(WINDOW_PRE)
        req.post_secs = float(WINDOW_POST)
        req.x_min, req.y_min, req.x_max, req.y_max = x0, y0, x1, y1

        self.pub_req.publish(req)
        self.get_logger().info(
            f"[{self.session_id}] RecorderRequest sent "
            f"ROI=({x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}), "
            f"pre={WINDOW_PRE:.1f}s post={WINDOW_POST:.1f}s"
        )

        # Spawn watcher thread: wait for clip, then VLM+UI
        threading.Thread(
            target=self._wait_clip_then_vlm, daemon=True
        ).start()

    def on_clip_ready_msg(self, msg: ClipReady) -> None:
        if (
            self.session_id
            and msg.session_id == self.session_id
            and msg.success
        ):
            self.clip_for_session = msg.clip_path

    def on_clip_ready_json(self, js: String) -> None:
        try:
            d = json.loads(js.data)
        except Exception:
            return
        if not isinstance(d, dict):
            return
        if (
            self.session_id
            and d.get("session_id") == self.session_id
            and d.get("success")
        ):
            self.clip_for_session = d.get("clip_path")

    def on_confirm_reply(self, msg: ConfirmReply) -> None:
        # Ignore replies for old/other sessions
        if not self.session_id or msg.session_id != self.session_id:
            return
        self.latest_reply = msg

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _roi_from_keypoints(
        self, kp: Optional[KeypointsWindow]
    ) -> Tuple[float, float, float, float]:
        # Default full-frame fallback
        if kp is None or kp.frames == 0 or kp.joints_per_frame == 0:
            return 0.0, 0.0, 1.0, 1.0

        J = int(kp.joints_per_frame)
        F = int(kp.frames)
        arr = kp.data
        if len(arr) < F * J * 3:
            return 0.0, 0.0, 1.0, 1.0

        xs, ys = [], []

        # row-major: frame0_j0x, frame0_j0y, frame0_j0z, ...
        for f in range(F):
            base = f * J * 3
            for j in range(J):
                x = arr[base + j * 3 + 0]
                y = arr[base + j * 3 + 1]
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    xs.append(x)
                    ys.append(y)

        if not xs:
            return 0.0, 0.0, 1.0, 1.0

        x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
        y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))

        # pad 30%
        pad_x = (x_max - x_min) * 0.30
        pad_y = (y_max - y_min) * 0.30
        x_min = max(0.0, x_min - pad_x)
        y_min = max(0.0, y_min - pad_y)
        x_max = min(1.0, x_max + pad_x)
        y_max = min(1.0, y_max + pad_y)

        return x_min, y_min, x_max, y_max

    def _set_hold(self, val: bool) -> None:
        msg = Bool(data=val)
        self.pub_hold.publish(msg)
        self.get_logger().info(f"/pipeline/hold = {val}")

    # ------------------------------------------------------------------
    # Clip → VLM → UI path
    # ------------------------------------------------------------------

    def _wait_clip_then_vlm(self, clip_msg: RecorderClipReady):
        """
        Called in a thread when /recorder/clip_ready arrives.

        - clip_msg.clip_path is the exact video file that the VLM will/has seen.
        - We wait for the VLM result for this session_id.
        - We build a ConfirmRequest for the UI kiosk.
        """
        session_id = clip_msg.session_id
        clip_path = clip_msg.clip_path

        # 1) Wait for VLM result for this session
        try:
            result = self._vlm_results.wait_for(session_id, timeout=self.confirm_timeout_s)
            label = result["label"]
            conf = float(result["confidence"])
            rationale = result.get("rationale", "")
        except Exception as e:
            self.get_logger().error(f"[{session_id}] waiting for VLM result failed: {e}")
            # resume pipeline and bail
            self._set_hold(False)
            return

        # 2) Build ConfirmRequest for the UI kiosk
        confirm = ConfirmRequest()
        confirm.session_id = session_id
        confirm.window_id = getattr(clip_msg, "window_id", 0)
        confirm.candidate_label = label
        confirm.candidate_conf = float(conf)
        confirm.hint = rationale or ""
        confirm.source = "vlm"

        # (We do NOT set clip_path/preview_frame_b64; kiosk finds the clip via session_id)

        # 3) Publish to kiosk
        self.pub_confirm.publish(confirm)
        self.get_logger().info(
            f"[{session_id}] ConfirmRequest → label={label} conf={conf:.2f}, clip={clip_path}"
        )

    def _finalize_session(self, resume: bool) -> None:
        if resume:
            self._set_hold(False)
        self.get_logger().info(
            f"[{self.session_id}] session end; resume={resume}"
        )
        self.session_id = None
        self.clip_for_session = None
        self.latest_reply = None


def main() -> None:
    rclpy.init()
    node = VlmBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

