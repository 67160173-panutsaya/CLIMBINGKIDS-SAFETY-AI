import json
import logging
import os
import queue
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
from ultralytics import YOLO


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

VIDEO_PATH = Path(__file__).resolve().with_name("climbingvedio.mp4")
OUTPUT_PATH = Path(__file__).resolve().with_name("climbingvedio_output.mp4")
MODEL_PATH = Path(__file__).resolve().with_name("yolov8n.pt")
POSE_MODEL_PATH = Path(__file__).resolve().with_name("yolov8n-pose.pt")
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
HOST = "127.0.0.1"
PORT = 8080
ENV_FILE_PATH = Path(__file__).resolve().with_name(".env")


def load_env_file():
    if not ENV_FILE_PATH.exists():
        return

    try:
        lines = ENV_FILE_PATH.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logging.error("อ่านไฟล์ .env ไม่สำเร็จ: %s", exc)
        return

    for line_number, line in enumerate(lines, start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("export "):
            entry = entry[7:].strip()
        if "=" not in entry:
            logging.warning("รูปแบบในไฟล์ .env ไม่ถูกต้องที่บรรทัด %s", line_number)
            continue

        name, value = (part.strip() for part in entry.split("=", 1))
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            logging.warning("ชื่อตัวแปรในไฟล์ .env ไม่ถูกต้องที่บรรทัด %s", line_number)
            continue
        if os.environ.get(name):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[name] = value


load_env_file()

PAGE = r"""<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ระบบตรวจจับการปีน</title>
  <style>
    :root { color-scheme: dark; font-family: "Segoe UI", system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { max-width: 1080px; margin: 32px auto; padding: 0 18px; background: #0b1220; color: #f3f4f6; }
    h1 { margin-bottom: 8px; letter-spacing: -.03em; }
    .panel { padding: 20px; margin: 18px 0; background: #151f31; border: 1px solid #26344a; border-radius: 16px; }
    button { padding: 11px 16px; border: 0; border-radius: 9px; background: #2563eb; color: white; font-size: .95rem; font-weight: 650; cursor: pointer; transition: filter .15s, transform .15s; }
    button:hover:not(:disabled) { filter: brightness(1.12); transform: translateY(-1px); }
    button:disabled { cursor: not-allowed; opacity: .55; }
    #stop { background: #b91c1c; }
    .stream-frame { position: relative; min-height: 200px; overflow: hidden; background: #030712; border-radius: 11px; }
    #stream { display: block; width: 100%; min-height: 200px; background: #030712; object-fit: contain; border-radius: 11px; }
    #browserCamera { display: block; width: 100%; max-height: 68vh; min-height: 200px; background: #030712; object-fit: contain; border-radius: 11px; }
    #browserCamera[hidden], #stream[hidden] { display: none; }
    #streamPlaceholder { position: absolute; inset: 0; display: grid; place-content: center; gap: 10px; padding: 24px; color: #aab8cc; text-align: center; pointer-events: none; }
    #streamPlaceholder[hidden] { display: none; }
    .muted { color: #aab8cc; }
    #status, #lineStatus { font-weight: 650; }
    dialog { width: min(560px, calc(100% - 28px)); max-height: min(90vh, 760px); overflow-y: auto; padding: 0; border: 1px solid #34445d; border-radius: 20px; background: #111b2c; color: #f3f4f6; box-shadow: 0 24px 80px #0009; }
    dialog::backdrop { background: rgb(2 6 23 / 78%); backdrop-filter: blur(5px); }
    .dialog-content { padding: 26px; }
    .eyebrow { margin: 0 0 8px; color: #93c5fd; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .dialog-title { margin: 0; font-size: 1.5rem; }
    .dialog-intro { margin: 8px 0 18px; color: #aab8cc; line-height: 1.55; }
    .choice { display: flex; gap: 12px; align-items: flex-start; margin: 10px 0; padding: 15px; border: 1px solid #34445d; border-radius: 13px; background: #172337; cursor: pointer; }
    .choice:has(input:checked) { border-color: #60a5fa; background: #1b2c45; box-shadow: inset 0 0 0 1px #60a5fa55; }
    .choice input { margin-top: 4px; accent-color: #60a5fa; }
    .choice-copy { display: grid; gap: 4px; }
    .choice-copy small { color: #aab8cc; line-height: 1.4; }
    .source-icon { display: grid; place-items: center; flex: 0 0 38px; height: 38px; border-radius: 10px; background: #22334c; color: #bfdbfe; font-size: 1.15rem; }
    #filePicker { width: 100%; margin-top: 12px; padding: 12px; border: 1px dashed #526680; border-radius: 10px; background: #0b1423; color: #dbeafe; }
    #fileName { overflow-wrap: anywhere; font-size: .9rem; }
    #uploadProgress { width: 100%; height: 8px; accent-color: #60a5fa; }
    #pickerError { min-height: 1.2em; color: #fca5a5; }
    .notice { margin: 16px 0; padding: 12px; border: 1px solid #695521; border-radius: 10px; background: #352b14; color: #f5d98c; font-size: .84rem; line-height: 1.5; }
    .dialog-actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
    #cancelPicker { background: #334155; }
    @media (max-width: 560px) { body { margin-top: 18px; } .dialog-content { padding: 20px; } }
  </style>
</head>
<body>
  <h1>ระบบตรวจจับการปีน/พื้นที่อันตราย</h1>
  <p class="muted">เลือกกล้องจริงหรือไฟล์วิดีโอจากเครื่อง แล้วกดเริ่มตรวจสอบ</p>
  <section class="panel">
    <button id="openPicker">เลือกกล้องหรือไฟล์วิดีโอ</button>
    <button id="stop" disabled>หยุดอัด/ตรวจสอบ</button>
    <span id="selectedSource" class="muted">ยังไม่ได้เลือกแหล่งวิดีโอ</span>
    <p>สถานะ: <span id="status">กำลังเชื่อมต่อ...</span></p>
    <p>LINE: <span id="lineStatus">กำลังตรวจสอบ...</span></p>
    <button id="testLine" type="button">ส่งข้อความทดสอบ LINE</button>
    <p id="output" class="muted"></p>
    <p class="muted">LINE จะส่งแจ้งเตือนอัตโนมัติเมื่อตรวจพบอันตราย ตั้งค่า LINE_CHANNEL_ACCESS_TOKEN และ LINE_TARGET_ID ในไฟล์ .env ข้างโปรแกรม</p>
  </section>
  <section class="panel">
    <div class="stream-frame">
      <img id="stream" src="/stream" alt="ภาพสดจากกล้องหรือวิดีโอที่กำลังตรวจสอบ">
      <video id="browserCamera" autoplay muted playsinline hidden></video>
      <div id="streamPlaceholder"><strong>ยังไม่มีภาพวิดีโอ</strong><span>เลือกกล้องหรือไฟล์ แล้วกดเริ่มตรวจสอบ</span></div>
    </div>
  </section>
  <dialog id="sourceDialog">
    <form id="sourceForm" class="dialog-content">
      <p class="eyebrow">CCTV · AI Monitoring</p>
      <h2 class="dialog-title">เริ่มตรวจสอบวิดีโอ</h2>
      <p class="dialog-intro">เลือกแหล่งภาพ ระบบจะวิเคราะห์และบันทึกวิดีโอที่มีกรอบตรวจจับ</p>
      <label class="choice">
        <input type="radio" name="source" value="camera" checked>
        <span class="source-icon" aria-hidden="true">◉</span>
        <span class="choice-copy"><strong>กล้องจริง</strong><small>ใช้กล้องหลักที่เชื่อมต่อกับเครื่องนี้</small></span>
      </label>
      <label class="choice">
        <input type="radio" name="source" value="file">
        <span class="source-icon" aria-hidden="true">▣</span>
        <span class="choice-copy"><strong>ไฟล์วิดีโอ</strong><small>เลือกไฟล์จากเครื่องเพื่อวิเคราะห์ย้อนหลัง</small></span>
      </label>
      <input id="filePicker" type="file" accept="video/*,.mp4,.avi,.mov,.mkv,.webm" hidden>
      <p id="fileName" class="muted">รองรับ MP4, AVI, MOV, MKV, M4V และ WEBM · สูงสุด 2 GB</p>
      <progress id="uploadProgress" max="100" value="0" hidden></progress>
      <p id="uploadStatus" class="muted" aria-live="polite"></p>
      <p class="notice">โหมดสมดุล: เลือกความละเอียดประมวลผลตามชนิดแหล่งภาพเพื่อให้เหมาะกับเครื่องโน้ตบุ๊ก ความแม่นยำยังขึ้นกับแสง มุมกล้อง และการบังภาพ ระบบ AI ไม่ทดแทนการเฝ้าระวังโดยคน</p>
      <p id="pickerError" role="alert"></p>
      <div class="dialog-actions">
        <button id="begin" type="submit">เริ่มตรวจสอบ</button>
        <button id="cancelPicker" type="button">ยกเลิก</button>
      </div>
    </form>
  </dialog>
  <script>
    const sourceDialog = document.getElementById("sourceDialog");
    const sourceForm = document.getElementById("sourceForm");
    const openPicker = document.getElementById("openPicker");
    const filePicker = document.getElementById("filePicker");
    const fileName = document.getElementById("fileName");
    const uploadProgress = document.getElementById("uploadProgress");
    const uploadStatus = document.getElementById("uploadStatus");
    const pickerError = document.getElementById("pickerError");
    const begin = document.getElementById("begin");
    const selectedSource = document.getElementById("selectedSource");
    const stop = document.getElementById("stop");
    const testLine = document.getElementById("testLine");
    const status = document.getElementById("status");
    const lineStatus = document.getElementById("lineStatus");
    const output = document.getElementById("output");
    const stream = document.getElementById("stream");
    const streamPlaceholder = document.getElementById("streamPlaceholder");
    const browserCamera = document.getElementById("browserCamera");
    const cameraCanvas = document.createElement("canvas");
    let cameraMediaStream = null;
    let cameraLoopRunning = false;

    stream.addEventListener("load", () => {
      streamPlaceholder.hidden = true;
    });
    function stopBrowserCamera() {
      cameraLoopRunning = false;
      if (cameraMediaStream) {
        cameraMediaStream.getTracks().forEach((track) => track.stop());
        cameraMediaStream = null;
      }
      browserCamera.pause();
      browserCamera.srcObject = null;
      browserCamera.hidden = true;
    }

    const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

    function cameraErrorMessage(error) {
      if (error.name === "NotAllowedError" || error.name === "SecurityError") {
        return "ไม่ได้รับอนุญาตให้ใช้กล้อง กรุณากดอนุญาตกล้องในเบราว์เซอร์แล้วลองใหม่";
      }
      if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
        return "ไม่พบกล้อง กรุณาตรวจสอบว่ามีกล้องเชื่อมต่ออยู่";
      }
      if (error.name === "NotReadableError" || error.name === "TrackStartError") {
        return "กล้องถูกใช้งานอยู่หรือเปิดไม่ได้ กรุณาปิดแอปอื่นที่ใช้กล้องแล้วลองใหม่";
      }
      return error.message || "เปิดกล้องไม่สำเร็จ";
    }

    async function sendCameraFrames() {
      while (cameraLoopRunning) {
        if (browserCamera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
          await wait(100);
          continue;
        }
        const width = browserCamera.videoWidth;
        const height = browserCamera.videoHeight;
        const scale = Math.min(1, 640 / Math.max(width, height));
        cameraCanvas.width = Math.max(1, Math.round(width * scale));
        cameraCanvas.height = Math.max(1, Math.round(height * scale));
        cameraCanvas.getContext("2d").drawImage(
          browserCamera, 0, 0, cameraCanvas.width, cameraCanvas.height
        );
        const frame = await new Promise((resolve) =>
          cameraCanvas.toBlob(resolve, "image/jpeg", 0.75)
        );
        if (!frame || !cameraLoopRunning) continue;
        try {
          const response = await fetch("/api/camera-frame", {
            method: "POST",
            headers: { "Content-Type": "image/jpeg" },
            body: frame
          });
          const data = await response.json();
          if (!response.ok) throw new Error(data.error || "วิเคราะห์ภาพจากกล้องไม่สำเร็จ");
          status.textContent = data.status;
        } catch (error) {
          if (cameraLoopRunning) {
            status.textContent = cameraErrorMessage(error);
            pickerError.textContent = cameraErrorMessage(error);
            stopBrowserCamera();
            await fetch("/api/stop", { method: "POST" }).catch(() => {});
            await refreshStatus();
          }
          return;
        }
      }
    }

    async function refreshStatus() {
      try {
        const response = await fetch("/api/status");
        const data = await response.json();
        status.textContent = data.status;
        lineStatus.textContent = data.alert_status;
        output.textContent = data.output_path ? `วิดีโอผลลัพธ์: ${data.output_path}` : "";
        openPicker.disabled = data.running;
        stop.disabled = !data.running;
      } catch (error) {
        status.textContent = "เชื่อมต่อกับเซิร์ฟเวอร์ไม่ได้";
      }
    }

    testLine.addEventListener("click", async () => {
      testLine.disabled = true;
      lineStatus.textContent = "กำลังส่งข้อความทดสอบ...";
      try {
        const response = await fetch("/api/test-line", { method: "POST" });
        const data = await response.json();
        lineStatus.textContent = data.alert_status || data.error || "ไม่ได้รับสถานะจากเซิร์ฟเวอร์";
      } catch (error) {
        lineStatus.textContent = `ทดสอบ LINE ไม่สำเร็จ: ${error.message}`;
      } finally {
        testLine.disabled = false;
      }
    });

    openPicker.addEventListener("click", () => {
      pickerError.textContent = "";
      sourceDialog.showModal();
    });

    document.querySelectorAll('input[name="source"]').forEach((radio) => {
      radio.addEventListener("change", () => {
        filePicker.hidden = sourceForm.querySelector('input[name="source"]:checked').value !== "file";
        pickerError.textContent = "";
      });
    });

    filePicker.addEventListener("change", () => {
      const file = filePicker.files[0];
      fileName.textContent = file
        ? `เลือกแล้ว: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`
        : "รองรับ MP4, AVI, MOV, MKV, M4V และ WEBM · สูงสุด 2 GB";
      pickerError.textContent = "";
    });

    document.getElementById("cancelPicker").addEventListener("click", () => {
      sourceDialog.close();
    });

    sourceForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const source = sourceForm.querySelector('input[name="source"]:checked').value;
      const file = filePicker.files[0];
      if (source === "file" && !file) {
        pickerError.textContent = "กรุณาเลือกไฟล์วิดีโอก่อนเริ่มตรวจสอบ";
        return;
      }
      if (source === "file" && file.size > 2 * 1024 * 1024 * 1024) {
        pickerError.textContent = "ไฟล์ใหญ่เกินไป (สูงสุด 2 GB)";
        return;
      }
      begin.disabled = true;
      pickerError.textContent = "";
      uploadProgress.hidden = source !== "file";
      uploadProgress.value = 0;
      uploadStatus.textContent = "";
      try {
        if (source === "camera") {
          if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error(
              "เบราว์เซอร์ไม่รองรับกล้อง ให้เปิดผ่าน http://127.0.0.1:8080 และอนุญาตสิทธิ์กล้อง"
            );
          }
          cameraMediaStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
              facingMode: "user",
              width: { ideal: 640 },
              height: { ideal: 480 },
              frameRate: { ideal: 15, max: 30 }
            }
          });
          cameraMediaStream.getVideoTracks().forEach((track) => {
            track.addEventListener("ended", () => {
              if (!cameraLoopRunning) return;
              pickerError.textContent = "การเชื่อมต่อกล้องถูกตัด";
              stopBrowserCamera();
              void fetch("/api/stop", { method: "POST" }).then(refreshStatus);
            }, { once: true });
          });
          browserCamera.srcObject = cameraMediaStream;
          browserCamera.hidden = true;
          stream.hidden = false;
          await browserCamera.play();
        } else {
          stopBrowserCamera();
          stream.hidden = false;
        }
        if (source === "file") {
          await new Promise((resolve, reject) => {
            const request = new XMLHttpRequest();
            request.open("POST", "/api/upload");
            request.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
            request.upload.addEventListener("progress", (progressEvent) => {
              if (!progressEvent.lengthComputable) return;
              const percent = Math.round(progressEvent.loaded / progressEvent.total * 100);
              uploadProgress.value = percent;
              uploadStatus.textContent = `กำลังส่งไฟล์ ${percent}%`;
            });
            request.addEventListener("load", () => {
              let result;
              try {
                result = JSON.parse(request.responseText);
              } catch {
                reject(new Error("เซิร์ฟเวอร์ตอบกลับข้อมูลไม่ถูกต้อง"));
                return;
              }
              if (request.status < 200 || request.status >= 300) {
                reject(new Error(result.error || `อัปโหลดไม่สำเร็จ (HTTP ${request.status})`));
                return;
              }
              resolve(result);
            });
            request.addEventListener("error", () => reject(new Error("การเชื่อมต่อขาดระหว่างอัปโหลดไฟล์")));
            request.addEventListener("abort", () => reject(new Error("ยกเลิกการอัปโหลดไฟล์แล้ว")));
            request.send(file);
          });
        }
        const backendSource = source === "camera" ? "browser_camera" : source;
        const response = await fetch("/api/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: backendSource })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        selectedSource.textContent = source === "camera"
          ? "แหล่งที่เลือก: กล้องจากเบราว์เซอร์ · ภาพสด"
          : `แหล่งที่เลือก: ${file.name}`;
        sourceDialog.close();
        await refreshStatus();
        if (source === "camera") {
          cameraLoopRunning = true;
          void sendCameraFrames();
        }
      } catch (error) {
        pickerError.textContent = source === "camera"
          ? cameraErrorMessage(error)
          : error.message || "เริ่มตรวจสอบไม่สำเร็จ";
        if (source === "camera") {
          stopBrowserCamera();
          stream.hidden = false;
        }
      } finally {
        begin.disabled = false;
        uploadStatus.textContent = "";
        uploadProgress.hidden = true;
      }
    });

    stop.addEventListener("click", async () => {
      stop.disabled = true;
      stopBrowserCamera();
      try {
        const response = await fetch("/api/stop", { method: "POST" });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        stream.hidden = false;
        await refreshStatus();
      } catch (error) {
        status.textContent = error.message;
      }
    });

    refreshStatus();
    sourceDialog.showModal();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""


def find_ankles(person_box, pose_boxes, pose_keypoints, matched_pose_indices):
    person_area = max(1, (person_box[2] - person_box[0]) * (person_box[3] - person_box[1]))
    best_match = None
    best_iou = 0.15

    for pose_index, pose_box in enumerate(pose_boxes):
        if pose_index in matched_pose_indices:
            continue
        intersection = (
            max(0, min(person_box[2], pose_box[2]) - max(person_box[0], pose_box[0]))
            * max(0, min(person_box[3], pose_box[3]) - max(person_box[1], pose_box[1]))
        )
        pose_area = max(1, (pose_box[2] - pose_box[0]) * (pose_box[3] - pose_box[1]))
        union = person_area + pose_area - intersection
        iou = intersection / union if union else 0
        if iou > best_iou:
            best_iou = iou
            best_match = pose_index

    if best_match is None:
        return [], None

    matched_pose_indices.add(best_match)
    keypoints = pose_keypoints[best_match]
    ankles = keypoints[[15, 16]]
    visible_ankles = ankles[ankles[:, 2] >= 0.2]
    points = [(int(point[0]), int(point[1])) for point in visible_ankles]
    if not points:
        return [], None
    foot_point = (
        int(sum(point[0] for point in points) / len(points)),
        int(sum(point[1] for point in points) / len(points)),
    )
    return points, foot_point


class VideoMonitor:
    def __init__(self):
        self.model = YOLO(str(MODEL_PATH))
        self.pose_model = YOLO(str(POSE_MODEL_PATH))
        self.lock = threading.RLock()
        self.camera_inference_lock = threading.Lock()
        self.stop_event = None
        self.worker = None
        self.running = False
        self.server_running = True
        self.latest_jpeg = None
        self.status = "พร้อมเริ่มตรวจสอบ"
        self.frame_count = 0
        self.output_path = ""
        self.alert_status = self._line_configuration_status()
        self.pending_video_path = None
        self.pending_video_name = None
        self.browser_camera = False
        self.camera_track_states = {}
        self.camera_frame_index = 0
        self.camera_writer = None

    @staticmethod
    def _line_configuration_status():
        missing = [
            name
            for name in ("LINE_CHANNEL_ACCESS_TOKEN", "LINE_TARGET_ID")
            if not os.getenv(name)
        ]
        if missing:
            return f"ยังไม่ได้ตั้งค่า LINE: {', '.join(missing)}"
        return "พร้อมส่งข้อความแจ้งเตือน"

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "status": self.status,
                "frames": self.frame_count,
                "output_path": self.output_path,
                "alert_status": self.alert_status,
            }

    def upload_video(self, filename, stream, content_length):
        safe_name = Path(filename.replace("\\", "/")).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise ValueError(
                "ชนิดไฟล์ไม่รองรับ กรุณาเลือกไฟล์ AVI, M4V, MKV, MOV, MP4 หรือ WEBM"
            )
        if content_length <= 0:
            raise ValueError("ไฟล์วิดีโอว่างเปล่าหรือไม่ได้รับข้อมูล")
        if content_length > MAX_UPLOAD_BYTES:
            raise ValueError("ไฟล์ใหญ่เกินไป (สูงสุด 2 GB)")
        with self.lock:
            if self.running:
                raise ValueError("หยุดตรวจสอบก่อนจึงจะเปลี่ยนไฟล์ได้")

        temp_path = None
        previous_path = None
        received = 0
        try:
            with tempfile.NamedTemporaryFile(
                prefix="cctv_upload_", suffix=suffix, delete=False
            ) as uploaded:
                temp_path = Path(uploaded.name)
                while received < content_length:
                    chunk = stream.read(min(1024 * 1024, content_length - received))
                    if not chunk:
                        raise ValueError("การอัปโหลดไฟล์ไม่สมบูรณ์")
                    uploaded.write(chunk)
                    received += len(chunk)

            with self.lock:
                if self.running:
                    raise ValueError("หยุดตรวจสอบก่อนจึงจะเปลี่ยนไฟล์ได้")
                previous_path = self.pending_video_path
                self.pending_video_path = temp_path
                self.pending_video_name = safe_name
                temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        if previous_path is not None:
            previous_path.unlink(missing_ok=True)
        return safe_name

    def start(self, source):
        with self.lock:
            if self.running:
                raise ValueError("กำลังตรวจสอบวิดีโออยู่แล้ว")

        if source == "browser_camera":
            with self.lock:
                self.stop_event = threading.Event()
                self.running = True
                self.browser_camera = True
                self.camera_track_states = {}
                self.camera_frame_index = 0
                self.camera_writer = None
                self.status = "กำลังรับภาพสดจากเบราว์เซอร์..."
                self.frame_count = 0
                self.output_path = str(OUTPUT_PATH)
                self.latest_jpeg = None
            return

        cleanup_path = None
        if source == "camera":
            with self.lock:
                cleanup_path = self.pending_video_path
                self.pending_video_path = None
                self.pending_video_name = None
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)
                cleanup_path = None
            capture, backend_name = self._open_camera()
            source_description = f"กล้องโน้ตบุ๊ก ({backend_name})"
        elif source == "file":
            with self.lock:
                video_path = self.pending_video_path or VIDEO_PATH
                video_name = self.pending_video_name or VIDEO_PATH.name
                if self.pending_video_path is not None:
                    cleanup_path = self.pending_video_path
                    self.pending_video_path = None
                    self.pending_video_name = None
            if not video_path.is_file():
                raise ValueError(f"ไม่พบไฟล์วิดีโอ: {video_path}")
            capture = cv2.VideoCapture(str(video_path))
            source_description = f"ไฟล์ {video_name}"
        else:
            raise ValueError("แหล่งวิดีโอไม่ถูกต้อง")

        if not capture.isOpened():
            capture.release()
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)
            raise ValueError(f"ไม่สามารถเปิดแหล่งวิดีโอ: {source_description}")

        with self.lock:
            if self.running:
                capture.release()
                if cleanup_path is not None:
                    cleanup_path.unlink(missing_ok=True)
                raise ValueError("กำลังตรวจสอบวิดีโออยู่แล้ว")
            self.stop_event = threading.Event()
            self.running = True
            self.status = f"กำลังตรวจสอบ: {source_description}"
            self.frame_count = 0
            self.output_path = str(OUTPUT_PATH)
            self.latest_jpeg = None
            self.worker = threading.Thread(
                target=self._process_video,
                args=(capture, self.stop_event, source_description, cleanup_path),
                daemon=True,
            )
            self.worker.start()

    @staticmethod
    def _open_camera():
        backends = (
            ("MSMF", cv2.CAP_MSMF),
            ("DirectShow", cv2.CAP_DSHOW),
            ("Default", cv2.CAP_ANY),
        )
        failures = []
        for backend_name, backend in backends:
            capture = cv2.VideoCapture(0, backend)
            if not capture.isOpened():
                failures.append(backend_name)
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS, 15)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            success, _ = capture.read()
            if success:
                logging.info(
                    "เปิดกล้องด้วย %s ที่ความละเอียด %.0fx%.0f",
                    backend_name,
                    capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                    capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                )
                return capture, backend_name

            failures.append(f"{backend_name} (อ่านภาพไม่ได้)")
            capture.release()

        raise ValueError(
            "เปิดกล้องโน้ตบุ๊กไม่สำเร็จด้วย backend: " + ", ".join(failures)
        )

    def stop(self):
        browser_camera = False
        with self.lock:
            if not self.running or self.stop_event is None:
                return
            self.status = "กำลังหยุดตรวจสอบ..."
            self.stop_event.set()
            browser_camera = self.browser_camera
            if browser_camera:
                self.running = False
                self.browser_camera = False
        if browser_camera:
            with self.camera_inference_lock:
                if self.camera_writer is not None:
                    self.camera_writer.release()
                    self.camera_writer = None
            with self.lock:
                self.status = "หยุดตรวจสอบแล้ว"

    def process_browser_camera_frame(self, jpeg_bytes):
        if not jpeg_bytes or len(jpeg_bytes) > 3 * 1024 * 1024:
            raise ValueError("ข้อมูลภาพว่างหรือใหญ่เกินไป")
        with self.camera_inference_lock:
            with self.lock:
                if not self.running or not self.browser_camera:
                    raise ValueError("ยังไม่ได้เริ่มตรวจสอบจากกล้อง")
                frame_index = self.camera_frame_index
                self.camera_frame_index += 2

            frame_array = cv2.imdecode(
                np.frombuffer(jpeg_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame_array is None:
                raise ValueError("อ่านภาพจากเบราว์เซอร์ไม่สำเร็จ")

            annotated, alerts = self._detect_frame(
                frame_array,
                self.camera_track_states,
                frame_index,
                4.0,
                4,
                2,
                416,
            )
            if self.camera_writer is None:
                height, width = annotated.shape[:2]
                self.camera_writer = cv2.VideoWriter(
                    str(OUTPUT_PATH),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    2.0,
                    (width, height),
                )
                if not self.camera_writer.isOpened():
                    self.camera_writer.release()
                    self.camera_writer = None
                    raise RuntimeError(f"ไม่สามารถสร้างไฟล์ผลลัพธ์: {OUTPUT_PATH}")
            self.camera_writer.write(annotated)
            encoded, output_jpeg = cv2.imencode(
                ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if not encoded:
                raise RuntimeError("แปลงภาพผลตรวจจับไม่สำเร็จ")
            with self.lock:
                self.latest_jpeg = output_jpeg.tobytes()
                self.frame_count += 1
                self.status = "กำลังตรวจสอบกล้องสด"

            for alert in alerts:
                threading.Thread(
                    target=self._send_line_alert, args=(alert,), daemon=True
                ).start()
            return self.get_status()

    def _camera_reader_loop(self, capture, stop_event, frame_queue):
        current_capture = capture
        captured_frames = 0
        try:
            while not stop_event.is_set():
                success, frame = current_capture.read()
                if not success:
                    current_capture.release()
                    with self.lock:
                        self.status = "กล้องหลุด กำลังพยายามเชื่อมต่อใหม่..."
                    if stop_event.wait(1):
                        break
                    try:
                        current_capture, backend_name = self._open_camera()
                    except ValueError as exc:
                        logging.warning("%s", exc)
                        continue
                    with self.lock:
                        self.status = f"เชื่อมต่อกล้องอีกครั้งแล้ว: {backend_name}"
                    continue

                encoded, jpeg = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                )
                if not encoded:
                    raise RuntimeError("ไม่สามารถแปลงภาพสดจากกล้องได้")
                captured_frames += 1
                with self.lock:
                    self.latest_jpeg = jpeg.tobytes()
                    self.frame_count = captured_frames

                try:
                    frame_queue.put_nowait((captured_frames, frame))
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    frame_queue.put_nowait((captured_frames, frame))
        except Exception:
            logging.exception("อ่านภาพจากกล้องไม่สำเร็จ")
            with self.lock:
                self.status = "เกิดข้อผิดพลาดขณะอ่านภาพจากกล้อง"
            stop_event.set()
        finally:
            current_capture.release()

    def _process_video(self, capture, stop_event, source_description, cleanup_path=None):
        writer = None
        error = None
        track_states = {}
        is_camera = source_description.startswith("กล้องโน้ตบุ๊ก")
        fps = 15.0 if is_camera else capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 20.0
        confirm_frames = max(1, round(fps * 0.25))
        clear_frames = max(1, round(fps * 0.5))
        inference_size = 416 if is_camera else 1280
        camera_queue = queue.Queue(maxsize=1)
        reader_thread = None
        last_processed_frame = 0

        try:
            if is_camera:
                reader_thread = threading.Thread(
                    target=self._camera_reader_loop,
                    args=(capture, stop_event, camera_queue),
                    daemon=True,
                )
                reader_thread.start()

            while not stop_event.is_set():
                if is_camera:
                    try:
                        frame_index, frame = camera_queue.get(timeout=1)
                    except queue.Empty:
                        if stop_event.is_set():
                            break
                        continue
                    if frame_index <= last_processed_frame:
                        continue
                    elapsed_frames = frame_index - last_processed_frame
                    last_processed_frame = frame_index
                else:
                    success, frame = capture.read()
                    if not success:
                        break
                    frame_index = last_processed_frame + 1
                    last_processed_frame = frame_index
                    elapsed_frames = 1

                annotated, alerts = self._detect_frame(
                    frame,
                    track_states,
                    frame_index,
                    fps,
                    confirm_frames,
                    clear_frames,
                    inference_size,
                )
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(OUTPUT_PATH),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"ไม่สามารถสร้างไฟล์ผลลัพธ์: {OUTPUT_PATH}")

                write_count = min(max(1, elapsed_frames), int(fps * 2)) if is_camera else 1
                for _ in range(write_count):
                    writer.write(annotated)
                if not is_camera:
                    encoded, jpeg = cv2.imencode(
                        ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                    )
                    if not encoded:
                        raise RuntimeError("ไม่สามารถแปลงเฟรมเป็นภาพสำหรับหน้าเว็บได้")
                    with self.lock:
                        self.latest_jpeg = jpeg.tobytes()
                        self.frame_count = frame_index
                for alert in alerts:
                    threading.Thread(
                        target=self._send_line_alert, args=(alert,), daemon=True
                    ).start()
        except Exception as exc:
            error = str(exc)
            logging.exception("เกิดข้อผิดพลาดระหว่างตรวจสอบวิดีโอ")
        finally:
            if is_camera:
                stop_event.set()
                if reader_thread is not None:
                    reader_thread.join(timeout=2)
            else:
                capture.release()
            if writer is not None:
                writer.release()
            if cleanup_path is not None:
                try:
                    cleanup_path.unlink(missing_ok=True)
                except OSError:
                    logging.exception("ลบไฟล์อัปโหลดชั่วคราวไม่สำเร็จ: %s", cleanup_path)
            with self.lock:
                self.running = False
                if error:
                    self.status = f"เกิดข้อผิดพลาด: {error}"
                elif stop_event.is_set() and self.status.startswith("เกิดข้อผิดพลาด"):
                    pass
                elif stop_event.is_set():
                    self.status = "หยุดตรวจสอบแล้ว"
                else:
                    self.status = f"ประมวลผลเสร็จแล้ว: {source_description}"

    def _detect_frame(
        self, frame, track_states, frame_index, fps, confirm_frames, clear_frames,
        inference_size,
    ):
        height = frame.shape[0]
        results = self.model.track(
            frame, conf=0.2, imgsz=inference_size, persist=True,
            tracker="bytetrack.yaml", verbose=False
        )[0]
        pose_result = self.pose_model.track(
            frame, conf=0.1, imgsz=inference_size, persist=True,
            tracker="bytetrack.yaml", verbose=False
        )[0]
        pose_boxes = pose_result.boxes.xyxy.cpu().numpy().astype(int)
        pose_keypoints = pose_result.keypoints.data.cpu().numpy()
        pose_track_ids = (
            pose_result.boxes.id.int().cpu().tolist()
            if pose_result.boxes.id is not None
            else [None] * len(pose_boxes)
        )
        matched_pose_indices = set()
        persons = []
        objects = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            coords_array = box.xyxy[0].cpu().numpy().astype(int).reshape(-1)
            if coords_array.size != 4:
                raise RuntimeError(
                    "YOLO ส่งกรอบวัตถุไม่ถูกต้อง "
                    f"(ต้องมี 4 ค่า x1,y1,x2,y2 แต่ได้ {coords_array.size})"
                )
            coords = tuple(int(value) for value in coords_array)
            if class_name == "person":
                track_id = int(box.id[0]) if box.id is not None else None
                ankle_points, foot_point = find_ankles(
                    coords, pose_boxes, pose_keypoints, matched_pose_indices
                )
                persons.append((coords, track_id, ankle_points, foot_point))
            else:
                objects.append(coords)

        for pose_index, pose_box in enumerate(pose_boxes):
            if pose_index in matched_pose_indices:
                continue
            px1, py1, px2, py2 = pose_box
            pose_area = max(1, (px2 - px1) * (py2 - py1))
            already_detected = False
            for person_box, _, _, _ in persons:
                intersection = (
                    max(0, min(person_box[2], px2) - max(person_box[0], px1))
                    * max(0, min(person_box[3], py2) - max(person_box[1], py1))
                )
                person_area = max(
                    1, (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])
                )
                union = person_area + pose_area - intersection
                if union and intersection / union >= 0.1:
                    already_detected = True
                    break
            if already_detected:
                continue

            pose_track_id = pose_track_ids[pose_index]
            track_id = f"pose-{pose_track_id}" if pose_track_id is not None else None
            ankles = pose_keypoints[pose_index][[15, 16]]
            visible_ankles = ankles[ankles[:, 2] >= 0.2]
            ankle_points = [
                (int(point[0]), int(point[1])) for point in visible_ankles
            ]
            foot_point = (
                (
                    int(sum(point[0] for point in ankle_points) / len(ankle_points)),
                    int(sum(point[1] for point in ankle_points) / len(ankle_points)),
                )
                if ankle_points
                else None
            )
            persons.append(
                (tuple(int(value) for value in pose_box), track_id, ankle_points, foot_point)
            )

        visible_track_ids = set()
        alerts = []
        for person_index, (person_box, track_id, ankle_points, foot_point) in enumerate(persons):
            px1, py1, px2, py2 = person_box
            person_height = max(1, py2 - py1)
            foot_y = foot_point[1] if foot_point is not None else py2
            if track_id is None:
                track_id = f"person-{person_index}"
            visible_track_ids.add(track_id)

            state = track_states.setdefault(
                track_id,
                {
                    "ground_y": float(foot_y),
                    "ground_height": float(person_height),
                    "climbing_streak": 0,
                    "safe_streak": 0,
                    "is_climbing": False,
                    "last_box": person_box,
                    "last_seen": frame_index,
                    "last_eval_frame": frame_index,
                },
            )
            elapsed_frames = max(1, frame_index - state.get("last_eval_frame", frame_index - 1))
            state["last_eval_frame"] = frame_index
            state["last_seen"] = frame_index
            state["last_box"] = person_box
            was_climbing = state["is_climbing"]
            feet_off_ground = (
                state["ground_y"] - foot_y > max(5, state["ground_height"] * 0.05)
                and person_height >= state["ground_height"] * 0.8
            )
            supporting_object = None
            if ankle_points:
                for detected_object in objects:
                    object_height = max(1, detected_object[3] - detected_object[1])
                    feet_over_object = any(
                        detected_object[0] <= ankle_x <= detected_object[2]
                        and detected_object[1] - 10 <= ankle_y
                        <= detected_object[1] + object_height * 0.45
                        for ankle_x, ankle_y in ankle_points
                    )
                    if feet_over_object:
                        supporting_object = detected_object
                        break
            on_elevated_object = supporting_object is not None
            climbing_candidate = feet_off_ground or on_elevated_object

            if not climbing_candidate and foot_y >= state["ground_y"]:
                state["ground_y"] = float(foot_y)
                state["ground_height"] = float(person_height)

            if climbing_candidate:
                state["climbing_streak"] += elapsed_frames
                state["safe_streak"] = 0
                if state["climbing_streak"] >= confirm_frames:
                    state["is_climbing"] = True
            else:
                state["climbing_streak"] = 0
                if state["is_climbing"]:
                    state["safe_streak"] += elapsed_frames
                    if state["safe_streak"] >= clear_frames:
                        state["is_climbing"] = False
                        state["safe_streak"] = 0

            if state["is_climbing"] and not was_climbing:
                center_x = int((px1 + px2) / 2)
                center_y = int((py1 + py2) / 2)
                reason = "FEET OFF GROUND" if feet_off_ground else "ON ELEVATED OBJECT"
                alerts.append(
                    {
                        "track_id": track_id,
                        "x": center_x,
                        "y": center_y,
                        "reason": reason,
                    }
                )

            if state["is_climbing"]:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                cv2.putText(frame, "WARNING: CLIMBING!", (px1, max(py1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                reason = "FEET OFF GROUND" if feet_off_ground else "ON ELEVATED OBJECT"
                cv2.putText(frame, reason, (px1, min(py2 + 22, height - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif climbing_candidate:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    f"Confirming climb: {state['climbing_streak']}/{confirm_frames} frames",
                    (px1, max(py1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                )
            else:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (px1, max(py1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for ankle_x, ankle_y in ankle_points:
                cv2.circle(
                    frame,
                    (ankle_x, ankle_y),
                    5,
                    (0, 0, 255) if climbing_candidate else (255, 255, 0),
                    -1,
                )

        for track_id, state in list(track_states.items()):
            if track_id not in visible_track_ids:
                elapsed_frames = max(
                    1, frame_index - state.get("last_eval_frame", frame_index - 1)
                )
                state["last_eval_frame"] = frame_index
                state["safe_streak"] += elapsed_frames
                if state["safe_streak"] >= clear_frames:
                    state["is_climbing"] = False
                    state["climbing_streak"] = 0

                if frame_index - state["last_seen"] <= max(1, round(fps * 0.15)):
                    x1, y1, x2, y2 = state["last_box"]
                    color = (0, 0, 255) if state["is_climbing"] else (0, 165, 255)
                    label = "WARNING: CLIMBING!" if state["is_climbing"] else "Tracking..."
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
                if frame_index - state["last_seen"] > fps * 2:
                    del track_states[track_id]

        for detected_object in objects:
            x1, y1, x2, y2 = detected_object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame, "Object", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        if any(state["is_climbing"] for state in track_states.values()):
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(frame, "DANGER: CLIMBING DETECTED!", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        return frame, alerts

    def _send_line_alert(self, alert):
        reason = "เท้าพ้นพื้น" if alert["reason"] == "FEET OFF GROUND" else "อยู่บนวัตถุที่ยกสูง"
        text = (
            "🚨 ตรวจพบอันตรายจากการปีน\n"
            f"ตำแหน่งในภาพ: x={alert['x']}, y={alert['y']}\n"
            f"Track ID: {alert['track_id']}\n"
            f"สาเหตุ: {reason}\n"
            f"เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self._send_line_message(text)

    def send_line_test(self):
        text = (
            "✅ ทดสอบการแจ้งเตือน LINE สำเร็จ\n"
            f"ระบบตรวจจับการปีนพร้อมส่งข้อความ\n"
            f"เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return self._send_line_message(text)

    def _send_line_message(self, text):
        token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
        target_id = os.getenv("LINE_TARGET_ID")
        if not token or not target_id:
            missing = [
                name
                for name, value in (
                    ("LINE_CHANNEL_ACCESS_TOKEN", token),
                    ("LINE_TARGET_ID", target_id),
                )
                if not value
            ]
            message = f"ยังส่งแจ้งเตือนไม่ได้: กรุณาตั้งค่า {', '.join(missing)}"
            logging.warning(message)
            with self.lock:
                self.alert_status = message
            return False

        payload = json.dumps(
            {"to": target_id, "messages": [{"type": "text", "text": text}]}
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.line.me/v2/bot/message/push",
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError(f"LINE API ตอบกลับ HTTP {response.status}")
            with self.lock:
                self.alert_status = f"ส่งแจ้งเตือน LINE ล่าสุดแล้ว ({datetime.now():%H:%M:%S})"
            return True
        except urllib.error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            try:
                response_data = json.loads(response_body)
            except json.JSONDecodeError:
                response_data = {}
            api_message = response_data.get("message")
            message = f"LINE API HTTP {exc.code}"
            if api_message:
                message += f": {api_message}"
            logging.error("ส่งข้อความ LINE ไม่สำเร็จ: %s", message)
            with self.lock:
                self.alert_status = f"ส่ง LINE ไม่สำเร็จ: {message}"
            return False
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            logging.error("ส่งข้อความ LINE ไม่สำเร็จ: %s", exc)
            with self.lock:
                self.alert_status = f"ส่ง LINE ไม่สำเร็จ: {exc}"
            return False


def make_handler(monitor):
    class RequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format_string, *args):
            logging.info("%s - %s", self.address_string(), format_string % args)

        def _json_response(self, status_code, data):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/status":
                self._json_response(200, monitor.get_status())
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last_frame = None
                try:
                    while monitor.server_running:
                        with monitor.lock:
                            jpeg = monitor.latest_jpeg
                        if jpeg and jpeg != last_frame:
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                                + jpeg
                                + b"\r\n"
                            )
                            self.wfile.flush()
                            last_frame = jpeg
                        else:
                            time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._json_response(404, {"error": "ไม่พบ URL ที่ร้องขอ"})

        def do_POST(self):
            if self.path == "/api/test-line":
                success = monitor.send_line_test()
                self._json_response(
                    200 if success else 502,
                    {"ok": success, "alert_status": monitor.get_status()["alert_status"]},
                )
                return

            if self.path == "/api/camera-frame":
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if content_length <= 0 or content_length > 3 * 1024 * 1024:
                        raise ValueError("ขนาดภาพไม่ถูกต้อง")
                    frame = self.rfile.read(content_length)
                    if len(frame) != content_length:
                        raise ValueError("ได้รับภาพจากกล้องไม่ครบ")
                    result = monitor.process_browser_camera_frame(frame)
                    self._json_response(200, result)
                except ValueError as exc:
                    self._json_response(400, {"error": str(exc)})
                except RuntimeError as exc:
                    logging.exception("ประมวลผลภาพกล้องไม่สำเร็จ")
                    self._json_response(500, {"error": str(exc)})
                return

            if self.path == "/api/upload":
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    filename = unquote(self.headers.get("X-File-Name", ""))
                    saved_name = monitor.upload_video(
                        filename, self.rfile, content_length
                    )
                    self._json_response(200, {"ok": True, "filename": saved_name})
                except (ValueError, OSError) as exc:
                    self._json_response(400, {"error": str(exc)})
                return

            if self.path not in ("/api/start", "/api/stop"):
                self._json_response(404, {"error": "ไม่พบ URL ที่ร้องขอ"})
                return
            try:
                if self.path == "/api/stop":
                    monitor.stop()
                    self._json_response(200, {"ok": True})
                    return

                content_length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(content_length) or b"{}")
                monitor.start(data.get("source"))
                self._json_response(200, {"ok": True})
            except (ValueError, json.JSONDecodeError) as exc:
                self._json_response(400, {"error": str(exc)})

    return RequestHandler


def main():
    monitor = VideoMonitor()
    server = ThreadingHTTPServer((HOST, PORT), make_handler(monitor))
    logging.info("เปิดหน้าเว็บตรวจจับที่ http://%s:%s", HOST, PORT)
    webbrowser.open(f"http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("กำลังปิดเซิร์ฟเวอร์")
    finally:
        monitor.server_running = False
        monitor.stop()
        server.server_close()
        if monitor.worker is not None:
            try:
                monitor.worker.join(timeout=10)
            except KeyboardInterrupt:
                logging.warning(
                    "ได้รับ Ctrl+C ระหว่างรอปิดงานตรวจจับ "
                    "โปรแกรมจะปิดโดยไม่แสดง traceback เพิ่ม"
                )
        if monitor.pending_video_path is not None:
            monitor.pending_video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
