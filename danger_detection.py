"""
监控视频危险事件检测系统 v6 (调试版)
支持单视频/单类别调试，优化阈值，添加边缘屏蔽和调试日志
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
from datetime import datetime
from pathlib import Path
from collections import deque

class DangerEventDetector:
    def __init__(self, log_dir="detection_logs", sample_interval=10, save_frames=True,
                 debug_video=None, debug_category=None,
                 fire_margin=0.05, person_margin=0.1,
                 fall_shrink_threshold=0.75, fall_low_threshold=0.60,
                 fall_speed_threshold=0.005, fall_duration_multiplier=1.5,
                 enable_debug_log=True, detection_mode="unrestricted"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"detection_log_{self.timestamp}.txt"
        self.enable_debug_log = enable_debug_log
        self.detection_mode = detection_mode

        self.f_log = open(self.log_file, "w", encoding="utf-8")

        self.sample_interval = sample_interval
        self.save_frames = save_frames
        self.frame_save_dir = self.log_dir / f"detected_frames_{self.timestamp}"
        if self.save_frames:
            self.frame_save_dir.mkdir(exist_ok=True)

        self.debug_video = debug_video
        self.debug_category = debug_category
        self.fire_margin = fire_margin
        self.person_margin = person_margin
        self.fall_shrink_threshold = fall_shrink_threshold
        self.fall_low_threshold = fall_low_threshold
        self.fall_speed_threshold = fall_speed_threshold
        self.fall_duration_multiplier = fall_duration_multiplier

        self.pose_model = None
        self.prev_gray = None
        self.fire_history = deque(maxlen=5)
        self.person_tracking = {}

        self.debug_info = {
            "fire_regions_all": [],
            "fall_candidates": [],
            "fight_candidates": [],
            "pose_detected_frames": 0,
            "fire_detected_frames": 0,
            "person_edge_filtered": 0,
            "person_edge_mask_applied": False,
            "fire_edge_filtered": 0,
            "fire_edge_mask_applied": False
        }

        self.log("=" * 80)
        self.log("监控视频危险事件检测系统 v6 初始化 (调试版)")
        mode_name = "全量检测模式" if detection_mode == "unrestricted" else "类别限制模式"
        self.log(f"检测模式: {mode_name} ({detection_mode})")
        self.log(f"抽帧间隔: 每 {self.sample_interval} 帧检测一次")
        self.log(f"保存异常帧: {'是' if self.save_frames else '否'}")
        self.log(f"调试日志: {'启用' if self.enable_debug_log else '禁用'}")
        if self.debug_video:
            self.log(f"调试模式: 仅处理视频 {self.debug_video}")
        if self.debug_category:
            self.log(f"调试模式: 仅处理类别 {self.debug_category}")
        self.log(f"火焰边缘屏蔽: {fire_margin*100}%")
        self.log(f"人体边缘屏蔽: {person_margin*100}%")
        self.log(f"摔倒身高缩水阈值: {fall_shrink_threshold}")
        self.log(f"摔倒低位阈值: {fall_low_threshold}")
        self.log("=" * 80)

    def __del__(self):
        if hasattr(self, 'f_log') and self.f_log:
            self.f_log.close()

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.f_log.write(log_entry + "\n")
        self.f_log.flush()

    def save_debug_info(self, video_path):
        """保存详细调试信息"""
        video_name = Path(video_path).stem
        debug_file = self.log_dir / f"debug_{video_name}_{self.timestamp}.json"

        debug_data = {
            "video": video_name,
            "timestamp": self.timestamp,
            "fire": {
                "detected_frames": self.debug_info.get("fire_detected_frames", 0),
                "edge_filtered_pixels": int(self.debug_info.get("fire_edge_filtered", 0)),
                "edge_mask_applied": self.debug_info.get("fire_edge_mask_applied", False),
                "sample_entries": self.debug_info.get("fire_regions_all", [])[:20]
            },
            "fall": {
                "candidates_count": len(self.debug_info.get("fall_candidates", [])),
                "sample_entries": self.debug_info.get("fall_candidates", [])[:50]
            },
            "fight": {
                "candidates_count": len(self.debug_info.get("fight_candidates", [])),
                "sample_entries": self.debug_info.get("fight_candidates", [])[:20]
            },
            "pose": {
                "detected_frames": self.debug_info.get("pose_detected_frames", 0),
                "edge_filtered_persons": int(self.debug_info.get("person_edge_filtered", 0))
            }
        }

        try:
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            self.log(f"[DEBUG] 调试信息已保存: {debug_file.name}")
        except Exception as e:
            self.log(f"[WARNING] 调试信息保存失败: {e}")

    def log_debug(self, message):
        if self.debug_video or self.debug_category:
            self.log(f"[DEBUG] {message}")

    def load_models(self):
        self.log("-" * 40)
        self.log("加载预训练模型...")
        try:
            self.pose_model = YOLO("yolov8n-pose.pt")
            self.log("[OK] YOLOv8n-Pose 姿态估计模型加载成功")
        except Exception as e:
            self.log(f"[ERROR] YOLOv8n-Pose 加载失败: {e}")
        self.log("-" * 40)

    def create_edge_mask(self, h, w, margin):
        """创建边缘屏蔽掩码"""
        mask = np.ones((h, w), dtype=np.uint8) * 255
        margin_h = int(h * margin)
        margin_w = int(w * margin)
        mask[:margin_h, :] = 0
        mask[-margin_h:, :] = 0
        mask[:, :margin_w] = 0
        mask[:, -margin_w:] = 0
        return mask

    def draw_fire_detection(self, frame, fire_events):
        """绘制火焰检测结果"""
        for event in fire_events:
            h, w = frame.shape[:2]
            confidence = event.get("confidence", 0)
            fire_regions = event.get("fire_regions", 0)
            fire_areas_data = event.get("fire_areas_data", [])

            for region in fire_areas_data[:5]:
                x, y, x2, y2 = region.get("bbox", [0, 0, 0, 0])
                if x2 > x and y2 > y:
                    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"FIRE#{len(fire_areas_data)} area:{region.get('area', 0):.0f}",
                               (x, max(y - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            label = f"FIRE! conf:{confidence:.2f} areas:{fire_regions}"
            cv2.rectangle(frame, (10, 10), (w-10, 50), (0, 0, 255), -1)
            cv2.putText(frame, label, (15, 38),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return frame

    def draw_person_keypoints(self, frame, keypoints, color=(0, 255, 0), track_id=None):
        """绘制人体关键点骨架"""
        if len(keypoints) < 17:
            return frame

        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

        for i, j in skeleton:
            if i < len(keypoints) and j < len(keypoints):
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(frame, pt1, pt2, color, 2)

        if track_id is not None:
            nose = keypoints[0]
            cv2.putText(frame, f"ID:{track_id}", (int(nose[0]), int(nose[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def draw_fall_detection(self, frame, fall_events, current_persons=None):
        """绘制摔倒检测结果"""
        for event in fall_events:
            h, w = frame.shape[:2]
            confidence = event.get("confidence", 0)
            fall_stage = event.get("stage", "unknown")
            height_shrink = event.get("height_shrink", 0)
            fall_speed = event.get("fall_speed", 0)

            label = f"FALL {confidence:.2f} [{fall_stage}] shrink:{height_shrink:.2f} speed:{fall_speed:.1f}"
            cv2.rectangle(frame, (10, 10), (w-10, 60), (0, 165, 255), -1)
            cv2.putText(frame, label, (15, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if current_persons:
            for p in current_persons:
                if p.get("has_fall"):
                    self.draw_person_keypoints(frame, p["keypoints"], color=(0, 165, 255), track_id=p.get("track_id"))
        return frame

    def draw_violence_detection(self, frame, fight_events, current_persons=None):
        """绘制冲突检测结果"""
        for event in fight_events:
            h, w = frame.shape[:2]
            confidence = event.get("confidence", 0)
            iou_overlap = event.get("iou_overlap", 0)
            min_distance = event.get("min_person_distance", 0)

            label = f"VIOLENCE! conf:{confidence:.2f}"
            cv2.rectangle(frame, (10, 10), (w-10, 80), (255, 0, 0), -1)
            cv2.putText(frame, label, (15, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f"IoU:{iou_overlap:.3f} dist:{min_distance:.0f}",
                       (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if current_persons and len(current_persons) >= 2:
            for p in current_persons:
                self.draw_person_keypoints(frame, p["keypoints"], color=(255, 0, 0), track_id=p.get("track_id"))
        return frame

    def detect_fire_smoke(self, frame, frame_count, edge_mask=None):
        """火焰检测 - 优化版：边缘屏蔽+区域合并"""
        results = []
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)
        else:
            motion = np.ones((h, w), dtype=np.uint8) * 255

        self.prev_gray = gray.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_fire1 = np.array([0, 120, 100])
        upper_fire1 = np.array([15, 255, 255])
        fire_mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)

        lower_fire2 = np.array([165, 120, 100])
        upper_fire2 = np.array([180, 255, 255])
        fire_mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)
        fire_mask_static = fire_mask1 | fire_mask2

        fire_mask = cv2.bitwise_and(fire_mask_static, fire_mask_static, mask=motion)

        if edge_mask is not None:
            fire_mask_orig = fire_mask.copy()
            fire_mask = cv2.bitwise_and(fire_mask, fire_mask, mask=edge_mask)
            filtered_pixels = cv2.countNonZero(fire_mask_orig) - cv2.countNonZero(fire_mask)
            self.debug_info["fire_edge_filtered"] += filtered_pixels
            self.debug_info["fire_edge_mask_applied"] = True

        fire_pixels = cv2.countNonZero(fire_mask)
        fire_ratio = fire_pixels / (w * h)

        kernel = np.ones((15, 15), np.uint8)
        fire_mask_dilated = cv2.dilate(fire_mask, kernel, iterations=2)

        contours, _ = cv2.findContours(fire_mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        temp_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, cw, ch = cv2.boundingRect(cnt)
                temp_regions.append({"bbox": [x, y, x+cw, y+ch], "area": area})

        merged_regions = []
        used = set()

        for i in range(len(temp_regions)):
            if i in used:
                continue
            reg1 = temp_regions[i]
            b1 = reg1["bbox"]
            merged = {
                "bbox": b1.copy(),
                "area": reg1["area"],
                "count": 1
            }
            used.add(i)

            for j in range(i + 1, len(temp_regions)):
                if j in used:
                    continue
                reg2 = temp_regions[j]
                b2 = reg2["bbox"]
                distance = np.sqrt(
                    ((b1[0] + b1[2]) / 2 - (b2[0] + b2[2]) / 2)**2 +
                    ((b1[1] + b1[3]) / 2 - (b2[1] + b2[3]) / 2)**2
                )
                max_dim = max(b1[2]-b1[0], b1[3]-b1[1], b2[2]-b2[0], b2[3]-b2[1])
                
                if distance < max_dim * 4.0:
                    merged["bbox"][0] = min(merged["bbox"][0], b2[0])
                    merged["bbox"][1] = min(merged["bbox"][1], b2[1])
                    merged["bbox"][2] = max(merged["bbox"][2], b2[2])
                    merged["bbox"][3] = max(merged["bbox"][3], b2[3])
                    merged["area"] += reg2["area"]
                    merged["count"] += 1
                    used.add(j)

            merged_regions.append(merged)

        fire_regions = []
        fire_areas_data = []
        total_fire_area = 0

        for mr in merged_regions:
            fire_regions.append({"bbox": mr["bbox"], "area": mr["area"]})
            fire_areas_data.append({"bbox": mr["bbox"], "area": mr["area"]})
            total_fire_area += mr["area"]

        self.fire_history.append({
            "frame": frame_count,
            "fire_ratio": fire_ratio,
            "regions": len(fire_regions),
            "total_area": total_fire_area,
            "areas": [r["area"] for r in fire_regions]
        })

        fire_score = 0
        expansion_rate = 0
        if fire_ratio > 0.003 and len(fire_regions) > 0:
            base_score = min(fire_ratio * 40 + min(len(fire_regions), 3) * 0.2, 1.0)

            if len(self.fire_history) >= 3:
                areas_prev = self.fire_history[-3]["total_area"]
                areas_curr = self.fire_history[-1]["total_area"]
                if areas_prev > 0:
                    expansion_rate = (areas_curr - areas_prev) / areas_prev

            if expansion_rate > 0.1:
                base_score *= 1.2

            fire_score = min(base_score, 1.0)

        if self.enable_debug_log and frame_count % 30 == 0 and fire_ratio > 0.001:
            self.log(f"  [DEBUG-FIRE] frame={frame_count} fire_ratio={fire_ratio:.4f} regions={len(fire_regions)} total_area={total_fire_area:.0f}")

        if fire_score > 0.2 and len(fire_regions) > 0:
            results.append({
                "type": "明火",
                "confidence": round(fire_score, 3),
                "fire_ratio": round(fire_ratio, 4),
                "fire_regions": len(fire_regions),
                "total_fire_area": round(total_fire_area, 1),
                "fire_areas": [r["area"] for r in fire_regions],
                "fire_areas_data": fire_areas_data,
                "expansion_rate": round(expansion_rate, 3)
            })

            self.debug_info["fire_detected_frames"] += 1
            self.debug_info["fire_regions_all"].append({
                "frame": frame_count,
                "regions": len(fire_regions),
                "total_area": total_fire_area,
                "score": fire_score
            })

        return results

    def detect_persons_pose(self, frame, frame_count, edge_mask=None):
        """使用Pose模型检测人体 - 开启ID追踪+边缘过滤"""
        if self.pose_model is None:
            return []

        try:
            results = self.pose_model.track(frame, persist=True, verbose=False)
            if len(results) == 0:
                return []

            pose_result = results[0]
            if pose_result.keypoints is None:
                return []

            keypoints_all = pose_result.keypoints.xy
            boxes = pose_result.boxes

            h, w = frame.shape[:2]
            persons = []
            edge_filtered_count = 0

            for i, kps in enumerate(keypoints_all):
                if len(kps) >= 17:
                    conf = float(boxes.conf[i]) if boxes is not None else 1.0
                    box = boxes.xyxy[i].cpu().numpy() if boxes is not None else [0, 0, 0, 0]
                    track_id = int(boxes.id[i]) if boxes.id is not None else i

                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2

                    if edge_mask is not None:
                        mask_h, mask_w = edge_mask.shape
                        scale_x = mask_w / w
                        scale_y = mask_h / h
                        mask_x = int(center_x * scale_x)
                        mask_y = int(center_y * scale_y)

                        if 0 <= mask_x < mask_w and 0 <= mask_y < mask_h:
                            if edge_mask[mask_y, mask_x] == 0:
                                edge_filtered_count += 1
                                if self.enable_debug_log and frame_count % 50 == 0:
                                    self.log(f"  [边缘过滤] frame={frame_count} track_id={track_id} center=({center_x:.0f},{center_y:.0f}) 位于画面边缘")
                                continue

                    shoulder_center = ((kps[5][0] + kps[6][0]) / 2, (kps[5][1] + kps[6][1]) / 2)
                    hip_center = ((kps[11][0] + kps[12][0]) / 2, (kps[11][1] + kps[12][1]) / 2)
                    body_center = ((shoulder_center[0] + hip_center[0]) / 2,
                                  (shoulder_center[1] + hip_center[1]) / 2)

                    nose_y = float(kps[0][1])
                    ankle_y = float(kps[16][1])
                    body_height = ankle_y - nose_y

                    persons.append({
                        "keypoints": kps.cpu().numpy(),
                        "confidence": conf,
                        "box": box,
                        "track_id": track_id,
                        "body_center": (float(body_center[0]), float(body_center[1])),
                        "hip_center": (float(hip_center[0]), float(hip_center[1])),
                        "nose_y": nose_y,
                        "ankle_y": ankle_y,
                        "body_height": body_height,
                        "has_fall": False
                    })

            self.debug_info["pose_detected_frames"] += 1
            self.debug_info["person_edge_filtered"] += edge_filtered_count

            if self.enable_debug_log and edge_filtered_count > 0 and frame_count % 20 == 0:
                self.log(f"  [DEBUG-POSE] frame={frame_count} 检测到{len(persons)}人 边缘过滤{edge_filtered_count}人")

            return persons
        except KeyError as e:
            self.log(f"[WARNING] Pose检测KeyError: {e}, debug_info未正确初始化")
            return persons
        except Exception as e:
            self.log(f"[WARNING] Pose检测异常: {e}")
            return []

    def detect_fall_temporal(self, person, frame_count, fps):
        """摔倒检测 - 优化阈值"""
        h = person.get("frame_height", 720)
        track_id = person.get("track_id", 0)
        nose_y = person["nose_y"]
        ankle_y = person["ankle_y"]
        body_height = person["body_height"]
        hip_center = person["hip_center"]

        if track_id not in self.person_tracking:
            self.person_tracking[track_id] = {
                "history": [],
                "fall_stage": "normal",
                "low_position_start": None,
                "initial_body_height": None,
            }

        tracking = self.person_tracking[track_id]

        if tracking["initial_body_height"] is None:
            tracking["initial_body_height"] = body_height

        initial_height = tracking["initial_body_height"]
        height_shrink_ratio = body_height / max(initial_height, 1)

        tracking["history"].append({
            "frame": frame_count,
            "hip_y": hip_center[1],
            "nose_y": nose_y,
            "body_height": body_height,
            "ankle_y": ankle_y
        })

        if len(tracking["history"]) > 90:
            tracking["history"].pop(0)

        results = []

        if len(tracking["history"]) >= 3:
            y_current = tracking["history"][-1]["hip_y"]
            y_prev = tracking["history"][0]["hip_y"]
            frames_diff = tracking["history"][-1]["frame"] - tracking["history"][0]["frame"]

            if frames_diff > 0:
                fall_speed = (y_current - y_prev) / frames_diff
            else:
                fall_speed = 0

            nose_y_norm = nose_y / h
            is_low_position = nose_y_norm > self.fall_low_threshold
            is_height_shrunk = height_shrink_ratio < self.fall_shrink_threshold
            is_falling_down = fall_speed > h * self.fall_speed_threshold

            is_stationary = False
            if len(tracking["history"]) >= 2:
                displacements = [
                    float(np.sqrt(float(tracking["history"][i]["hip_y"] - tracking["history"][i-1]["hip_y"])**2))
                    for i in range(1, len(tracking["history"]))
                ]
                total_displacement = sum(displacements)
                is_stationary = total_displacement < h * 0.15

            debug_entry = {
                "frame": frame_count,
                "track_id": track_id,
                "height_shrink_ratio": round(height_shrink_ratio, 3),
                "nose_y_norm": round(nose_y_norm, 3),
                "fall_speed": round(fall_speed, 3),
                "is_low": is_low_position,
                "is_shrunk": is_height_shrunk,
                "is_falling": is_falling_down,
                "is_stationary": is_stationary,
                "stage": tracking["fall_stage"]
            }

            if self.enable_debug_log and frame_count % 30 == 0:
                self.log(f"  [DEBUG-FALL] frame={frame_count} track_id={track_id} "
                        f"shrink={height_shrink_ratio:.3f} nose_norm={nose_y_norm:.3f} "
                        f"speed={fall_speed:.3f} low={is_low_position} shrunk={is_height_shrunk} "
                        f"falling={is_falling_down} stage={tracking['fall_stage']}")

            if (is_height_shrunk and is_low_position) or (is_height_shrunk and is_falling_down):
                tracking["fall_stage"] = "suspect"
                tracking["low_position_start"] = frame_count
                confidence = min(0.6 + (1 - height_shrink_ratio) * 0.3, 0.9)
                results.append({
                    "type": "疑似摔倒",
                    "confidence": round(confidence, 3),
                    "stage": "suspect",
                    "height_shrink": round(1 - height_shrink_ratio, 3),
                    "fall_speed": round(fall_speed, 2)
                })
                person["has_fall"] = True
                debug_entry["detected"] = True
                self.debug_info["fall_candidates"].append(debug_entry)
            elif tracking["fall_stage"] == "suspect":
                low_duration = frame_count - (tracking["low_position_start"] or frame_count)
                expected_low_duration = self.fall_duration_multiplier * fps

                if is_low_position and is_stationary and low_duration > expected_low_duration:
                    tracking["fall_stage"] = "confirmed"
                    confidence = min(0.8 + (low_duration / expected_low_duration) * 0.2, 1.0)
                    results.append({
                        "type": "确认摔倒",
                        "confidence": round(confidence, 3),
                        "stage": "confirmed",
                        "duration": low_duration,
                        "height_shrink": round(1 - height_shrink_ratio, 3)
                    })
                    person["has_fall"] = True
                elif is_low_position:
                    results.append({
                        "type": "疑似摔倒",
                        "confidence": 0.7,
                        "stage": "suspect",
                        "low_duration": low_duration,
                        "height_shrink": round(1 - height_shrink_ratio, 3)
                    })
                    person["has_fall"] = True
                else:
                    tracking["fall_stage"] = "normal"
                    tracking["low_position_start"] = None
                    person["has_fall"] = False

                debug_entry["detected"] = True
                self.debug_info["fall_candidates"].append(debug_entry)
            else:
                debug_entry["detected"] = False
                if len(self.debug_info["fall_candidates"]) < 100:
                    self.debug_info["fall_candidates"].append(debug_entry)

        return results

    def calculate_iou(self, box1, box2):
        """计算两个BBox的IoU"""
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1]))
        box2_area = (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1]))

        union_area = box1_area + box2_area - inter_area

        if union_area > 0:
            iou = inter_area / union_area
        else:
            iou = 0

        return float(iou)

    def detect_violence_keypoints(self, persons, frame=None):
        """冲突检测 - BBox叠加热度判定"""
        results = []

        if len(persons) < 2 or frame is None:
            return results

        h, w = frame.shape[:2]

        person_distances = []
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                dist = float(np.sqrt(
                    (float(persons[i]["body_center"][0]) - float(persons[j]["body_center"][0]))**2 +
                    (float(persons[i]["body_center"][1]) - float(persons[j]["body_center"][1]))**2
                ))
                person_distances.append((i, j, dist))

        person_distances.sort(key=lambda x: x[2])
        min_distance = person_distances[0][2] if person_distances else float('inf')

        proximity_threshold = h * 0.4
        is_close = min_distance < proximity_threshold

        iou_overlap = 0
        if is_close:
            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    iou = self.calculate_iou(persons[i]["box"], persons[j]["box"])
                    if iou > 0:
                        iou_overlap = max(iou_overlap, iou)

        interaction_score = 0.0
        if is_close:
            interaction_score = float(1 - (min_distance / proximity_threshold))

            for i in range(len(persons)):
                for j in range(i + 1, len(persons)):
                    kps_i = persons[i]["keypoints"]
                    kps_j = persons[j]["keypoints"]

                    left_hand_i = kps_i[7]
                    right_hand_i = kps_i[10]
                    left_hand_j = kps_j[7]
                    right_hand_j = kps_j[10]

                    hand_distances = [
                        float(np.sqrt((float(left_hand_i[0]) - float(left_hand_j[0]))**2 + (float(left_hand_i[1]) - float(left_hand_j[1]))**2)),
                        float(np.sqrt((float(left_hand_i[0]) - float(right_hand_j[0]))**2 + (float(left_hand_i[1]) - float(right_hand_j[1]))**2)),
                        float(np.sqrt((float(right_hand_i[0]) - float(left_hand_j[0]))**2 + (float(right_hand_i[1]) - float(left_hand_j[1]))**2)),
                        float(np.sqrt((float(right_hand_i[0]) - float(right_hand_j[0]))**2 + (float(right_hand_i[1]) - float(right_hand_j[1]))**2))
                    ]
                    min_hand_dist = float(min(hand_distances))

                    if min_hand_dist < h * 0.25:
                        interaction_score += 0.2

        confidence = min(max(interaction_score, iou_overlap * 2), 1.0)
        if confidence > 0.3:
            results.append({
                "type": "剧烈运动/冲突",
                "confidence": round(confidence, 3),
                "min_person_distance": round(min_distance, 1),
                "iou_overlap": round(iou_overlap, 3),
                "persons_count": len(persons)
            })

            self.debug_info["fight_candidates"].append({
                "frame": 0,
                "min_distance": round(min_distance, 1),
                "iou_overlap": round(iou_overlap, 3),
                "confidence": round(confidence, 3)
            })

        return results

    def detect_violence_optical_flow(self, frames_buffer, person_boxes=None):
        """冲突检测 - 连续帧光流"""
        results = []

        if len(frames_buffer) < 3:
            return results

        gray_frames = []
        for frame in frames_buffer[-3:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frames.append(gray)

        flow1 = cv2.calcOpticalFlowFarneback(
            cv2.resize(gray_frames[0], (320, 240)),
            cv2.resize(gray_frames[1], (320, 240)),
            None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        flow2 = cv2.calcOpticalFlowFarneback(
            cv2.resize(gray_frames[1], (320, 240)),
            cv2.resize(gray_frames[2], (320, 240)),
            None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        avg_flow = (flow1 + flow2) / 2
        magnitude = np.sqrt(avg_flow[..., 0]**2 + avg_flow[..., 1]**2)

        violent_regions = 0
        max_magnitude = 0
        total_magnitude = 0

        h, w = magnitude.shape
        block_h, block_w = h // 4, w // 4

        for i in range(4):
            for j in range(4):
                block = magnitude[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                block_mean = float(np.mean(block))
                if block_mean > 3.0:
                    violent_regions += 1
                max_magnitude = max(max_magnitude, float(np.max(block)))
                total_magnitude += block_mean

        avg_magnitude = total_magnitude / 16

        violence_score = 0
        if violent_regions >= 3:
            violence_score = min(
                (violent_regions / 16) * 0.5 +
                (max_magnitude / 20) * 0.3 +
                (avg_magnitude / 5) * 0.2,
                1.0
            )

        if violence_score > 0.25 and violent_regions >= 3:
            results.append({
                "type": "剧烈运动/冲突(光流)",
                "confidence": round(violence_score, 3),
                "max_magnitude": round(max_magnitude, 2),
                "violent_regions": violent_regions,
                "avg_motion": round(avg_magnitude, 2)
            })

        return results

    def save_annotated_frame(self, frame, video_name, frame_count, events, category, current_persons=None):
        """保存带标注的异常帧 - 按视频目录统一保存，置信度最高的事件类型命名"""
        safe_video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).strip()
        save_dir = self.frame_save_dir / safe_video_name
        save_dir.mkdir(exist_ok=True)

        annotated_frame = frame.copy()

        fire_events = [e for e in events if "明火" in e.get("type", "")]
        fall_events = [e for e in events if "摔倒" in e.get("type", "")]
        fight_events = [e for e in events if "冲突" in e.get("type", "") or "剧烈运动" in e.get("type", "")]

        if fire_events:
            annotated_frame = self.draw_fire_detection(annotated_frame, fire_events)

        if fall_events:
            annotated_frame = self.draw_fall_detection(annotated_frame, fall_events, current_persons)

        if fight_events:
            annotated_frame = self.draw_violence_detection(annotated_frame, fight_events, current_persons)

        if events:
            best_event = max(events, key=lambda e: e.get("confidence", 0))
            event_type = best_event['type']
        else:
            event_type = "unknown"
        filename = f"{safe_video_name}_frame{frame_count:06d}_{event_type}.jpg"
        filepath = save_dir / filename

        cv2.imwrite(str(filepath), annotated_frame)
        self.log(f"  [保存异常帧] {filename}")

        return filepath

    def process_video(self, video_path, category):
        """处理单个视频"""
        self.log("-" * 80)
        self.log(f"开始处理视频: {video_path}")
        self.log(f"预期危险类别: {category}")

        self.fire_history.clear()
        self.person_tracking.clear()

        debug_video_name = os.path.basename(video_path)
        if self.debug_video and self.debug_video not in debug_video_name:
            self.log(f"[跳过] 不在调试视频范围内: {debug_video_name}")
            return None

        if self.debug_category and self.debug_category != category:
            self.log(f"[跳过] 不在调试类别范围内: {category}")
            return None

        if not os.path.exists(video_path):
            self.log(f"[ERROR] 视频文件不存在: {video_path}")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log(f"[ERROR] 无法打开视频: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.log(f"视频信息: {width}x{height}, {fps:.1f}fps, 共{total_frames}帧")
        frames_to_process = total_frames // self.sample_interval
        self.log(f"将检测约 {frames_to_process} 帧")

        edge_mask = self.create_edge_mask(height, width, max(self.fire_margin, self.person_margin))

        frame_count = 0
        processed_count = 0
        events_log = []
        saved_frames = []
        frames_buffer = deque(maxlen=5)
        prev_frame = None

        self.prev_gray = None
        self.log("开始抽帧分析（v6调试版）...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h, w = frame.shape[:2]

            frames_buffer.append(frame)

            if frame_count % self.sample_interval != 0:
                prev_frame = frame.copy()
                continue

            processed_count += 1

            if processed_count % 10 == 0:
                self.log(f"  进度: {processed_count}/{frames_to_process} 帧 ({processed_count*100//max(1,frames_to_process)}%)")

            all_events = []
            current_persons = []

            fire_events = []
            if self.detection_mode == "restricted":
                if "fire" in category.lower():
                    fire_events = self.detect_fire_smoke(frame, frame_count, edge_mask)
            else:
                fire_events = self.detect_fire_smoke(frame, frame_count, edge_mask)

            if fire_events:
                all_events.extend(fire_events)
                for event in fire_events:
                    self.log(f"  [危险检测] 帧{frame_count}: {event['type']} - 置信度:{event['confidence']:.3f} 区域数:{event['fire_regions']} 总面积:{event['total_fire_area']:.0f}")

            fall_events_list = []
            fight_events_kp_list = []
            fight_events_flow_list = []

            if self.detection_mode == "restricted":
                if "fall" in category.lower() or "fight" in category.lower():
                    persons = self.detect_persons_pose(frame, frame_count, edge_mask)

                    for person in persons:
                        person["frame_height"] = h
                        current_persons.append(person)

                    if "fall" in category.lower():
                        for person in persons:
                            fall_events = self.detect_fall_temporal(person, frame_count, fps)
                            if fall_events:
                                fall_events_list.extend(fall_events)
                                for event in fall_events:
                                    self.log(f"  [危险检测] 帧{frame_count}: {event['type']} - 置信度:{event['confidence']:.3f} [{event.get('stage','?')}]")

                    if "fight" in category.lower():
                        if len(persons) >= 2:
                            fight_events_kp = self.detect_violence_keypoints(persons, frame)
                            if fight_events_kp:
                                fight_events_kp_list.extend(fight_events_kp)
                                for event in fight_events_kp:
                                    self.log(f"  [危险检测] 帧{frame_count}: {event['type']} - 置信度:{event['confidence']:.3f} IoU:{event.get('iou_overlap', 0):.3f}")

                        if len(frames_buffer) >= 3:
                            fight_events_flow = self.detect_violence_optical_flow(list(frames_buffer))
                            if fight_events_flow:
                                for event in fight_events_flow:
                                    event["type"] = "剧烈运动/冲突(光流)"
                                fight_events_flow_list.extend(fight_events_flow)
                                for event in fight_events_flow:
                                    self.log(f"  [危险检测] 帧{frame_count}: {event['type']} - 置信度:{event['confidence']:.3f}")
            else:
                persons = self.detect_persons_pose(frame, frame_count, edge_mask)

                for person in persons:
                    person["frame_height"] = h
                    current_persons.append(person)

                for person in persons:
                    fall_events = self.detect_fall_temporal(person, frame_count, fps)
                    if fall_events:
                        fall_events_list.extend(fall_events)
                        for event in fall_events:
                            self.log(f"  [危险检测] 帧{frame_count}: {event['type']} - 置信度:{event['confidence']:.3f} [{event.get('stage','?')}]")

                if len(persons) >= 2:
                    fight_events_kp = self.detect_violence_keypoints(persons, frame)
                    if fight_events_kp:
                        fight_events_kp_list.extend(fight_events_kp)
                        for event in fight_events_kp:
                            self.log(f"  [危险检测] 帧{frame_count}: {event['type']} - 置信度:{event['confidence']:.3f} IoU:{event.get('iou_overlap', 0):.3f}")

                if len(frames_buffer) >= 3:
                    fight_events_flow = self.detect_violence_optical_flow(list(frames_buffer))
                    if fight_events_flow:
                        for event in fight_events_flow:
                            event["type"] = "剧烈运动/冲突(光流)"
                        fight_events_flow_list.extend(fight_events_flow)
                        for event in fight_events_flow:
                            self.log(f"  [危险检测] 帧{frame_count}: {event['type']} - 置信度:{event['confidence']:.3f}")

            all_events.extend(fall_events_list)
            all_events.extend(fight_events_kp_list)
            all_events.extend(fight_events_flow_list)

            if all_events:
                events_log.append({
                    "frame": frame_count,
                    "events": all_events
                })

                if self.save_frames:
                    saved_path = self.save_annotated_frame(
                        frame, os.path.basename(video_path), frame_count,
                        all_events, "unrestricted" if self.detection_mode == "unrestricted" else category, current_persons
                    )
                    saved_frames.append(str(saved_path))

            prev_frame = frame.copy()

        cap.release()
        self.person_tracking.clear()

        self.log(f"视频处理完成! 共检测到 {len(events_log)} 个危险帧")
        self.log(f"保存异常帧数量: {len(saved_frames)}")

        summary = self.generate_summary(video_path, category, events_log, total_frames, fps, saved_frames)
        return summary

    def generate_summary(self, video_path, category, events_log, total_frames, fps, saved_frames):
        """生成检测报告"""
        self.log("=" * 80)
        self.log("检测报告摘要")
        self.log("=" * 80)

        basename = os.path.basename(video_path)
        self.log(f"视频文件: {basename}")
        self.log(f"预期危险类别: {category}")
        self.log(f"总帧数: {total_frames}, 帧率: {fps:.1f} FPS")
        self.log(f"抽帧间隔: 每 {self.sample_interval} 帧")

        if total_frames > 0:
            processed = total_frames // self.sample_interval
            danger_rate = len(events_log) / max(1, processed) * 100
            self.log(f"检测到危险帧数: {len(events_log)} ({danger_rate:.2f}%)")

        event_types_count = {}
        for entry in events_log:
            for event in entry["events"]:
                t = event["type"]
                event_types_count[t] = event_types_count.get(t, 0) + 1

        if event_types_count:
            self.log("各类危险事件统计:")
            for etype, count in event_types_count.items():
                self.log(f"  - {etype}: {count} 次")

        self.log(f"保存异常帧数量: {len(saved_frames)}")
        self.log("=" * 80)

        return {
            "video_path": video_path,
            "video_name": os.path.basename(video_path),
            "category": category,
            "total_frames": total_frames,
            "sample_interval": self.sample_interval,
            "fps": fps,
            "danger_frames_count": len(events_log),
            "danger_rate": len(events_log) / max(1, total_frames // self.sample_interval) * 100 if total_frames > 0 else 0,
            "event_types": event_types_count,
            "events_log": [{"frame": e["frame"], "events": e["events"]} for e in events_log],
            "saved_frames": saved_frames
        }

    def run(self, video_dir):
        """运行检测"""
        self.log("=" * 80)
        self.log("危险事件检测系统 v6 启动 (调试版)")
        self.log("=" * 80)

        self.load_models()

        video_dir = Path(video_dir)
        if not video_dir.exists():
            self.log(f"[ERROR] 视频目录不存在: {video_dir}")
            return

        all_results = []

        category_dirs = [d for d in video_dir.iterdir() if d.is_dir()]
        for category_dir in category_dirs:
            category = category_dir.name

            if self.debug_category and self.debug_category != category:
                continue

            self.log(f"\n>>> 发现危险类别文件夹: {category}")

            videos = list(category_dir.glob("*.mp4")) + list(category_dir.glob("*.avi")) + list(category_dir.glob("*.mkv"))
            if not videos:
                self.log(f"  文件夹 {category} 中没有找到视频文件")
                continue

            for video_path in videos:
                self.fire_history.clear()
                self.person_tracking.clear()
                self.debug_info = {
                    "fire_regions_all": [],
                    "fall_candidates": [],
                    "fight_candidates": [],
                    "pose_detected_frames": 0,
                    "fire_detected_frames": 0,
                    "person_edge_filtered": 0,
                    "person_edge_mask_applied": False,
                    "fire_edge_filtered": 0,
                    "fire_edge_mask_applied": False
                }

                result = self.process_video(str(video_path), category)

                if result:
                    all_results.append(result)

                if self.enable_debug_log:
                    self.save_debug_info(video_path)

        output_file = self.log_dir / f"detection_results_{self.timestamp}.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            self.log(f"\n所有检测完成! 结果已保存到: {output_file}")
        except (TypeError, ValueError) as e:
            self.log(f"\n[ERROR] JSON保存失败: {e}")

        self.log(f"异常帧图片保存目录: {self.frame_save_dir}")

        self.print_final_summary(all_results)

        return all_results

    def print_final_summary(self, all_results):
        """打印最终汇总"""
        self.log("\n" + "=" * 80)
        self.log("最终检测结果汇总")
        self.log("=" * 80)

        for result in all_results:
            self.log(f"\n视频: {result['video_name']}")
            self.log(f"  预期类别: {result['category']}")
            self.log(f"  危险帧数: {result['danger_frames_count']} ({result['danger_rate']:.2f}%)")
            if result['event_types']:
                for etype, count in result['event_types'].items():
                    self.log(f"  - {etype}: {count}次")
            self.log(f"  保存帧数: {len(result.get('saved_frames', []))}")

        total_videos = len(all_results)
        total_danger = sum(r['danger_frames_count'] for r in all_results)
        total_saved = sum(len(r.get('saved_frames', [])) for r in all_results)

        self.log(f"\n总体统计:")
        self.log(f"  处理视频数: {total_videos}")
        self.log(f"  检测到危险的视频数: {sum(1 for r in all_results if r['danger_frames_count'] > 0)}")
        self.log(f"  总危险帧数: {total_danger}")
        self.log(f"  保存异常帧总数: {total_saved}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="监控视频危险事件检测系统 v6")
    parser.add_argument("--video", type=str, default=None, help="调试模式：仅处理包含此关键词的视频")
    parser.add_argument("--category", type=str, default=None, help="调试模式：仅处理此类别")
    parser.add_argument("--interval", type=int, default=10, help="抽帧间隔")
    parser.add_argument("--no-save", action="store_true", help="不保存异常帧")
    parser.add_argument("--fire-margin", type=float, default=0.05, help="火焰边缘屏蔽比例")
    parser.add_argument("--person-margin", type=float, default=0.10, help="人体边缘屏蔽比例")
    parser.add_argument("--fall-shrink", type=float, default=0.80, help="摔倒身高缩水阈值")
    parser.add_argument("--fall-low", type=float, default=0.55, help="摔倒低位阈值")
    parser.add_argument("--fall-speed", type=float, default=0.002, help="摔倒下坠速度阈值")
    parser.add_argument("--detection-mode", type=str, default="unrestricted",
                       choices=["restricted", "unrestricted"],
                       help="检测模式: restricted(按类别检测) / unrestricted(全量检测，默认)")
    args = parser.parse_args()

    detector = DangerEventDetector(
        log_dir="d:/1_LNY/code/MultiMod-VisionDet/detection_logs",
        sample_interval=args.interval,
        save_frames=not args.no_save,
        debug_video=args.video,
        debug_category=args.category,
        fire_margin=args.fire_margin,
        person_margin=args.person_margin,
        fall_shrink_threshold=args.fall_shrink,
        fall_low_threshold=args.fall_low,
        fall_speed_threshold=args.fall_speed,
        detection_mode=args.detection_mode
    )
    detector.run("d:/1_LNY/code/MultiMod-VisionDet/data/video")











    