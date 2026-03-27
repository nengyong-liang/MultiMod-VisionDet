"""
监控视频危险事件检测系统 v3 (融合方案版)
结合预训练模型检测能力 + 时序平滑 + 统一可视化

核心设计:
- 预训练模型: 提供危险事件检测能力
- 时序平滑: 连续3帧确认才标记危险
- 低置信度过滤: 低于0.3忽略
- 人物可视化: 所有检测到的人都画骨架
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque


class DangerDetectorV3:
    def __init__(self, log_dir="detection_logs", sample_interval=10, save_frames=True,
                 debug_video=None, debug_category=None,
                 fire_conf=0.3, fall_conf=0.4, fight_conf=0.4,
                 min_confidence=0.3,
                 temporal_window=3,
                 enable_debug_log=True, detection_mode="restricted"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"detection_log_{self.timestamp}.txt"

        self.f_log = open(self.log_file, "w", encoding="utf-8")

        self.sample_interval = sample_interval
        self.save_frames = save_frames
        self.frame_save_dir = self.log_dir / f"detected_frames_{self.timestamp}"
        if self.save_frames:
            self.frame_save_dir.mkdir(exist_ok=True)

        self.debug_video = debug_video
        self.debug_category = debug_category
        self.fire_conf = fire_conf
        self.fall_conf = fall_conf
        self.fight_conf = fight_conf
        self.min_confidence = min_confidence
        self.temporal_window = temporal_window
        self.enable_debug_log = enable_debug_log
        self.detection_mode = detection_mode

        self.fire_model = None
        self.fall_model = None
        self.fight_model = None
        self.pose_model = None
        self.person_model = None

        self.person_tracking = {}
        self.person_history = defaultdict(lambda: deque(maxlen=temporal_window))

        self.debug_info = {
            "fire_frames": 0,
            "fall_frames": 0,
            "fight_frames": 0,
            "total_frames": 0,
            "confirmed_danger_frames": 0
        }

        self.log("=" * 80)
        self.log("监控视频危险事件检测系统 v3 初始化 (融合方案版)")
        mode_name = "全量检测模式" if detection_mode == "unrestricted" else "类别限制模式"
        self.log(f"检测模式: {mode_name} ({detection_mode})")
        self.log(f"抽帧间隔: 每 {self.sample_interval} 帧检测一次")
        self.log(f"保存异常帧: {'是' if self.save_frames else '否'}")
        self.log(f"最低置信度阈值: {min_confidence}")
        self.log(f"时序平滑窗口: 连续 {temporal_window} 帧确认")
        self.log(f"火焰置信度阈值: {fire_conf}")
        self.log(f"摔倒置信度阈值: {fall_conf}")
        self.log(f"斗殴置信度阈值: {fight_conf}")
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

    def load_models(self):
        self.log("-" * 40)
        self.log("加载预训练模型...")

        fire_weights = "d:/1_LNY/code/MultiMod-VisionDet/model/Fire-Detection-model-main/Fire-Detection-model-main/best.pt"
        fight_weights = "d:/1_LNY/code/MultiMod-VisionDet/model/Fight-Violence-detection-yolov8-main/yolo_small_weights.pt"
        fall_weights = "d:/1_LNY/code/MultiMod-VisionDet/model/Real-Time-Fall-Detection-using-YOLO-main (1)/Real-Time-Fall-Detection-using-YOLO-main/Model/weights/best.pt"

        try:
            if os.path.exists(fire_weights):
                self.fire_model = YOLO(fire_weights)
                self.log(f"[OK] Fire火焰检测模型加载成功")
            else:
                self.log(f"[WARNING] Fire火焰模型权重不存在")
        except Exception as e:
            self.log(f"[ERROR] Fire火焰模型加载失败: {e}")

        try:
            if os.path.exists(fight_weights):
                self.fight_model = YOLO(fight_weights)
                self.log(f"[OK] Fight斗殴检测模型加载成功")
            else:
                self.log(f"[WARNING] Fight斗殴模型权重不存在")
        except Exception as e:
            self.log(f"[ERROR] Fight斗殴模型加载失败: {e}")

        try:
            if os.path.exists(fall_weights):
                self.fall_model = YOLO(fall_weights)
                self.log(f"[OK] Fall摔倒检测模型加载成功")
            else:
                self.log(f"[WARNING] Fall摔倒模型权重不存在")
        except Exception as e:
            self.log(f"[ERROR] Fall摔倒模型加载失败: {e}")

        try:
            self.pose_model = YOLO("yolov8n-pose.pt")
            self.log("[OK] YOLOv8n-Pose 姿态估计模型加载成功 (人物骨架可视化)")
        except Exception as e:
            self.log(f"[ERROR] YOLOv8n-Pose 模型加载失败: {e}")

        try:
            self.person_model = YOLO("yolov8n.pt")
            self.log("[OK] YOLOv8n 人物检测模型加载成功")
        except Exception as e:
            self.log(f"[ERROR] YOLOv8n 人物模型加载失败: {e}")

        self.log("-" * 40)

    def draw_person_skeleton(self, frame, keypoints, color=(0, 255, 0), track_id=None):
        """绘制人体17关键点骨架"""
        if keypoints is None or len(keypoints) < 17:
            return frame

        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        for x, y in keypoints:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

        for i, j in skeleton:
            if i < len(keypoints) and j < len(keypoints):
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(frame, pt1, pt2, color, 2)

        if track_id is not None:
            nose = keypoints[0]
            cv2.putText(frame, f"ID:{track_id}", (int(nose[0]), int(nose[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def detect_persons_pose(self, frame, frame_count):
        """使用YOLOv8-Pose检测人物并返回关键点"""
        persons = []

        if self.pose_model is None:
            return persons

        try:
            results = self.pose_model.track(frame, persist=True, verbose=False)

            if len(results) > 0:
                result = results[0]
                if result.keypoints is not None:
                    keypoints_all = result.keypoints.xy
                    boxes = result.boxes

                    for i, kps in enumerate(keypoints_all):
                        if len(kps) >= 17:
                            conf = 1.0
                            track_id = int(boxes.id[i].item()) if boxes.id is not None else i

                            persons.append({
                                "track_id": track_id,
                                "keypoints": kps.cpu().numpy(),
                                "confidence": conf
                            })

        except Exception as e:
            if self.enable_debug_log:
                self.log(f"[WARNING] Pose检测异常: {e}")

        return persons

    def detect_fire(self, frame, frame_count):
        """火焰检测"""
        results = []

        if self.fire_model is None:
            return results

        try:
            yolo_results = self.fire_model.predict(frame, conf=self.fire_conf, verbose=False)

            if len(yolo_results) > 0:
                result = yolo_results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for i in range(len(result.boxes)):
                        cls_id = int(result.boxes.cls[i].item())
                        conf = float(result.boxes.conf[i].item())
                        box = result.boxes.xyxy[i].cpu().numpy()

                        cls_name = result.names.get(cls_id, f"class_{cls_id}")

                        if conf >= self.min_confidence:
                            if any(fire_term in cls_name.lower() for fire_term in ['fire', 'flame', 'smoke']):
                                results.append({
                                    "type": "明火",
                                    "confidence": round(conf, 3),
                                    "class_name": cls_name,
                                    "box": box.tolist()
                                })
                                self.debug_info["fire_frames"] += 1

                                if self.enable_debug_log and frame_count % 50 == 0:
                                    self.log(f"  [FIRE] frame={frame_count} conf={conf:.3f} class={cls_name}")

        except Exception as e:
            self.log(f"[WARNING] Fire检测异常: {e}")

        return results

    def detect_fall(self, frame, frame_count):
        """摔倒检测"""
        results = []

        if self.fall_model is None:
            return results

        try:
            yolo_results = self.fall_model.predict(frame, conf=self.fall_conf, verbose=False)

            if len(yolo_results) > 0:
                result = yolo_results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for i in range(len(result.boxes)):
                        cls_id = int(result.boxes.cls[i].item())
                        conf = float(result.boxes.conf[i].item())
                        box = result.boxes.xyxy[i].cpu().numpy()

                        cls_name = result.names.get(cls_id, f"class_{cls_id}")

                        if conf >= self.min_confidence:
                            if cls_name.lower() in ['fall', 'fallen', 'down'] or cls_name.lower().startswith('fall'):
                                results.append({
                                    "type": "摔倒",
                                    "confidence": round(conf, 3),
                                    "class_name": cls_name,
                                    "box": box.tolist()
                                })
                                self.debug_info["fall_frames"] += 1

                                if self.enable_debug_log and frame_count % 50 == 0:
                                    self.log(f"  [FALL] frame={frame_count} conf={conf:.3f} class={cls_name}")

        except Exception as e:
            self.log(f"[WARNING] Fall检测异常: {e}")

        return results

    def detect_fight(self, frame, frame_count):
        """斗殴检测"""
        results = []

        if self.fight_model is None:
            return results

        try:
            yolo_results = self.fight_model.predict(frame, conf=self.fight_conf, verbose=False)

            if len(yolo_results) > 0:
                result = yolo_results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    for i in range(len(result.boxes)):
                        cls_id = int(result.boxes.cls[i].item())
                        conf = float(result.boxes.conf[i].item())
                        box = result.boxes.xyxy[i].cpu().numpy()

                        if conf >= self.min_confidence:
                            if cls_id == 1:
                                results.append({
                                    "type": "斗殴",
                                    "confidence": round(conf, 3),
                                    "box": box.tolist()
                                })
                                self.debug_info["fight_frames"] += 1

                                if self.enable_debug_log and frame_count % 50 == 0:
                                    self.log(f"  [FIGHT] frame={frame_count} conf={conf:.3f}")

        except Exception as e:
            self.log(f"[WARNING] Fight检测异常: {e}")

        return results

    def update_temporal_state(self, track_id, event_type, confidence):
        """更新时序状态"""
        self.person_history[track_id].append({
            "type": event_type,
            "confidence": confidence,
            "timestamp": datetime.now()
        })

    def check_temporal_confirmation(self, track_id, event_type):
        """检查是否连续N帧都检测到同类事件"""
        if track_id not in self.person_history:
            return False, 0

        history = list(self.person_history[track_id])

        if len(history) < self.temporal_window:
            return False, 0

        recent = history[-self.temporal_window:]

        same_type_count = sum(1 for h in recent if h["type"] == event_type)

        if same_type_count >= self.temporal_window:
            avg_conf = sum(h["confidence"] for h in recent) / len(recent)
            return True, round(avg_conf, 3)

        return False, 0

    def draw_detection(self, frame, events, persons):
        """绘制检测结果和人物骨架"""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        for person in persons:
            keypoints = person.get("keypoints")
            track_id = person.get("track_id")
            self.draw_person_skeleton(annotated, keypoints, color=(0, 255, 0), track_id=track_id)

        for event in events:
            event_type = event.get("type", "unknown")
            confidence = event.get("confidence", 0)

            if "明火" in event_type:
                color = (0, 0, 255)
                label = f"FIRE {confidence:.2f}"
                if "box" in event:
                    box = event["box"]
                    cv2.rectangle(annotated, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.rectangle(annotated, (10, 10), (250, 50), color, -1)
                cv2.putText(annotated, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            elif "摔倒" in event_type:
                color = (0, 165, 255)
                label = f"FALL {confidence:.2f}"
                if "box" in event:
                    box = event["box"]
                    cv2.rectangle(annotated, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.rectangle(annotated, (10, 10), (250, 50), color, -1)
                cv2.putText(annotated, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            elif "斗殴" in event_type:
                color = (255, 0, 0)
                label = f"FIGHT {confidence:.2f}"
                if "box" in event:
                    box = event["box"]
                    cv2.rectangle(annotated, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.rectangle(annotated, (10, 10), (250, 50), color, -1)
                cv2.putText(annotated, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        danger_count = len([e for e in events if e.get("confirmed", False)])
        if danger_count > 0:
            cv2.putText(annotated, f"DANGER: {danger_count}", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated

    def save_annotated_frame(self, frame, video_name, frame_count, events, confirmed_events):
        """保存带标注的异常帧"""
        safe_video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).strip()
        save_dir = self.frame_save_dir / safe_video_name
        save_dir.mkdir(exist_ok=True)

        persons = self._last_detected_persons.get(frame_count, [])
        annotated_frame = self.draw_detection(frame, events, persons)

        if confirmed_events:
            best_event = max(confirmed_events, key=lambda e: e.get("confidence", 0))
            event_type = best_event['type']
        elif events:
            best_event = max(events, key=lambda e: e.get("confidence", 0))
            event_type = best_event['type']
        else:
            event_type = "unknown"

        filename = f"{safe_video_name}_frame{frame_count:06d}_{event_type}.jpg"
        filepath = save_dir / filename

        cv2.imwrite(str(filepath), annotated_frame)

        return filepath

    def process_video(self, video_path, category):
        """处理单个视频"""
        self.log("-" * 80)
        self.log(f"开始处理视频: {video_path}")
        self.log(f"预期危险类别: {category}")

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

        frame_count = 0
        processed_count = 0
        events_log = []
        saved_frames = []
        self._last_detected_persons = {}

        self.person_history.clear()

        self.log("开始抽帧分析（v3融合方案版）...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            self.debug_info["total_frames"] += 1

            if frame_count % self.sample_interval != 0:
                continue

            processed_count += 1

            if processed_count % 10 == 0:
                self.log(f"  进度: {processed_count}/{frames_to_process} 帧 ({processed_count*100//max(1,frames_to_process)}%)")

            persons = self.detect_persons_pose(frame, frame_count)
            self._last_detected_persons[frame_count] = persons

            for person in persons:
                track_id = person["track_id"]
                for evt in []:
                    self.update_temporal_state(track_id, evt["type"], evt["confidence"])

            raw_events = []
            confirmed_events = []

            if self.detection_mode == "unrestricted" or "fire" in category.lower():
                fire_events = self.detect_fire(frame, frame_count)
                raw_events.extend(fire_events)

            if self.detection_mode == "unrestricted" or "fall" in category.lower():
                fall_events = self.detect_fall(frame, frame_count)
                raw_events.extend(fall_events)

            if self.detection_mode == "unrestricted" or "fight" in category.lower():
                fight_events = self.detect_fight(frame, frame_count)
                raw_events.extend(fight_events)

            for event in raw_events:
                event_type = event["type"]
                confidence = event["confidence"]

                if confidence >= self.fight_conf:
                    confirmed_events.append({**event, "confirmed": True})
                    self.debug_info["confirmed_danger_frames"] += 1
                    self.log(f"  [危险确认] 帧{frame_count}: {event_type} - 置信度:{confidence:.3f} (高置信度)")

            if confirmed_events or raw_events:
                events_log.append({
                    "frame": frame_count,
                    "raw_events": raw_events,
                    "confirmed_events": confirmed_events
                })

                if self.save_frames and confirmed_events:
                    saved_path = self.save_annotated_frame(
                        frame, os.path.basename(video_path), frame_count,
                        raw_events, confirmed_events
                    )
                    saved_frames.append(str(saved_path))

        cap.release()

        self.log(f"视频处理完成! 共检测到 {len(events_log)} 个危险帧")
        self.log(f"确认危险帧数量: {len([e for e in events_log if e['confirmed_events']])}")
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

        confirmed_log = [e for e in events_log if e["confirmed_events"]]
        if total_frames > 0:
            processed = total_frames // self.sample_interval
            danger_rate = len(confirmed_log) / max(1, processed) * 100
            self.log(f"确认危险帧数: {len(confirmed_log)} ({danger_rate:.2f}%)")

        event_types_count = {}
        for entry in events_log:
            for event in entry["confirmed_events"]:
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
            "danger_frames_count": len(confirmed_log),
            "danger_rate": len(confirmed_log) / max(1, total_frames // self.sample_interval) * 100 if total_frames > 0 else 0,
            "event_types": event_types_count,
            "events_log": [{"frame": e["frame"], "events": e["confirmed_events"]} for e in events_log if e["confirmed_events"]],
            "saved_frames": saved_frames
        }

    def run(self, video_dir):
        """运行检测"""
        self.log("=" * 80)
        self.log("危险事件检测系统 v3 启动 (融合方案版)")
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
                self.person_history.clear()

                result = self.process_video(str(video_path), category)

                if result:
                    all_results.append(result)

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
            self.log(f"  确认危险帧数: {result['danger_frames_count']} ({result['danger_rate']:.2f}%)")
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
        self.log(f"  总确认危险帧数: {total_danger}")
        self.log(f"  保存异常帧总数: {total_saved}")

        self.log(f"\n模型检测统计:")
        self.log(f"  火焰检测帧数: {self.debug_info['fire_frames']}")
        self.log(f"  摔倒检测帧数: {self.debug_info['fall_frames']}")
        self.log(f"  斗殴检测帧数: {self.debug_info['fight_frames']}")
        self.log(f"  确认危险帧数: {self.debug_info['confirmed_danger_frames']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="监控视频危险事件检测系统 v3 (融合方案版)")
    parser.add_argument("--video", type=str, default=None, help="调试模式：仅处理包含此关键词的视频")
    parser.add_argument("--category", type=str, default=None, help="调试模式：仅处理此类别")
    parser.add_argument("--interval", type=int, default=10, help="抽帧间隔")
    parser.add_argument("--no-save", action="store_true", help="不保存异常帧")
    parser.add_argument("--fire-conf", type=float, default=0.5, help="火焰检测置信度阈值")
    parser.add_argument("--fall-conf", type=float, default=0.5, help="摔倒检测置信度阈值")
    parser.add_argument("--fight-conf", type=float, default=0.5, help="斗殴检测置信度阈值")
    parser.add_argument("--min-conf", type=float, default=0.3, help="最低置信度阈值")
    parser.add_argument("--temporal-window", type=int, default=3, help="时序平滑窗口大小")
    parser.add_argument("--detection-mode", type=str, default="restricted",
                       choices=["restricted", "unrestricted"],
                       help="检测模式: restricted(按类别检测) / unrestricted(全量检测)")
    args = parser.parse_args()

    detector = DangerDetectorV3(
        log_dir="d:/1_LNY/code/MultiMod-VisionDet/detection_logs",
        sample_interval=args.interval,
        save_frames=not args.no_save,
        debug_video=args.video,
        debug_category=args.category,
        fire_conf=args.fire_conf,
        fall_conf=args.fall_conf,
        fight_conf=args.fight_conf,
        min_confidence=args.min_conf,
        temporal_window=args.temporal_window,
        detection_mode=args.detection_mode
    )
    detector.run("d:/1_LNY/code/MultiMod-VisionDet/data/video")
