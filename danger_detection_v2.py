"""
监控视频危险事件检测系统 v7 (预训练模型版)
全面使用预训练模型替代传统方法

三个检测模型:
- Fire: YOLOv10微调模型 (本地权重 best.pt)
- Fall: YOLOv11微调模型 (本地权重 best.pt)
- Fight: YOLOv8微调模型 (本地权重 yolo_small_weights.pt)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import json
from datetime import datetime
from pathlib import Path
from collections import deque


class DangerDetectorV2:
    def __init__(self, log_dir="detection_logs", sample_interval=10, save_frames=True,
                 debug_video=None, debug_category=None,
                 fire_conf=0.25, fall_conf=0.4, fight_conf=0.4,
                 enable_debug_log=True, detection_mode="unrestricted"):
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
        self.enable_debug_log = enable_debug_log
        self.detection_mode = detection_mode

        self.fire_model = None
        self.fall_model = None
        self.fight_model = None
        self.person_model = None

        self.person_tracking = {}
        self.fall_history = {}

        self.debug_info = {
            "fire_frames": 0,
            "fall_frames": 0,
            "fight_frames": 0,
            "total_frames": 0
        }

        self.log("=" * 80)
        self.log("监控视频危险事件检测系统 v7 初始化 (预训练模型版)")
        mode_name = "全量检测模式" if detection_mode == "unrestricted" else "类别限制模式"
        self.log(f"检测模式: {mode_name} ({detection_mode})")
        self.log(f"抽帧间隔: 每 {self.sample_interval} 帧检测一次")
        self.log(f"保存异常帧: {'是' if self.save_frames else '否'}")
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
                self.log(f"[OK] Fire火焰检测模型加载成功: {fire_weights}")
            else:
                self.log(f"[WARNING] Fire火焰模型权重不存在: {fire_weights}")
        except Exception as e:
            self.log(f"[ERROR] Fire火焰模型加载失败: {e}")

        try:
            if os.path.exists(fight_weights):
                self.fight_model = YOLO(fight_weights)
                self.log(f"[OK] Fight斗殴检测模型加载成功: {fight_weights}")
            else:
                self.log(f"[WARNING] Fight斗殴模型权重不存在: {fight_weights}")
        except Exception as e:
            self.log(f"[ERROR] Fight斗殴模型加载失败: {e}")

        try:
            if os.path.exists(fall_weights):
                self.fall_model = YOLO(fall_weights)
                self.log(f"[OK] Fall摔倒检测模型加载成功: {fall_weights}")
            else:
                self.log(f"[WARNING] Fall摔倒模型权重不存在: {fall_weights}")
        except Exception as e:
            self.log(f"[ERROR] Fall摔倒模型加载失败: {e}")

        try:
            self.person_model = YOLO("yolov8n.pt")
            self.log("[OK] YOLOv8n 通用检测模型加载成功 (用于辅助检测)")
        except Exception as e:
            self.log(f"[ERROR] YOLOv8n 通用模型加载失败: {e}")

        self.log("-" * 40)

    def detect_fire_yolo(self, frame, frame_count):
        """使用YOLOv10检测火焰/烟雾"""
        results = []

        if self.fire_model is None:
            return results

        try:
            yolo_results = self.fire_model.predict(frame, conf=self.fire_conf, verbose=False)

            if len(yolo_results) > 0:
                result = yolo_results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    fire_count = 0
                    fire_confidence = 0

                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())

                        cls_name = result.names.get(cls_id, f"class_{cls_id}")
                        if any(fire_term in cls_name.lower() for fire_term in ['fire', 'flame', 'smoke']):
                            fire_count += 1
                            fire_confidence = max(fire_confidence, conf)

                    if fire_count > 0:
                        avg_conf = fire_confidence
                        results.append({
                            "type": "明火",
                            "confidence": round(avg_conf, 3),
                            "fire_count": fire_count,
                            "class_name": cls_name
                        })
                        self.debug_info["fire_frames"] += 1

                        if self.enable_debug_log and frame_count % 30 == 0:
                            self.log(f"  [FIRE-YOLO] frame={frame_count} count={fire_count} conf={avg_conf:.3f}")

        except Exception as e:
            self.log(f"[WARNING] Fire YOLO检测异常: {e}")

        return results

    def detect_fire_hsv(self, frame, frame_count):
        """HSV颜色空间火焰检测（备用方案）"""
        results = []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_fire1 = np.array([0, 120, 100])
        upper_fire1 = np.array([15, 255, 255])
        fire_mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)

        lower_fire2 = np.array([165, 120, 100])
        upper_fire2 = np.array([180, 255, 255])
        fire_mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)

        fire_mask = fire_mask1 | fire_mask2

        kernel = np.ones((5, 5), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

        fire_pixels = cv2.countNonZero(fire_mask)
        h, w = frame.shape[:2]
        fire_ratio = fire_pixels / (w * h)

        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_regions = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                large_regions += 1

        if fire_ratio > 0.01 and large_regions > 0:
            confidence = min(fire_ratio * 50, 0.9)
            results.append({
                "type": "明火",
                "confidence": round(confidence, 3),
                "fire_ratio": round(fire_ratio, 4),
                "regions": large_regions,
                "method": "hsv"
            })
            self.debug_info["fire_frames"] += 1

            if self.enable_debug_log and frame_count % 30 == 0:
                self.log(f"  [FIRE-HSV] frame={frame_count} ratio={fire_ratio:.4f} regions={large_regions}")

        return results

    def detect_fall_yolo(self, frame, frame_count):
        """使用YOLOv11微调模型检测摔倒"""
        results = []

        if self.fall_model is None:
            return results

        try:
            yolo_results = self.fall_model.predict(frame, conf=self.fall_conf, verbose=False)

            if len(yolo_results) > 0:
                result = yolo_results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        box = boxes.xyxy[i].cpu().numpy()

                        cls_name = result.names.get(cls_id, f"class_{cls_id}")

                        if cls_name.lower() in ['fall', 'fallen', 'down'] or cls_name.lower().startswith('fall'):
                            results.append({
                                "type": "摔倒",
                                "confidence": round(conf, 3),
                                "class_name": cls_name,
                                "box": box.tolist()
                            })
                            self.debug_info["fall_frames"] += 1

                            if self.enable_debug_log and frame_count % 30 == 0:
                                self.log(f"  [FALL-YOLO] frame={frame_count} conf={conf:.3f} class={cls_name}")

        except Exception as e:
            self.log(f"[WARNING] Fall YOLO检测异常: {e}")

        return results

    def detect_fight_yolo(self, frame, frame_count):
        """使用YOLOv8微调模型检测斗殴"""
        results = []

        if self.fight_model is None:
            return results

        try:
            yolo_results = self.fight_model.predict(frame, conf=self.fight_conf, verbose=False)

            if len(yolo_results) > 0:
                result = yolo_results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        box = boxes.xyxy[i].cpu().numpy()

                        cls_name = result.names.get(cls_id, f"class_{cls_id}")

                        if cls_id == 1:
                            results.append({
                                "type": "斗殴",
                                "confidence": round(conf, 3),
                                "class_name": cls_name,
                                "box": box.tolist()
                            })
                            self.debug_info["fight_frames"] += 1

                            if self.enable_debug_log and frame_count % 30 == 0:
                                self.log(f"  [FIGHT-YOLO] frame={frame_count} conf={conf:.3f} class={cls_name}")

        except Exception as e:
            self.log(f"[WARNING] Fight YOLO检测异常: {e}")

        return results

    def detect_persons_yolo(self, frame, frame_count):
        """使用YOLOv8检测人物（用于辅助检测）"""
        persons = []

        if self.person_model is None:
            return persons

        try:
            yolo_results = self.person_model.predict(frame, conf=0.5, classes=[0], verbose=False)

            if len(yolo_results) > 0:
                result = yolo_results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    for i in range(len(boxes)):
                        conf = float(boxes.conf[i].item())
                        box = boxes.xyxy[i].cpu().numpy()
                        track_id = int(boxes.id[i].item()) if boxes.id is not None else i

                        persons.append({
                            "track_id": track_id,
                            "confidence": conf,
                            "box": box.tolist()
                        })

        except Exception as e:
            if self.enable_debug_log:
                self.log(f"[WARNING] Person检测异常: {e}")

        return persons

    def draw_detection(self, frame, events):
        """绘制检测结果"""
        annotated = frame.copy()

        for event in events:
            event_type = event.get("type", "unknown")
            confidence = event.get("confidence", 0)

            if "明火" in event_type or "火灾" in event_type:
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

            elif "斗殴" in event_type or "冲突" in event_type:
                color = (255, 0, 0)
                label = f"FIGHT {confidence:.2f}"
                if "box" in event:
                    box = event["box"]
                    cv2.rectangle(annotated, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.rectangle(annotated, (10, 10), (250, 50), color, -1)
                cv2.putText(annotated, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return annotated

    def save_annotated_frame(self, frame, video_name, frame_count, events):
        """保存带标注的异常帧"""
        safe_video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).strip()
        save_dir = self.frame_save_dir / safe_video_name
        save_dir.mkdir(exist_ok=True)

        annotated_frame = self.draw_detection(frame, events)

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

        self.log("开始抽帧分析（v7预训练模型版）...")

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

            all_events = []

            if self.detection_mode == "unrestricted" or "fire" in category.lower():
                fire_events = self.detect_fire_yolo(frame, frame_count)
                if not fire_events:
                    fire_events = self.detect_fire_hsv(frame, frame_count)

                if fire_events:
                    all_events.extend(fire_events)
                    for event in fire_events:
                        self.log(f"  [危险检测] 帧{frame_count}: {event['type']} - 置信度:{event['confidence']:.3f}")

            if self.detection_mode == "unrestricted" or "fall" in category.lower():
                fall_events = self.detect_fall_yolo(frame, frame_count)
                if fall_events:
                    all_events.extend(fall_events)
                    for event in fall_events:
                        self.log(f"  [危险检测] 帧{frame_count}: {event['type']} - 置信度:{event['confidence']:.3f}")

            if self.detection_mode == "unrestricted" or "fight" in category.lower():
                fight_events = self.detect_fight_yolo(frame, frame_count)
                if fight_events:
                    all_events.extend(fight_events)
                    for event in fight_events:
                        self.log(f"  [危险检测] 帧{frame_count}: {event['type']} - 置信度:{event['confidence']:.3f}")

            if all_events:
                events_log.append({
                    "frame": frame_count,
                    "events": all_events
                })

                if self.save_frames:
                    saved_path = self.save_annotated_frame(
                        frame, os.path.basename(video_path), frame_count, all_events
                    )
                    saved_frames.append(str(saved_path))

        cap.release()

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
        self.log("危险事件检测系统 v7 启动 (预训练模型版)")
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
                self.person_tracking.clear()
                self.fall_history.clear()

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

        self.log(f"\n模型检测统计:")
        self.log(f"  火焰检测帧数: {self.debug_info['fire_frames']}")
        self.log(f"  摔倒检测帧数: {self.debug_info['fall_frames']}")
        self.log(f"  斗殴检测帧数: {self.debug_info['fight_frames']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="监控视频危险事件检测系统 v7 (预训练模型版)")
    parser.add_argument("--video", type=str, default=None, help="调试模式：仅处理包含此关键词的视频")
    parser.add_argument("--category", type=str, default=None, help="调试模式：仅处理此类别")
    parser.add_argument("--interval", type=int, default=10, help="抽帧间隔")
    parser.add_argument("--no-save", action="store_true", help="不保存异常帧")
    parser.add_argument("--fire-conf", type=float, default=0.25, help="火焰检测置信度阈值")
    parser.add_argument("--fall-conf", type=float, default=0.4, help="摔倒检测置信度阈值")
    parser.add_argument("--fight-conf", type=float, default=0.4, help="斗殴检测置信度阈值")
    parser.add_argument("--detection-mode", type=str, default="unrestricted",
                       choices=["restricted", "unrestricted"],
                       help="检测模式: restricted(按类别检测) / unrestricted(全量检测，默认)")
    args = parser.parse_args()

    detector = DangerDetectorV2(
        log_dir="d:/1_LNY/code/MultiMod-VisionDet/detection_logs",
        sample_interval=args.interval,
        save_frames=not args.no_save,
        debug_video=args.video,
        debug_category=args.category,
        fire_conf=args.fire_conf,
        fall_conf=args.fall_conf,
        fight_conf=args.fight_conf,
        detection_mode=args.detection_mode
    )
    detector.run("d:/1_LNY/code/MultiMod-VisionDet/data/video")
