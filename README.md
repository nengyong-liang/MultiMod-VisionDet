# 监控视频危险事件检测系统 v4

基于预训练模型的多模态危险事件检测系统，支持**火焰检测**、**摔倒检测**和**斗殴检测**三种危险事件类型。

## 核心特性

### 🎯 检测能力
- **明火检测**：基于 YOLOv10 微调模型，识别火焰和烟雾
- **摔倒检测**：基于 YOLOv11 微调模型 + Pose 姿态估计，支持躺地持续报警
- **斗殴检测**：基于 YOLOv8 微调模型，识别暴力冲突行为

### 🔧 技术亮点
- **多模型融合**：集成 4 个预训练模型（Fire/Fall/Fight/Pose）
- **时序平滑机制**：连续 N 帧确认，减少瞬时误报
- **NMS 去重叠**：非极大值抑制消除重叠检测框
- **姿态可视化**：人体 17 关键点骨架绘制
- **置信度筛选**：多级阈值过滤，提升检测质量
- **持续报警**：摔倒后躺地状态持续跟踪报警

### 📊 检测模式
- **Restricted 模式**：按视频类别限制检测类型（适合演示）
- **Unrestricted 模式**：全量检测所有危险事件（实际应用）

---

## 快速开始

### 环境要求

```bash
Python >= 3.8
torch (CPU/GPU 版本均可)
ultralytics >= 8.0
opencv-python
numpy
```

### 安装依赖

```bash
pip install torch ultralytics opencv-python numpy
```

### 目录结构

```
run/
├── danger_detection_v4.py    # 主程序
├── models/                    # 预训练模型
│   ├── fire_detection_best.pt
│   ├── fight_violence_yolo_small.pt
│   ├── fall_detection_best.pt
│   └── yolov8n-pose.pt
├── data/video/               # 待检测视频（按类别分类）
│   ├── fire/
│   ├── fall/
│   └── fight/
└── logs/                     # 输出日志和结果
```

### 基本使用

```bash
# 进入 run 目录
cd run

# 检测所有视频
python danger_detection_v4.py

# 调试单个视频
python danger_detection_v4.py --video "fire1.mp4" --interval 10

# 指定类别检测
python danger_detection_v4.py --category "fall" --interval 5
```

---

## 参数说明

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--video` | str | None | 调试模式：仅处理包含此关键词的视频 |
| `--category` | str | None | 调试模式：仅处理此类别（fire/fall/fight） |
| `--interval` | int | 10 | 抽帧间隔（每 N 帧检测一次） |
| `--detection-mode` | str | restricted | 检测模式：restricted/unrestricted |

### 置信度阈值

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--fire-conf` | float | 0.4 | 火焰检测置信度阈值 (0-1) |
| `--fall-conf` | float | 0.4 | 摔倒检测置信度阈值 (0-1) |
| `--fight-conf` | float | 0.4 | 斗殴检测置信度阈值 (0-1) |
| `--min-conf` | float | 0.3 | 最低置信度阈值 (0-1) |

### 时序平滑

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--temporal-window` | int | 3 | 时序平滑窗口大小（连续 N 帧确认） |

### 摔倒检测优化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--fall-height-ratio` | float | 1.5 | 摔倒高度比例阈值（人体高/宽） |
| `--fall-duration` | int | 30 | 摔倒后持续报警帧数 |
| `--fall-save-interval` | int | 10 | 持续报警时保存间隔（每 N 帧） |

### 可视化优化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--person-kp-conf` | float | 0.5 | 人物关键点平均置信度阈值 |
| `--nms-iou` | float | 0.5 | NMS IoU 阈值（消除重叠框） |

### 输出控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--no-save` | flag | False | 不保存异常帧 |
| `--save-suspect` | flag | False | 保存疑似危险帧 |

---

## 使用示例

### 1. 火焰视频检测（高置信度）

```bash
python danger_detection_v4.py \
    --video "fire1.mp4" \
    --interval 10 \
    --fire-conf 0.35 \
    --nms-iou 0.5
```

### 2. 摔倒视频检测（持续报警）

```bash
python danger_detection_v4.py \
    --video "fall_02.mp4" \
    --interval 5 \
    --fall-conf 0.35 \
    --fall-duration 30 \
    --fall-save-interval 10
```

### 3. 斗殴视频检测（全量模式）

```bash
python danger_detection_v4.py \
    --video "fight_1234.mp4" \
    --interval 5 \
    --fight-conf 0.35 \
    --detection-mode unrestricted
```

### 4. 批量检测所有视频

```bash
python danger_detection_v4.py \
    --interval 10 \
    --save-suspect \
    --temporal-window 3
```

---

## 输出结果

### 日志文件

位置：`logs/detection_log_YYYYMMDD_HHMMSS.txt`

包含：
- 模型加载信息
- 每帧检测进度
- 危险事件确认记录
- NMS 优化统计

### JSON 结果

位置：`logs/detection_results_YYYYMMDD_HHMMSS.json`

```json
{
  "video_name": "fire1.mp4",
  "category": "fire",
  "total_frames": 1729,
  "confirmed_count": 107,
  "danger_rate": 62.21,
  "event_types": {
    "明火": 169
  },
  "saved_frames": [...]
}
```

### 异常帧图片

位置：`logs/detected_frames_YYYYMMDD_HHMMSS/`

命名规则：
- 确认帧：`confirmed_videof000120_FIRE.jpg`
- 疑似帧：`suspect_videof000120_FIRE.jpg`

标注内容：
- 检测框（红色=斗殴，橙色=摔倒，蓝色=火焰）
- 人体骨架（绿色 17 关键点）
- 置信度标签
- 危险计数

---

## 模型说明

### 预训练模型来源

| 模型 | 文件名 | 来源项目 | 类别 |
|------|--------|---------|------|
| **Fire** | fire_detection_best.pt | Fire-Detection-model-main | fire, smoke |
| **Fall** | fall_detection_best.pt | Real-Time-Fall-Detection-using-YOLO | fall, fallen, down |
| **Fight** | fight_violence_yolo_small.pt | Fight-Violence-detection-yolov8 | class 0=NoViolence, class 1=Violence |
| **Pose** | yolov8n-pose.pt | Ultralytics YOLOv8 | 17 人体关键点 |

### 模型权重

- Fire 模型：~5.3 MB
- Fall 模型：~15.3 MB
- Fight 模型：~21.5 MB
- Pose 模型：~6.5 MB

---

## 性能优化建议

### 1. 提升检测速度
```bash
# 增大抽帧间隔（牺牲精度换速度）
--interval 20

# 降低置信度阈值（可能增加误报）
--fire-conf 0.3 --fall-conf 0.3 --fight-conf 0.3
```

### 2. 减少误报
```bash
# 提高置信度阈值
--fire-conf 0.5 --fall-conf 0.5 --fight-conf 0.5

# 增大时序窗口
--temporal-window 5

# 提高关键点筛选标准
--person-kp-conf 0.6
```

### 3. 提升召回率
```bash
# 降低 NMS 阈值（保留更多检测框）
--nms-iou 0.3

# 降低最低置信度
--min-conf 0.2

# 启用疑似帧保存
--save-suspect
```

---

## 常见问题

### Q1: 为什么摔倒检测帧数很少？
A: 默认设置需要连续 3 帧确认，且摔倒后需持续 30 帧才开始保存。可通过以下参数调整：
```bash
--temporal-window 2 --fall-duration 15
```

### Q2: 检测框重叠严重？
A: 调整 NMS 阈值：
```bash
--nms-iou 0.3  # 更严格的重叠过滤
```

### Q3: 如何检测视频中的多种危险？
A: 使用 unrestricted 模式：
```bash
--detection-mode unrestricted
```

### Q4: 人物骨架显示异常？
A: 提高关键点置信度阈值：
```bash
--person-kp-conf 0.6
```

---

## 项目结构

```
MultiMod-VisionDet/
├── run/                          # 可分发版本
│   ├── danger_detection_v4.py
│   ├── models/
│   └── logs/
├── model/                        # 原始模型文件
│   ├── Fire-Detection-model-main/
│   ├── Fight-Violence-detection-yolov8-main/
│   └── Real-Time-Fall-Detection-using-YOLO-main/
├── data/video/                   # 测试视频
├── danger_detection.py           # 传统方法版本
├── danger_detection_v2.py        # 预训练模型版本
├── danger_detection_v3.py        # 融合方案版本
└── danger_detection_v4.py        # 当前版本（融合优化）
```

---

## 版本历史

### v4 (当前版本)
- ✅ 修复时序平滑逻辑空实现
- ✅ 添加 NMS 非极大值抑制
- ✅ 实现摔倒持续报警机制
- ✅ 优化可视化标签布局
- ✅ 添加参数合法性校验

### v3
- ✅ 融合预训练模型 + 时序平滑
- ✅ 人物骨架可视化
- ✅ 多阈值策略

### v2
- ✅ 全面使用预训练模型
- ✅ 支持 restricted/unrestricted 模式

### v1
- ✅ 传统方法实现（HSV 火焰检测 + Pose 摔倒检测）

---

## 许可证

本项目仅供学术研究和教学使用。

---

## 联系方式

如有问题或建议，请提交 Issue 或联系开发者。

---

## 致谢

感谢以下开源项目：
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Fire-Detection-model-main](https://github.com/...)
- [Fight-Violence-detection-yolov8](https://github.com/...)
- [Real-Time-Fall-Detection-using-YOLO](https://github.com/...)
