一、需求处理
用户提出任何要求，助手先分析问题、商榷点、补充项并与用户确认
获得用户确认后，助手方可开始开发

二、Python 运行环境
每次运行代码前，助手先确认当前虚拟环境：EVAC-MIND / EVAC-MIND-1 / 其他
按环境匹配对应路径：
EVAC-MIND → C:\ProgramData\anaconda3\envs\EVAC-MIND
EVAC-MIND-1 → C:\ProgramData\anaconda3\envs\EVAC-MIND-1

三、硬件适配
所有虚拟环境仅安装 CPU 版 PyTorch，无 GPU 支持；如需 GPU，需重新安装带 CUDA 的 torch

四、包安装
用户需要安装库时，助手根据当前环境依赖兼容性，推荐适配版本