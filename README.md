# BetterNTE

更好的异环 — 异环（Neverness to Everness）自动化辅助工具，对标 [ok-wuthering-waves](https://github.com/ok-oldking/ok-wuthering-waves)。

基于纯视觉识别的异环自动化辅助，灵感来自 OK-WW（鸣潮），旨在为异环提供同等甚至更好的自动化体验。

## 当前功能

- **HSV 颜色阈值检测**：绿条、黄点、蓝圈三路并行识别
- **水平投影法**：精确计算绿条几何中心，不受边缘不规则影响
- **PD 控制器**：比例-微分双通道 A/D 操控，力度可调
- **状态机驱动**：IDLE → CASTING → WAITING_BITE → HOOKING → CONTROLLING → FINISHED 全自动循环
- **PySide6 GUI**：截图上传、ROI 框选、实时预览、检测标注
- **三重安全停止**：F8 全局热键 / 鼠标角落急停 / GUI 停止按钮

## 计划功能

对标 OK-WW，逐步扩展异环的自动化能力：

- **模板匹配**：引入 OpenCV `matchTemplate` + COCO 特征标注，识别 UI 元素
- **目标检测**：YOLOv8 ONNX 推理，用于复杂场景的目标识别
- **任务调度系统**：TriggerTask（持续监听）+ OneTimeTask（手动触发），支持多任务并行
- **后台交互**：PostMessage 窗口消息模式，游戏无需置顶
- **自动战斗**：角色技能循环、目标锁定、走位
- **自动拾取**：掉落物检测与自动拾取
- **自动日常**：日常任务、副本自动化

## 环境要求

- Windows 10/11
- Python 3.11+
- **管理员权限**（DirectInput 模拟键盘需要绕过 UIPI）

## 安装

```bash
git clone https://github.com/HP26666/BetterNTE.git
cd BetterNTE
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## 运行

**管理员模式启动**（必须）：

```bash
run_admin.bat
```

或手动：

```bash
.venv\Scripts\activate
python -m fishpp.main
```

## 使用流程

1. 启动程序，确认右上角显示绿色「管理员」
2. 新建配置或加载已有配置
3. 截屏（5 秒延时）或上传游戏截图
4. 框选绿条区域 + 蓝圈区域
5. 点击「开始运行」，切到游戏窗口

详细教程见 [tutorial.html](tutorial.html)（程序内也可点击查看）。

## 项目结构

```
BetterNTE/
├── src/fishpp/
│   ├── main.py              # 入口
│   ├── models.py            # 数据模型（AppConfig, Observation, ROI...）
│   ├── capture.py           # 屏幕截图（mss）
│   ├── vision.py            # 视觉检测（HSV 阈值 + 轮廓 + 水平投影）
│   ├── state_machine.py     # 钓鱼状态机
│   ├── control.py           # 输入控制 + PD 脉冲计算
│   ├── worker.py            # 识别工作线程
│   ├── config.py            # 配置文件读写
│   └── gui/
│       ├── main_window.py   # 主窗口 UI
│       └── annotation.py    # ROI 框选 + 取色对话框
├── configs/                 # 用户配置文件（自动生成，不入库）
├── tutorial.html            # 使用教程 + 算法详解
├── run_admin.bat            # 管理员启动脚本
└── pyproject.toml           # 项目配置
```

## 致谢

- [ok-wuthering-waves](https://github.com/ok-oldking/ok-wuthering-waves) — 鸣潮自动化辅助工具，本项目的设计参考

## 依赖

| 包 | 用途 |
|---|---|
| PySide6 >= 6.7 | GUI 框架 |
| mss >= 10.0.0 | 全屏截图 |
| opencv-python >= 4.10 | 图像处理 + HSV 检测 |
| numpy >= 2.0.0 | 数值计算 |
| pydirectinput >= 1.0.4 | DirectInput 键盘模拟 |
| pynput | F8 全局热键（可选） |

## License

[Apache License 2.0](LICENSE)

## Author

**氕氘氚** — [GitHub: HP26666](https://github.com/HP26666)
