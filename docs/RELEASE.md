# Windows 发行候选版 0.1.0-rc1

## 启动

将整个文件夹放到本地可读目录，双击 `AIResearchAssistant.exe`。
不要只复制 EXE，不要删除 `_internal`。
模型放在 EXE 旁的 `models/`，引擎放在 `runtime/`。
无需安装 Python；这是发行候选版，不是经过多台干净电脑验证的正式版。

初始材料库为空，导入自己的 TXT、MD、DOCX 或 XLSX。
同一 Windows 用户若已使用开发版，会恢复既有材料库。
资料和索引保存在用户数据目录的 `AIResearchAssistant` 文件夹，而非发行目录。
API 设置需填写自己的密钥，密钥只在当前会话内存，不随发行包提供。

默认检索模型为 BGE-M3。本地回答可选 Qwen3；无 NVIDIA 显卡时选择 CPU，
CPU 首次加载和回答可能较慢。CUDA 包需要兼容的 NVIDIA 驱动，
仅在开发机实测，不能保证所有设备可用。关闭程序会停止它自己启动的推理进程。
模型和库占用多 GB 磁盘；不包含大规模测试资料。

## 包结构

- `AIResearchAssistant.exe`：唯一启动入口。
- `_internal/`：Python/Qt/检索依赖，必须完整保留。
- `models/bge-m3/`：检索模型；E5 对照模型不附带。
- `models/qwen3-4b-gguf/`：离线回答模型。
- `runtime/llama-bin/`：CPU 推理运行包。
- `runtime/llama-cuda13/`：可选 CUDA 推理运行包。

模型、引擎属于第三方资源。发布到 GitHub 前必须核对下载来源、
模型许可证、量化模型的来源和各二进制运行库的再分发条款；
本地候选版打包完成不表示完成所有第三方许可审查。
不要把 `.env`、私人文档、材料库、索引或整个开发目录上传。
候选版未签名，Windows 可能提示未知发布者。

本机 EXE 隔离自检已通过：移除 Anaconda PATH 后，合成 Word/Excel 导入、
公式显示、BGE-M3 CPU 检索和 Qwen3 CPU 回答正常，未调用 API。
打包配置排除了错误从 Poppler 收集的同名 ICU DLL，Qt 使用 Windows 系统 ICU。
这不是干净独立电脑的兼容性验收，付费 API 与冻结版 GPU 路径尚未实测。

## 开发者重现

在项目根目录用已验证的环境执行：

```powershell
python -m PyInstaller packaging/assistant.spec --distpath release/0.1.0-rc1 --workpath build/pyinstaller --noconfirm
python -m tools.release --assemble
```

在隔离合成材料上通过 EXE 自检后，运行 `python -m tools.release --archive`，
生成完整 Windows ZIP 和 SHA256 校验文件。正式公开发布前仍需独立电脑验收。

`requirements.txt` 记录开发环境版本，并非全依赖 lockfile；
新环境安装与 CPU/CUDA wheel 选择仍需验证。
源码 ZIP 采用白名单，不附带模型、密钥、材料库或压力数据。
可执行目录可整体复制；模型超过常见仓库单文件限制，不应提交进源码仓库。
