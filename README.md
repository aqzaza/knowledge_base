# AI 技术资料研究助手

Windows 桌面程序：导入 TXT、Markdown、Word（DOCX）和 Excel（XLSX），
检索材料并查看来源；可选择本地 Qwen3 回答或使用自己的 DeepSeek API 配置。
本地回答是固定 RAG，DeepSeek 模式支持受限的文档工具调用。
材料管理支持递归导入文件夹和不同目录的同名文件，最多 500 份、总计 20 MiB；
单文件仍最多 1 MiB。这是容量边界，不是所有规模下的性能保证。

## 当前启动方式

在已配置好的本机环境中运行：

```powershell
& 'D:\Anaconda\envs\langchain1.2\python.exe' 'D:\Postgraduade_study\项目1\desktop_app.py'
```

以上路径是开发机路径。发行候选版位于
`release/0.1.0-rc1/AIResearchAssistant/`，双击其中的 `AIResearchAssistant.exe`。
必须保留整个目录，不能只复制 EXE。打包与使用细节见 `docs/RELEASE.md`；
候选版仍需干净 Windows 机器验收和第三方许可证审核后正式公开发布。
`requirements.txt` 记录已验证的开发环境直接依赖，不是全量锁文件。
公式阅读视图的额外依赖记录在 `requirements-desktop.txt`，不是完整发布依赖清单。
阅读视图离线渲染常见数学公式；不支持的复杂 LaTeX 保留源码，
“原始文本”始终保留导入内容不变。
模型和 llama.cpp 运行包单独放在 `models/` 和 `runtime/`，不随源码提交。
DeepSeek 密钥由使用者在桌面中填写，仅保存在当前会话内存。
不要发布 `.env` 或私人材料。

## 文件组织

| 位置 | 用途 |
| --- | --- |
| `desktop_app.py` | 唯一桌面启动入口 |
| `read_document.py`、`split_documents.py` | 文本和 Office 读取、分块 |
| `embed_query.py`、`index_cache.py` | 检索、引用检查、向量缓存；保留 CLI 验证入口 |
| `document_tools.py`、`local_answer.py` | 文档工具、DeepSeek Agent、本地模型会话 |
| `material_store.py` | 用户材料持久保存 |
| `app_paths.py` | 开发与 EXE 模式下的外部模型路径 |
| `packaging/`、`tools/` | EXE 打包配置与发行自检/白名单组装 |
| `build/`、`release/` | 被 Git 忽略的构建文件和本地发行候选版 |
| `tests/` | 离线自动化测试 |
| `docs/` | API、Office、设备、本地回答、材料保存说明 |
| `data/` | 开发和导入样例；`import_qa_test.txt` 被桌面测试引用 |
| `corpus/`、`eval/`、`compare_retrieval.py` | 早期检索对照资料、来源许可证、开发题与对照脚本 |
| `models/`、`runtime/` | 本地权重和运行库，保持现有路径 |
| `PROJECT_MENTOR_GUIDE.md`、`PROJECT_STATUS.md` | 协作约定与真实进度记录 |

桌面只使用用户导入的材料，不自动导入 `data/` 或 `corpus/`。
历史状态中的旧命令和文件名只用于记录当时的操作，不是当前启动说明。

## 离线测试

在项目根目录执行，不调用付费 API、不加载真实大模型：

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'D:\Anaconda\envs\langchain1.2\python.exe' -m unittest discover -s tests -v
```

## 使用说明

- [本地回答与 CPU/GPU](docs/LOCAL_ANSWER.md)
- [Embedding 设备兼容](docs/DEVICE_SUPPORT.md)
- [API 设置与隐私](docs/API_SETTINGS.md)
- [Word 和 Excel 读取边界](docs/OFFICE_IMPORT.md)
- [材料保存与缓存](docs/MATERIAL_STORAGE.md)
- [140 份中文资料的容量测试与已知问题](docs/CAPACITY_TEST.md)
- [Windows 发行候选版](docs/RELEASE.md)

引用编号有效不代表证据一定支持结论。检索问答不保证精确表格计算，
也不保证复杂材料的总体准确率。
