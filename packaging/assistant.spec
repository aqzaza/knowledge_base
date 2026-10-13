from PyInstaller.utils.hooks import collect_all, copy_metadata
from pathlib import Path

root = Path(SPECPATH).parent

datas, binaries, hiddenimports = [], [], []
for package in ("sentence_transformers", "transformers", "langchain_deepseek"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h
for package in ("sentence-transformers", "transformers", "langchain-text-splitters",
                "langchain-core", "langchain-deepseek"):
    datas += copy_metadata(package, recursive=True)
hiddenimports += ["docx", "openpyxl", "dotenv", "matplotlib.backends.backend_agg",
                  "tools.release"]
a = Analysis(
    [str(root / "desktop_app.py")], pathex=[str(root)], binaries=binaries, datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tensorflow", "tkinter", "IPython", "pytest", "torchvision", "torchaudio"],
    hookspath=[], runtime_hooks=[], noarchive=False,
)
# Qt uses Windows' unversioned ICU API. Never bundle Poppler's same-named ICU.
a.binaries = [entry for entry in a.binaries
              if Path(entry[0]).name.lower() not in {"icuuc.dll", "icudt78.dll"}]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name="AIResearchAssistant", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name="AIResearchAssistant")
