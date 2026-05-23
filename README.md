# TorchAudio Backport for Python 3.8 (CUDA 11.3)

[中文](#中文) | [Русский](#русский) | **English**

A backported and modified version of [torchaudio](https://github.com/pytorch/audio) that supports **Python 3.8** and **CUDA 11.3**, based on the torchaudio main branch (v2.11.0a0).

## What Is This?

This is a modified build of torchaudio designed for users who still need **Python 3.8** compatibility. The official PyTorch ecosystem dropped Python 3.8 support starting from PyTorch 2.5+, but many legacy environments and projects still rely on Python 3.8. This project bridges that gap by backporting the latest torchaudio features to Python 3.8 with CUDA 11.3 support.

This repository is part of the **PyTorch Python 3.8 Backport Suite**, which includes:
- [pytorch_backport_py38](https://github.com/Lanurence666/pytorch_backport_py38) — Backported PyTorch 2.13.0a0+cu113 for Python 3.8
- **audio_backport_py38** (this repo) — Backported torchaudio 2.11.0a0+cu113 for Python 3.8

## Modifications & Fixes

### 1. Soundfile Fallback Backend

**Problem:** The upstream torchaudio (v2.9+) hard-depends on `torchcodec` for all audio I/O operations (`load()` and `save()`). However, `torchcodec` requires FFmpeg and does not provide pre-built wheels for Python 3.8 on Windows, making it impossible to install on Python 3.8 environments.

**Fix:** We implemented an automatic fallback mechanism in `torchaudio/__init__.py`:

- Added `_TORCHCODEC_AVAILABLE` detection flag that properly checks both the `_torchcodec` C extension and the `torchcodec` package availability
- Implemented `_load_with_soundfile()` — a soundfile-based audio loading function that supports `frame_offset`, `num_frames`, `normalize`, and `channels_first` parameters
- Implemented `_save_with_soundfile()` — a soundfile-based audio saving function with proper channel handling
- Modified `load()` and `save()` to automatically fall back to soundfile when torchcodec is unavailable, with a `UserWarning` to inform the user

**Key code change:**
```python
_TORCHCODEC_AVAILABLE = False
try:
    from ._torchcodec import load_with_torchcodec, save_with_torchcodec
    import torchcodec
    _TORCHCODEC_AVAILABLE = True
except ImportError:
    pass

def load(...):
    if _TORCHCODEC_AVAILABLE:
        return load_with_torchcodec(...)
    else:
        warnings.warn("torchcodec is not available. Falling back to soundfile backend.")
        return _load_with_soundfile(...)
```

### 2. TorchCodec Import Detection Fix

**Problem:** The original code only imported `from ._torchcodec import ...` without checking if the `torchcodec` package itself was available. This could cause `ImportError: TorchCodec is required for load_with_torchcodec` at runtime even when the C extension was present.

**Fix:** Added `import torchcodec` to the try block to ensure both the C extension and the Python package are available before setting `_TORCHCODEC_AVAILABLE = True`.

## Key Features

| Feature | Status |
|---------|--------|
| Python 3.8 support | ✅ |
| CUDA 11.3 support | ✅ |
| Audio load/save with soundfile fallback | ✅ |
| Audio resampling (`torchaudio.transforms.Resample`) | ✅ |
| Spectrogram / MelSpectrogram / MFCC | ✅ |
| Kaldi compliance (Fbank, Mfcc, etc.) | ✅ |
| Pitch detection | ✅ |
| All torchaudio transforms | ✅ |
| All torchaudio models (wav2vec2, conformer, etc.) | ✅ |
| TorchCodec backend (when available) | ✅ |
| Soundfile fallback backend | ✅ |

## Debugging & Test Results

We performed comprehensive testing on Windows x64 with Python 3.8 + CUDA 11.3:

| Test | Result |
|------|--------|
| `torchaudio.load()` with WAV file | ✅ Pass |
| `torchaudio.save()` and reload roundtrip | ✅ Pass |
| `torchaudio.transforms.Resample` | ✅ Pass |
| `torchaudio.transforms.MFCC` | ✅ Pass |
| `torchaudio.transforms.Spectrogram` | ✅ Pass |
| `torchaudio.transforms.MelSpectrogram` | ✅ Pass |
| `torchaudio.transforms.PitchShift` | ✅ Pass |
| `torchaudio.functional.detect_pitch_language` | ✅ Pass |
| Soundfile fallback (without torchcodec) | ✅ Pass |
| TorchCodec backend (when installed) | ✅ Pass |
| C extension (`_torchaudio.pyd`) | ✅ Pass |

**Known limitations:**
- Soundfile fallback only supports WAV/FLAC/OGG formats (no MP3 encoding via soundfile)
- `torch.compile()` is not fully supported on this Python 3.8 build
- NumPy 2.x compatibility warning may appear (recommend `numpy<2`)

## Requirements

- Python 3.8
- PyTorch 2.13.0a0+cu113 (from [pytorch_backport_py38](https://github.com/Lanurence666/pytorch_backport_py38))
- soundfile (`pip install soundfile`)
- numpy < 2

## Installation

### Option 1: Install from Release Wheel (Recommended)

Download the wheel from the [GitHub Releases](https://github.com/Lanurence666/audio_backport_py38/releases) page and install:

```bash
pip install torchaudio-2.11.0a0+cu113-cp38-cp38-win_amd64.whl
```

### Option 2: Build from Source

**Prerequisites:**
- Python 3.8 with PyTorch 2.13.0a0+cu113 installed
- Visual Studio 2019+ with C++ build tools
- CMake 3.20+
- Ninja build system

**Build steps:**

```bash
# Clone this repository
git clone https://github.com/Lanurence666/audio_backport_py38.git
cd audio_backport_py38

# Install build dependencies
pip install soundfile numpy

# Build and install
python setup.py install

# Or build a wheel
python setup.py bdist_wheel
```

**Environment variables for custom builds:**
```bash
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3
set TORCH_CUDA_ARCH_LIST=5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX
set PYTORCH_BUILD_VERSION=2.11.0a0+cu113
```

## Quick Test

After installation, verify everything works:

```python
import torch
import torchaudio

print(f"torchaudio version: {torchaudio.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test audio loading
waveform, sample_rate = torchaudio.load('test.wav')
print(f"Loaded audio: shape={waveform.shape}, sample_rate={sample_rate}")

# Test resampling
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
resampled = resampler(waveform)
print(f"Resampled: shape={resampled.shape}")
```

See [test_torchaudio.py](test/test_torchaudio.py) for a comprehensive test suite.

## Version Info

| Component | Version |
|-----------|---------|
| torchaudio | 2.11.0a0+cu113 |
| PyTorch (required) | 2.13.0a0+cu113 |
| Python | 3.8 |
| CUDA | 11.3 |
| Platform | Windows x64 |

## License

This project follows the original torchaudio license (BSD 2-Clause). See [LICENSE](LICENSE) for details.

## Acknowledgments

- [PyTorch Audio Team](https://github.com/pytorch/audio) for the original torchaudio project
- The backport modifications were made to support legacy Python 3.8 environments

---

<a id="中文"></a>

# TorchAudio Python 3.8 回移植（CUDA 11.3）

基于 torchaudio 主分支（v2.11.0a0）的修改版本，支持 **Python 3.8** 和 **CUDA 11.3**。

## 这是什么？

这是一个为仍需要 **Python 3.8** 兼容性的用户修改构建的 torchaudio。官方 PyTorch 生态系统从 PyTorch 2.5+ 起放弃了 Python 3.8 支持，但许多遗留环境和项目仍依赖 Python 3.8。本项目通过将最新的 torchaudio 功能回移植到 Python 3.8 + CUDA 11.3 来弥补这一差距。

本仓库是 **PyTorch Python 3.8 回移植套件** 的一部分：
- [pytorch_backport_py38](https://github.com/Lanurence666/pytorch_backport_py38) — 回移植的 PyTorch 2.13.0a0+cu113
- **audio_backport_py38**（本仓库）— 回移植的 torchaudio 2.11.0a0+cu113

## 修改与修复

### 1. Soundfile 回退后端

**问题：** 上游 torchaudio（v2.9+）硬依赖 `torchcodec` 进行所有音频 I/O 操作。然而 `torchcodec` 需要 FFmpeg，且不为 Python 3.8 Windows 提供预编译 wheel，导致无法在 Python 3.8 环境中安装。

**修复：** 我们在 `torchaudio/__init__.py` 中实现了自动回退机制：
- 添加了 `_TORCHCODEC_AVAILABLE` 检测标志
- 实现了基于 soundfile 的 `_load_with_soundfile()` 函数
- 实现了基于 soundfile 的 `_save_with_soundfile()` 函数
- 修改了 `load()` 和 `save()` 以在 torchcodec 不可用时自动回退到 soundfile

### 2. TorchCodec 导入检测修复

**问题：** 原始代码仅导入 C 扩展，未检查 `torchcodec` Python 包是否可用，可能导致运行时 `ImportError`。

**修复：** 在 try 块中添加 `import torchcodec`，确保 C 扩展和 Python 包都可用后才标记为可用。

## 测试结果

在 Windows x64 + Python 3.8 + CUDA 11.3 环境下进行了全面测试，所有核心功能测试通过，包括音频加载/保存、重采样、MFCC、频谱图、基频检测等。

**已知限制：**
- Soundfile 回退仅支持 WAV/FLAC/OGG 格式
- `torch.compile()` 在此 Python 3.8 构建上不完全支持
- 可能出现 NumPy 2.x 兼容性警告（建议使用 `numpy<2`）

## 安装

从 [GitHub Releases](https://github.com/Lanurence666/audio_backport_py38/releases) 下载 wheel 安装：

```bash
pip install torchaudio-2.11.0a0+cu113-cp38-cp38-win_amd64.whl
```

或从源码构建：
```bash
git clone https://github.com/Lanurence666/audio_backport_py38.git
cd audio_backport_py38
pip install soundfile numpy
python setup.py install
```

## 许可证

本项目遵循原始 torchaudio 许可证（BSD 2-Clause）。详见 [LICENSE](LICENSE)。

---

<a id="русский"></a>

# TorchAudio бэкпорт для Python 3.8 (CUDA 11.3)

Модифицированная версия [torchaudio](https://github.com/pytorch/audio) с поддержкой **Python 3.8** и **CUDA 11.3**, основанная на основной ветке torchaudio (v2.11.0a0).

## Что это?

Это модифицированная сборка torchaudio для пользователей, которым по-прежнему нужна совместимость с **Python 3.8**. Официальная экосистема PyTorch прекратила поддержку Python 3.8 начиная с PyTorch 2.5+, но многие устаревшие среды и проекты всё ещё зависят от Python 3.8. Этот проект устраняет этот разрыв, перенося последние функции torchaudio на Python 3.8 с поддержкой CUDA 11.3.

Этот репозиторий является частью **набора бэкпортов PyTorch для Python 3.8**:
- [pytorch_backport_py38](https://github.com/Lanurence666/pytorch_backport_py38) — Бэкпорт PyTorch 2.13.0a0+cu113 для Python 3.8
- **audio_backport_py38** (этот репозиторий) — Бэкпорт torchaudio 2.11.0a0+cu113 для Python 3.8

## Модификации и исправления

### 1. Резервный бэкенд Soundfile

**Проблема:** Основная ветка torchaudio (v2.9+) жёстко зависит от `torchcodec` для всех операций ввода-вывода аудио. Однако `torchcodec` требует FFmpeg и не предоставляет предустановленные wheel-пакеты для Python 3.8 на Windows.

**Исправление:** Мы реализовали автоматический механизм отката в `torchaudio/__init__.py`:
- Добавили флаг обнаружения `_TORCHCODEC_AVAILABLE`
- Реализовали функцию загрузки аудио на основе soundfile `_load_with_soundfile()`
- Реализовали функцию сохранения аудио на основе soundfile `_save_with_soundfile()`
- Изменили `load()` и `save()` для автоматического отката на soundfile при недоступности torchcodec

### 2. Исправление обнаружения импорта TorchCodec

**Проблема:** Исходный код импортировал только C-расширение без проверки доступности пакета `torchcodec`, что могло вызвать `ImportError` во время выполнения.

**Исправление:** Добавили `import torchcodec` в блок try для проверки доступности как C-расширения, так и Python-пакета.

## Результаты тестирования

Проведено комплексное тестирование на Windows x64 + Python 3.8 + CUDA 11.3. Все основные функциональные тесты пройдены, включая загрузку/сохранение аудио, ресемплинг, MFCC, спектрограмму, определение высоты тона и др.

**Известные ограничения:**
- Резервный бэкенд soundfile поддерживает только форматы WAV/FLAC/OGG
- `torch.compile()` не полностью поддерживается в этой сборке Python 3.8
- Может появиться предупреждение о совместимости с NumPy 2.x (рекомендуется `numpy<2`)

## Установка

Загрузите wheel-пакет из [GitHub Releases](https://github.com/Lanurence666/audio_backport_py38/releases) и установите:

```bash
pip install torchaudio-2.11.0a0+cu113-cp38-cp38-win_amd64.whl
```

Или соберите из исходного кода:
```bash
git clone https://github.com/Lanurence666/audio_backport_py38.git
cd audio_backport_py38
pip install soundfile numpy
python setup.py install
```

## Лицензия

Этот проект следует оригинальной лицензии torchaudio (BSD 2-Clause). См. [LICENSE](LICENSE) для подробностей.