"""
TorchAudio Backport Py38 - Comprehensive Test Suite
===================================================
This script tests the key functionality of the backported torchaudio.
Run: python test/test_torchaudio.py
"""

import sys
import os
import tempfile
import warnings

import torch
import torchaudio


def test_version():
    print("[Test 1] Version check")
    print(f"  torchaudio version: {torchaudio.__version__}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
    assert torchaudio.__version__.startswith("2.11"), f"Unexpected version: {torchaudio.__version__}"
    print("  PASSED")


def test_generate_test_audio():
    print("[Test 2] Generate test audio signal")
    sample_rate = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = (0.5 * torch.sin(2 * 3.14159265 * 440 * t)).unsqueeze(0).float()
    print(f"  Generated waveform: shape={waveform.shape}, dtype={waveform.dtype}")
    assert waveform.shape[0] == 1
    assert waveform.shape[1] == sample_rate
    print("  PASSED")
    return waveform, sample_rate


def test_save_and_load(waveform, sample_rate):
    print("[Test 3] Save and load audio")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torchaudio.save(tmp_path, waveform, sample_rate)
        print(f"  Saved to: {tmp_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded_waveform, loaded_sr = torchaudio.load(tmp_path)
        print(f"  Loaded: shape={loaded_waveform.shape}, sr={loaded_sr}")
        assert loaded_sr == sample_rate, f"Sample rate mismatch: {loaded_sr} != {sample_rate}"
        assert loaded_waveform.shape[0] == 1, f"Channel mismatch: {loaded_waveform.shape[0]}"
        print("  PASSED")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_resample(waveform, sample_rate):
    print("[Test 4] Resampling")
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
    resampled = resampler(waveform)
    print(f"  Original: {waveform.shape}, Resampled: {resampled.shape}")
    assert resampled.shape[0] == waveform.shape[0]
    print("  PASSED")


def test_spectrogram(waveform):
    print("[Test 5] Spectrogram")
    spec_transform = torchaudio.transforms.Spectrogram(n_fft=512)
    spec = spec_transform(waveform)
    print(f"  Spectrogram shape: {spec.shape}")
    assert spec.dim() == 2
    print("  PASSED")


def test_mel_spectrogram(waveform, sample_rate):
    print("[Test 6] MelSpectrogram")
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=512, n_mels=64)
    mel_spec = mel_transform(waveform)
    print(f"  MelSpectrogram shape: {mel_spec.shape}")
    assert mel_spec.dim() == 2
    print("  PASSED")


def test_mfcc(waveform, sample_rate):
    print("[Test 7] MFCC")
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=13)
    mfcc = mfcc_transform(waveform)
    print(f"  MFCC shape: {mfcc.shape}")
    assert mfcc.dim() == 2
    assert mfcc.shape[0] == 13
    print("  PASSED")


def test_pitch_detection(waveform, sample_rate):
    print("[Test 8] Pitch detection")
    pitch = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate)
    print(f"  Pitch shape: {pitch.shape}")
    print(f"  Mean pitch: {pitch.mean().item():.1f} Hz")
    assert pitch.shape[0] == waveform.shape[0]
    print("  PASSED")


def test_amplitude_to_db():
    print("[Test 9] Amplitude to DB conversion")
    spec = torch.rand(1, 128, 100) * 10
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    db_spec = db_transform(spec)
    print(f"  Input range: [{spec.min().item():.2f}, {spec.max().item():.2f}]")
    print(f"  DB range: [{db_spec.min().item():.2f}, {db_spec.max().item():.2f}]")
    assert db_spec.shape == spec.shape
    print("  PASSED")


def test_soundfile_fallback():
    print("[Test 10] Soundfile fallback detection")
    try:
        import torchcodec
        print("  TorchCodec is available - primary backend will be used")
    except ImportError:
        print("  TorchCodec is NOT available - soundfile fallback will be used")
        assert hasattr(torchaudio, '_load_with_soundfile'), "soundfile fallback not available"
        assert hasattr(torchaudio, '_save_with_soundfile'), "soundfile fallback not available"
        print("  Soundfile fallback functions are present")
    print("  PASSED")


def test_stereo_audio():
    print("[Test 11] Stereo audio handling")
    sample_rate = 16000
    duration = 0.5
    t = torch.linspace(0, duration, int(sample_rate * duration))
    left = 0.5 * torch.sin(2 * 3.14159265 * 440 * t)
    right = 0.5 * torch.sin(2 * 3.14159265 * 880 * t)
    stereo = torch.stack([left, right])
    print(f"  Stereo waveform shape: {stereo.shape}")
    assert stereo.shape[0] == 2

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torchaudio.save(tmp_path, stereo, sample_rate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded, sr = torchaudio.load(tmp_path)
        print(f"  Loaded stereo: shape={loaded.shape}, sr={sr}")
        assert loaded.shape[0] == 2, f"Expected 2 channels, got {loaded.shape[0]}"
        print("  PASSED")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_frame_offset():
    print("[Test 12] Frame offset loading")
    sample_rate = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = (0.5 * torch.sin(2 * 3.14159265 * 440 * t)).unsqueeze(0).float()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torchaudio.save(tmp_path, waveform, sample_rate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded, sr = torchaudio.load(tmp_path, frame_offset=4000, num_frames=2000)
        print(f"  Loaded with offset: shape={loaded.shape}")
        assert loaded.shape[1] == 2000, f"Expected 2000 frames, got {loaded.shape[1]}"
        print("  PASSED")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    print("=" * 60)
    print("TorchAudio Backport Py38 - Test Suite")
    print("=" * 60)

    results = []
    waveform, sample_rate = None, None

    tests = [
        ("Version check", test_version),
        ("Generate test audio", lambda: test_generate_test_audio()),
    ]

    for name, test_fn in tests:
        try:
            result = test_fn()
            if result is not None:
                waveform, sample_rate = result
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  FAILED: {e}")

    if waveform is not None:
        remaining_tests = [
            ("Save and load", lambda: test_save_and_load(waveform, sample_rate)),
            ("Resampling", lambda: test_resample(waveform, sample_rate)),
            ("Spectrogram", lambda: test_spectrogram(waveform)),
            ("MelSpectrogram", lambda: test_mel_spectrogram(waveform, sample_rate)),
            ("MFCC", lambda: test_mfcc(waveform, sample_rate)),
            ("Pitch detection", lambda: test_pitch_detection(waveform, sample_rate)),
            ("AmplitudeToDB", lambda: test_amplitude_to_db()),
            ("Soundfile fallback", lambda: test_soundfile_fallback()),
            ("Stereo audio", lambda: test_stereo_audio()),
            ("Frame offset", lambda: test_frame_offset()),
        ]
        for name, test_fn in remaining_tests:
            try:
                test_fn()
                results.append((name, True, None))
            except Exception as e:
                results.append((name, False, str(e)))
                print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    for name, ok, err in results:
        status = "PASS" if ok else f"FAIL ({err})"
        print(f"  [{status}] {name}")
    print(f"\nTotal: {passed} passed, {failed} failed out of {len(results)} tests")

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()