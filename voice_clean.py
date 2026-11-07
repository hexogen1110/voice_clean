#!/usr/bin/env python3
"""
voice_clean.py — 人聲擷取 / 降噪 / 去爆音工具 (for language-learning recordings)

Pipeline:
  1. ffmpeg 解碼 → mono 48 kHz float
  2. 高通濾波 (high-pass) 去除低頻隆隆聲 / 喷麥 (plosive) 爆音
  3. 去爆音 (de-pop / de-click):偵測高頻能量突升的瞬態,在 STFT 頻域修補
  4. (模型後再做一次殘餘爆音清除)
  5. DeepFilterNet3 深度學習語音增強 (speech enhancement) → 去除背景噪音
  6. 響度正規化 (loudness normalization, EBU R128) + 限幅器 (limiter) 防止再爆音
  7. 可選:放慢速度 (time-stretch, 不變調) 方便跟讀

用法:
  python3 voice_clean.py input.mp3                       # 輸出 input_clean.mp3
  python3 voice_clean.py input.mp3 -o out.mp3 --slow 0.85
  python3 voice_clean.py *.mp3 --atten 30               # 批次處理,保留一點環境聲
需要: ffmpeg, numpy, scipy, soundfile, torch, deepfilternet
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, medfilt

SR = 48000  # DeepFilterNet 原生取樣率


def find_ffmpeg() -> str:
    """找 ffmpeg:先找系統 PATH,找不到就用 imageio-ffmpeg 內建的執行檔。"""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("找不到 ffmpeg。請擇一:\n"
                 "  (A) pip install imageio-ffmpeg   (最簡單,免設定 PATH)\n"
                 "  (B) winget install Gyan.FFmpeg   (裝完要重開終端機)")


FFMPEG = find_ffmpeg()


# ---------------------------------------------------------------- I/O
def decode(path: str) -> np.ndarray:
    cmd = [FFMPEG, "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
           "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def encode(x: np.ndarray, out: str, slow: float, lufs: float, bitrate: str):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        tmp = t.name
    sf.write(tmp, x, SR, subtype="FLOAT")
    af = []
    if slow and abs(slow - 1.0) > 1e-3:
        af.append(f"atempo={slow}")
    # 兩段式其實較準,這裡用單段 loudnorm + 真峰值限制即可
    af.append(f"loudnorm=I={lufs}:TP=-1.5:LRA=11")
    af.append("alimiter=limit=0.89:attack=2:release=50:level=disabled")
    cmd = [FFMPEG, "-v", "error", "-y", "-i", tmp, "-af", ",".join(af),
           "-ar", "44100"]
    if out.lower().endswith(".mp3"):
        cmd += ["-c:a", "libmp3lame", "-b:a", bitrate]
    cmd.append(out)
    subprocess.run(cmd, check=True)
    os.remove(tmp)


# ---------------------------------------------------------------- DSP
def highpass(x: np.ndarray, fc: float = 80.0) -> np.ndarray:
    sos = butter(4, fc, btype="highpass", fs=SR, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def detect_pops(x: np.ndarray, rise_db: float = 18.0, hf: float = 5000.0,
                max_ms: float = 60.0) -> np.ndarray:
    """偵測爆音 / 敲擊聲 (pops / clicks / transients)。
    特徵:高頻 (>5 kHz) 能量包絡在 1~2 ms 內突然比周圍 300 ms 的中位數高出
    rise_db 以上,且持續很短。語音的擦音 (s, sch, f) 是緩升、較低,不會誤判。
    回傳每個樣本是否屬於爆音的布林遮罩。"""
    h = sosfiltfilt(butter(4, hf, btype="highpass", fs=SR, output="sos"), x)
    win = int(0.001 * SR)                       # 1 ms 包絡
    hop = win
    n = len(h) // hop
    env = np.sqrt(np.mean(h[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    edb = 20 * np.log10(env)
    base = medfilt(edb, 301)                   # 300 ms 滑動中位數 = 背景基準
    over = edb - base
    # 突升:與 3 ms 前相比上升很多
    rise = edb - np.r_[np.full(3, edb[0]), edb[:-3]]
    trig = (over > rise_db) & (rise > rise_db * 0.6)
    mask = np.zeros(n, bool)
    maxf = int(max_ms)
    i = 0
    while i < n:
        if trig[i]:
            s = max(i - 2, 0)
            e = i
            # 延伸到包絡掉回 基準+6 dB 為止(含衰減尾巴)
            while e < n - 1 and over[e] > 6 and e - s < maxf:
                e += 1
            mask[s : e + 2] = True
            i = e + 1
        else:
            i += 1
    return np.r_[np.repeat(mask, hop), np.zeros(len(x) - n * hop, bool)]


def remove_pops(x: np.ndarray, rise_db: float = 18.0) -> tuple[np.ndarray, int]:
    """在 STFT (短時傅立葉轉換) 頻域修補爆音:被標記的音框,各頻率的幅度
    限制為前後 ±120 ms 正常音框的中位數,保留相位,讓底下的語音盡量留下。"""
    from scipy.signal import stft, istft
    m = detect_pops(x, rise_db)
    # 計算事件數
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    nev = int((d == 1).sum())
    if nev == 0:
        return x, 0
    nfft, hop = 1024, 256
    f, t, X = stft(x, SR, nperseg=nfft, noverlap=nfft - hop, boundary="even")
    mag, ph = np.abs(X), np.angle(X)
    # 音框層級遮罩 (稍微擴張一格)
    fm = np.array([m[max(int(c * hop) - nfft // 2, 0): int(c * hop) + nfft // 2].any()
                   for c in range(X.shape[1])])
    fm = fm | np.r_[False, fm[:-1]] | np.r_[fm[1:], False]
    ctx = int(0.12 * SR / hop)
    good = ~fm
    for j in np.flatnonzero(fm):
        l, r = max(j - ctx, 0), min(j + ctx + 1, X.shape[1])
        nb = np.flatnonzero(good[l:r]) + l
        ref = np.median(mag[:, nb], axis=1) if len(nb) else np.zeros(mag.shape[0])
        mag[:, j] = np.minimum(mag[:, j], ref)
    _, y = istft(mag * np.exp(1j * ph), SR, nperseg=nfft, noverlap=nfft - hop)
    y = y[: len(x)].astype(np.float32)
    if len(y) < len(x):
        y = np.r_[y, np.zeros(len(x) - len(y), np.float32)]
    # 只替換爆音區附近,其餘保持原樣 (平滑過渡)
    g = np.convolve(fm.repeat(hop)[: len(x)].astype(np.float32),
                    np.hanning(hop * 2) / (hop), "same")
    g = np.clip(g, 0, 1)
    if len(g) < len(x):
        g = np.r_[g, np.zeros(len(x) - len(g))]
    return (x * (1 - g) + y * g).astype(np.float32), nev


# ---------------------------------------------------------------- Deep model
_DF = None


def _patch_torchaudio():
    """新版 torchaudio (>=2.9) 移除了 torchaudio.backend,
    但 deepfilternet 0.5.6 還在 import 它 → 補一個假模組 (shim) 讓它能載入。
    本程式不使用 df 的檔案讀寫功能,所以不影響結果。"""
    try:
        import torchaudio.backend.common  # noqa: F401
        return
    except ImportError:
        pass
    import types
    import torchaudio
    meta = getattr(torchaudio, "AudioMetaData", None)
    if meta is None:
        class meta:  # 佔位用
            pass
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = meta
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


def enhance_dfn(x: np.ndarray, atten_db: float | None, chunk_s: float = 30.0):
    import torch
    import warnings
    warnings.filterwarnings("ignore")
    _patch_torchaudio()
    from df.enhance import init_df, enhance
    global _DF
    if _DF is None:  # 批次處理時模型只載入一次
        _DF = init_df(log_level="ERROR")
    model, state, _ = _DF
    chunk = int(chunk_s * SR)
    ov = int(0.5 * SR)  # 0.5 s 重疊,交叉淡化 (crossfade) 避免接縫
    out = np.zeros_like(x)
    w = np.zeros_like(x)
    pos = 0
    while pos < len(x):
        s, e = max(pos - ov, 0), min(pos + chunk + ov, len(x))
        seg = torch.from_numpy(x[s:e]).unsqueeze(0)
        y = enhance(model, state, seg, atten_lim_db=atten_db).squeeze(0).numpy()
        win = np.ones(e - s, dtype=np.float32)
        if s > 0:
            win[:ov] = np.linspace(0, 1, ov)
        if e < len(x):
            win[-ov:] = np.linspace(1, 0, ov)
        out[s:e] += y[: e - s] * win
        w[s:e] += win
        pos += chunk
        print(f"\r  DeepFilterNet: {min(pos, len(x)) / len(x):5.1%}", end="",
              flush=True)
    print()
    return out / np.maximum(w, 1e-6)


# ---------------------------------------------------------------- main
def process(inp, out, a):
    print(f"[1/5] 解碼 {inp}")
    x = decode(inp)
    print(f"      長度 {len(x) / SR:.1f} s")
    print("[2/5] 高通濾波 (去低頻隆隆 / 喷麥)")
    x = highpass(x, a.hp)
    print("[3/5] 去爆音 / 敲擊聲 (頻域修補)")
    x, nc = remove_pops(x, a.pop_db)
    print(f"      修補 {nc} 個爆音")
    print("[4/5] 深度學習人聲增強 (DeepFilterNet3)")
    x = enhance_dfn(x, a.atten)
    # 模型後再清一次殘餘爆音
    x, nc2 = remove_pops(x, a.pop_db + 10)  # 較嚴格,避免傷到子音
    print(f"      後處理再修補 {nc2} 個殘餘爆音")
    print(f"[5/5] 響度正規化 {a.lufs} LUFS" +
          (f"、放慢至 {a.slow}x" if a.slow != 1.0 else "") + f" → {out}")
    encode(x, out, a.slow, a.lufs, a.bitrate)


AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".mp4"}


def collect_inputs(items, recursive):
    """展開萬用字元 (Windows 的 cmd/PowerShell 不會自動展開 *.mp3) 與資料夾。"""
    import glob
    files = []
    for it in items:
        if os.path.isdir(it):
            pat = os.path.join(it, "**", "*") if recursive else os.path.join(it, "*")
            cand = glob.glob(pat, recursive=recursive)
        elif any(c in it for c in "*?["):
            cand = glob.glob(it, recursive=recursive)
        else:
            cand = [it]
        for f in sorted(cand):
            if os.path.isfile(f) and os.path.splitext(f)[1].lower() in AUDIO_EXT \
                    and "_clean" not in os.path.basename(f):  # 跳過已處理的輸出
                files.append(f)
    return list(dict.fromkeys(files))  # 去重複、保持順序


def main():
    p = argparse.ArgumentParser(description="人聲擷取 / 降噪 / 去爆音 (支援批次)")
    p.add_argument("inputs", nargs="+", help="檔案、萬用字元 (*.mp3) 或資料夾")
    p.add_argument("-o", "--output", help="輸出檔 (單一輸入時)")
    p.add_argument("-d", "--outdir", help="批次輸出資料夾 (預設放在原檔旁)")
    p.add_argument("-r", "--recursive", action="store_true", help="包含子資料夾")
    p.add_argument("--overwrite", action="store_true",
                   help="重新處理已存在的輸出 (預設跳過,可中斷後續跑)")
    p.add_argument("--atten", type=float, default=None,
                   help="最大降噪量 dB;預設不限 (最乾淨)。設 20~30 可保留自然感")
    p.add_argument("--slow", type=float, default=1.0,
                   help="播放速度 (0.5~1.0),例如 0.8 = 慢 20%%,不變調")
    p.add_argument("--lufs", type=float, default=-16.0, help="目標響度")
    p.add_argument("--hp", type=float, default=80.0, help="高通截止頻率 Hz")
    p.add_argument("--pop-db", type=float, default=18.0,
                   help="爆音偵測門檻 dB (越小越敏感,預設 18)")
    p.add_argument("--bitrate", default="192k")
    a = p.parse_args()
    files = collect_inputs(a.inputs, a.recursive)
    if not files:
        sys.exit("沒有找到音檔")
    tag = "_clean" if a.slow == 1.0 else f"_clean_{a.slow}x"
    ok, skip, fail = 0, 0, []
    for k, inp in enumerate(files, 1):
        if a.output and len(files) == 1:
            out = a.output
        else:
            root = os.path.splitext(os.path.basename(inp))[0]
            od = a.outdir or os.path.dirname(inp)
            if a.outdir and a.recursive:  # 保留子資料夾結構
                base = os.path.commonpath([os.path.abspath(f) for f in files]) \
                    if len(files) > 1 else os.path.dirname(os.path.abspath(inp))
                od = os.path.join(a.outdir, os.path.relpath(
                    os.path.dirname(os.path.abspath(inp)), base))
            os.makedirs(od or ".", exist_ok=True)
            out = os.path.join(od, root + tag + ".mp3")
        print(f"\n===== ({k}/{len(files)}) {inp} =====")
        if os.path.exists(out) and not a.overwrite:
            print(f"  已存在,跳過 → {out}")
            skip += 1
            continue
        try:
            process(inp, out, a)
            ok += 1
        except Exception as e:  # 一個檔失敗不影響其他檔
            print(f"  失敗: {e}")
            fail.append(inp)
    print(f"\n完成:成功 {ok}、跳過 {skip}、失敗 {len(fail)}")
    for f in fail:
        print(f"  失敗: {f}")


if __name__ == "__main__":
    sys.exit(main())
