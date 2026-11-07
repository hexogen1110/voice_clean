# record_processor — 人聲擷取 / 降噪 / 去爆音工具

把充滿背景雜音、爆音的錄音整理成清楚的人聲，方便語言學習（跟讀、聽寫）。
支援單檔與批次處理，可選擇放慢速度（不變調）。

## 功能

- **去爆音（de-pop / de-click）**：移除敲擊、麥克風碰撞等短暫的寬頻爆音
- **降噪（noise reduction）**：用深度學習模型 DeepFilterNet3 分離人聲與背景噪音
- **音量一致**：響度正規化（loudness normalization），每個檔案聽起來一樣大聲，不會再爆音
- **放慢播放**：時間伸縮（time-stretch），速度變慢但音高不變
- **批次處理**：資料夾、子資料夾、萬用字元（wildcard），可中斷後續跑

## 處理流程

```
輸入音檔
  │ 1. ffmpeg 解碼 → 單聲道 48 kHz
  │ 2. 高通濾波 (high-pass filter, 80 Hz)：去除低頻隆隆聲、噴麥 (plosive)
  │ 3. 去爆音：偵測 5 kHz 以上能量的瞬間突升 (transient)，
  │    在短時傅立葉轉換 (STFT) 頻域中以前後正常音框的中位數修補
  │ 4. DeepFilterNet3 語音增強 (speech enhancement)
  │    └ 之後以較嚴格門檻再清一次殘餘爆音
  │ 5. 響度正規化 (EBU R128, -16 LUFS) + 限幅器 (limiter)，可選放慢
  ▼
輸出 *_clean.mp3
```

去爆音只針對「1～3 ms 內突然升高 18 dB 以上」的瞬態。德文的 s、sch、f 等擦音（fricative）是緩慢升起的，不會被誤判。

## 安裝

需要 Python 3.10 以上。

```bash
pip install deepfilternet torch torchaudio numpy scipy soundfile imageio-ffmpeg
```

- `imageio-ffmpeg` 內含 ffmpeg 執行檔，不需另外設定 PATH。如果系統已經裝了 ffmpeg（例如 `winget install Gyan.FFmpeg`），程式會優先使用系統的 ffmpeg。
- 第一次執行時，DeepFilterNet3 模型（約 8 MB）會自動從 GitHub 下載到使用者快取資料夾。
- 安裝時如果出現 `deepfilternet requires numpy<2.0` 的警告，可以忽略。實測搭配 numpy 2.x 可以正常運作。

## 使用方式

### 單檔

```bash
python voice_clean.py 錄音.mp3                    # 輸出 錄音_clean.mp3
python voice_clean.py 錄音.mp3 -o 乾淨版.mp3       # 指定輸出檔名
python voice_clean.py 錄音.mp3 --slow 0.85        # 放慢 15% → 錄音_clean_0.85x.mp3
```

### 批次

```bash
python voice_clean.py D:\German                         # 處理資料夾內所有音檔
python voice_clean.py D:\German -r                      # 包含子資料夾
python voice_clean.py D:\German -r -d D:\German_clean   # 輸出到另一個資料夾（保留子資料夾結構）
python voice_clean.py "D:\German\*.mp3"                 # 萬用字元（請加引號）
python voice_clean.py a.mp3 b.m4a c.wav                 # 指定多個檔案
```

批次處理時的行為：

- 支援格式：mp3、wav、m4a、flac、ogg、aac、wma、mp4
- 檔名含 `_clean` 的檔案會自動跳過，避免重複處理輸出檔
- 輸出檔已存在就跳過，中斷後重新執行會從沒處理的檔案接著做；加 `--overwrite` 可強制重做
- 某個檔案失敗不會中斷整批，最後會列出成功、跳過、失敗的統計
- 模型只載入一次

## 參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `inputs` | （必填） | 檔案、萬用字元或資料夾，可以給多個 |
| `-o, --output` | — | 輸出檔名（只能用在單一輸入） |
| `-d, --outdir` | 原檔旁 | 批次輸出資料夾 |
| `-r, --recursive` | 關 | 包含子資料夾 |
| `--overwrite` | 關 | 重新處理已存在的輸出 |
| `--slow` | `1.0` | 播放速度，例如 `0.8` 代表慢 20%，音高不變 |
| `--atten` | 不限 | 最大降噪量（dB）。不設定時最乾淨；設 `20`～`30` 會保留一點環境聲，聽起來較自然 |
| `--pop-db` | `18` | 爆音偵測門檻（dB）。數值越小越敏感 |
| `--hp` | `80` | 高通濾波截止頻率（Hz）。男聲很低沉時可以降到 `60` |
| `--lufs` | `-16` | 目標響度。耳機聆聽建議 `-16`～`-14` |
| `--bitrate` | `192k` | MP3 位元率（bitrate） |

## 調整建議

| 狀況 | 建議 |
|---|---|
| 聲音太「乾」，像在真空裡 | `--atten 25` |
| 還有爆音沒清掉 | `--pop-db 14`（太低可能傷到 t、k、p 等爆破音） |
| 子音被吃掉、聽起來悶 | `--pop-db 22` |
| 跟讀練習 | `--slow 0.8` 或 `--slow 0.85` |
| 低頻嗡嗡聲（例如冷氣、風扇） | `--hp 120` |

## 效能

只用 CPU 時，速度約為即時的 10 倍：2.5 分鐘的錄音大約 15 秒處理完。不需要 GPU。

## 疑難排解

**`FileNotFoundError: [WinError 2] 系統找不到指定的檔案`**
找不到 ffmpeg。執行 `pip install imageio-ffmpeg`，或安裝系統版 ffmpeg 後重新開啟終端機。

**`ModuleNotFoundError: No module named 'torchaudio.backend'`**
新版 torchaudio 移除了這個模組，但 deepfilternet 0.5.6 還在使用它。目前版本的 `voice_clean.py` 已經內建替代模組（shim），請更新到最新的程式檔。

**模型下載失敗（403 / 連線錯誤）**
公司網路可能擋了 GitHub。可以手動下載
`https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3.zip`，
再解壓縮到：
- Windows：`%LOCALAPPDATA%\DeepFilterNet\DeepFilterNet\Cache\DeepFilterNet3\`
- Linux：`~/.cache/DeepFilterNet/DeepFilterNet3/`

## 限制

- 人聲和爆音完全重疊的片段，那一小段人聲可能會變弱。
- DeepFilterNet 會保留所有人聲。多人同時說話時無法分出單一說話者。
- 輸出一律是單聲道（mono）MP3。

## 授權與致謝

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)：MIT / Apache-2.0
- [FFmpeg](https://ffmpeg.org/)
