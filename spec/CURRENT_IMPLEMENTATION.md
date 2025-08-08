# run.py 現在の実装内容詳細

## 1. 初期設定部分 (lines 1-45)
- 必要なライブラリのインポート
- Timerクラスの定義（GPU同期考慮の時間計測）
- ログ設定（タイムスタンプ付きINFOレベル）
- グローバルtimerインスタンス作成

## 2. コマンドライン引数定義 (lines 46-112)

### 必須引数
- `image`: 入力画像パス（複数可）

### オプション引数
- `--device`: 実行デバイス（cuda:0/cpu）
- `--pretrained-model-name-or-path`: モデルパス
- `--chunk-size`: メモリ最適化用（デフォルト8192）
- `--mc-resolution`: マーチングキューブ解像度（デフォルト256）
- `--no-remove-bg`: 背景除去スキップフラグ
- `--foreground-ratio`: 前景サイズ比（デフォルト0.85）
- `--output-dir`: 出力先（デフォルト"output/"）
- `--model-save-format`: 出力形式（obj/glb）
- `--bake-texture`: テクスチャベイキング有効化フラグ
- `--texture-resolution`: テクスチャ解像度（デフォルト2048）
- `--render`: NeRFレンダリング動画生成フラグ

## 3. 初期化処理 (lines 114-129)
- 出力ディレクトリ作成
- CUDA利用可能性チェック（なければCPUへフォールバック）
- TSRモデルのロード（HuggingFaceまたはローカルから）
- チャンクサイズ設定
- モデルをデバイスに転送

## 4. 画像前処理 (lines 131-152)

```python
各入力画像に対して:
if 背景除去スキップ:
    - RGB画像として読み込み
else:
    - rembgで背景除去
    - 前景をリサイズ（指定比率）
    - アルファチャンネル処理（透明部分を0.5のグレーに）
    - 前処理済み画像を保存（output/{i}/input.png）
```

## 5. 各画像の3D処理ループ (lines 154-202)

```python
各画像に対して:
1. 3D推論（scene_codes生成）
   - model([image])でTriplane特徴量を生成

2. レンダリング（--renderフラグが有効な場合）
   - 30視点からレンダリング
   - 各視点の画像を保存（render_000.png〜render_029.png）
   - MP4動画生成（render.mp4）

3. メッシュ抽出
   - マーチングキューブでメッシュ生成
   - 頂点カラーまたはテクスチャ付き

4. 出力
   if テクスチャベイキング有効:
       - UV展開とテクスチャ生成
       - テクスチャ画像保存（texture.png）
       - UV座標付きOBJファイル出力（※export_obj_with_textureは削除済み）
   else:
       - 頂点カラー付きメッシュを直接出力
```

## 主な処理フロー

```
入力画像
  ↓ 背景除去・前処理
前処理済み画像
  ↓ TSRモデル
scene_codes（Triplane）
  ↓ 
  ├→ レンダリング → 多視点画像・動画
  └→ メッシュ抽出 → 3Dモデル（OBJ/GLB）
                    └→ テクスチャ（オプション）
```

## 主要な依存関係

### 外部ライブラリ
- `argparse`: コマンドライン引数処理
- `logging`: ログ出力
- `numpy`: 数値計算
- `torch`: 深層学習フレームワーク
- `PIL (Image)`: 画像処理
- `rembg`: 背景除去
- `xatlas`: UV展開（インポートされているが未使用）

### 内部モジュール
- `tsr.system.TSR`: メインの3Dモデル
- `tsr.utils`: ユーティリティ関数
  - `remove_background`: 背景除去
  - `resize_foreground`: 前景リサイズ
  - `save_video`: 動画保存
- `tsr.bake_texture`: テクスチャベイキング
- `export_obj_with_texture`: OBJ出力（削除済みのため要修正）

## 問題点

1. **モノリシック構造**: 200行以上が1ファイルに集中
2. **グローバル変数**: timerがグローバル
3. **エラーハンドリング不足**: try-exceptなし
4. **責務の混在**: 引数処理、画像処理、3D処理が混在
5. **インポートエラー**: export_obj_with_textureが削除済み
6. **ハードコード**: マジックナンバー多数（30視点、0.5のグレー値など）