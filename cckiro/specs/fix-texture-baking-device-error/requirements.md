# テクスチャベイキング時のデバイスエラー修正 - 要件ファイル

## 問題の概要
`python run.py examples/chair.png --output-dir output/ --bake-texture --texture-resolution 2048` を実行した際に、RuntimeErrorが発生する。

## エラーの詳細
- **エラーメッセージ**: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`
- **発生場所**: `tsr/models/nerf_renderer.py` の78行目、`query_triplane` メソッド内
- **具体的な処理**: `F.grid_sample` 関数を呼び出す際に、inputとgridが異なるデバイス上にある

## 満たすべき要件

### 1. 基本要件
- [ ] テクスチャベイキング機能（`--bake-texture`オプション）が正常に動作すること
- [ ] GPUが利用可能な環境で、すべてのテンソルが同じデバイス（cuda:0）上で処理されること
- [ ] CPUのみの環境でも動作すること（fallback対応）

### 2. 技術要件
- [ ] `F.grid_sample`に渡される`input`と`grid`が同じデバイス上にあること
- [ ] `query_triplane`メソッド内でのテンソル処理が一貫したデバイス上で行われること
- [ ] `positions_to_colors`関数から`query_triplane`に渡されるデータが適切なデバイス上にあること

### 3. エラー処理要件
- [ ] デバイスの不一致が検出された場合、適切にデバイスを統一すること
- [ ] エラーメッセージが分かりやすく、デバッグしやすいこと

### 4. パフォーマンス要件
- [ ] 不要なデバイス間のデータ転送を避けること
- [ ] GPUメモリの効率的な使用

## 成功基準
1. `python run.py examples/chair.png --output-dir output/ --bake-texture --texture-resolution 2048` が正常に実行される
2. 出力ディレクトリにテクスチャ付きの3Dメッシュが生成される
3. エラーやワーニングが発生しない

## 影響範囲
- `tsr/models/nerf_renderer.py`: `query_triplane`メソッド
- `tsr/bake_texture.py`: `positions_to_colors`関数、`bake_texture`関数
- `run.py`: `generate_3d_mesh_from_image`関数（間接的）