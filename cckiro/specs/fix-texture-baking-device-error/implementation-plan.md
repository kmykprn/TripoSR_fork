# テクスチャベイキング時のデバイスエラー修正 - 実装計画ファイル

## 実装手順

### 1. デバッグログの削除
- `tsr/models/nerf_renderer.py`からデバッグ用のprint文とフラグを削除
- `tsr/bake_texture.py`からデバッグ用のprint文を削除

### 2. 本修正の実装
- `tsr/bake_texture.py`の`positions_to_colors`関数を修正
  - `positions`テンソルを`scene_code.device`に配置
  - `queried_grid["color"]`を`.cpu()`でCPUに転送してからnumpy変換

### 3. 具体的な変更内容

#### tsr/bake_texture.py の修正
```python
def positions_to_colors(model, scene_code, positions_texture, texture_resolution):
    # 変更前: positions = torch.tensor(positions_texture.reshape(-1, 4)[:, :-1])
    # 変更後: scene_codeと同じデバイスに配置
    positions = torch.tensor(
        positions_texture.reshape(-1, 4)[:, :-1], device=scene_code.device
    )
    
    with torch.no_grad():
        queried_grid = model.renderer.query_triplane(
            model.decoder,
            positions,
            scene_code,
        )
    
    # 変更前: rgb_f = queried_grid["color"].numpy().reshape(-1, 3)
    # 変更後: CPUに転送してからnumpy変換
    rgb_f = queried_grid["color"].cpu().numpy().reshape(-1, 3)
    
    rgba_f = np.insert(rgb_f, 3, positions_texture.reshape(-1, 4)[:, -1], axis=1)
    rgba_f[rgba_f[:, -1] == 0.0] = [0, 0, 0, 0]
    return rgba_f.reshape(texture_resolution, texture_resolution, 4)
```

### 4. 動作確認
- コマンド実行: `python run.py examples/chair.png --output-dir output/ --bake-texture --texture-resolution 2048`
- エラーが発生しないことを確認
- 出力ディレクトリにテクスチャ付き3Dメッシュが生成されることを確認

### 5. コミット＆プッシュ
- 修正内容をコミット
- リモートリポジトリにプッシュ
- mainブランチへのPR作成

## リスク評価
- **低リスク**: 変更は最小限で、既存のロジックに影響しない
- **互換性**: GPU環境とCPU環境の両方で動作可能
- **パフォーマンス**: 必要最小限のデバイス転送のみ実施