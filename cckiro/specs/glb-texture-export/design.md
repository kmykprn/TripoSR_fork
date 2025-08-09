# GLB形式でのテクスチャ付きメッシュ出力機能 - 設計書

## 1. アーキテクチャ概要

### 1.1 処理フロー
```
入力画像
    ↓
TSRモデル処理（既存）
    ↓
メッシュ抽出（既存）
    ↓
テクスチャベイキング（既存）
    ↓
出力形式による分岐【修正箇所】
    ├─ GLB形式: trimeshでUV座標付きメッシュ出力
    └─ OBJ形式: xatlasで出力（既存）
```

### 1.2 修正対象ファイル
- `/home/yikeda/work/TripoSR_fork/run.py`
  - `generate_3d_mesh_from_image`関数のメッシュ出力部分

## 2. 詳細設計

### 2.1 出力処理の分岐ロジック

```python
# run.py の 107-125行目付近を修正

if args.bake_texture:
    # テクスチャベイキングあり
    out_texture_path = os.path.join(output_dir, str(image_index), "texture.png")
    
    # UV展開とテクスチャ色の計算（既存処理）
    timer.start("Baking texture")
    bake_output = bake_texture(meshes[0], model, scene_codes[0], args.texture_resolution)
    timer.end("Baking texture")
    
    # 出力形式による分岐
    timer.start("Exporting mesh and texture")
    
    if args.model_save_format == "glb":
        # GLB形式での出力（新規実装）
        export_glb_with_texture(
            out_mesh_path=out_mesh_path,
            out_texture_path=out_texture_path,
            mesh=meshes[0],
            bake_output=bake_output
        )
    else:
        # OBJ形式での出力（既存処理）
        xatlas.export(
            out_mesh_path,
            meshes[0].vertices[bake_output["vmapping"]],
            bake_output["indices"],
            bake_output["uvs"],
            meshes[0].vertex_normals[bake_output["vmapping"]]
        )
        # テクスチャ画像の保存
        Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8))\
            .transpose(Image.FLIP_TOP_BOTTOM)\
            .save(out_texture_path)
    
    timer.end("Exporting mesh and texture")
else:
    # テクスチャベイキングなし（既存処理を維持）
    timer.start("Exporting mesh")
    meshes[0].export(out_mesh_path)
    timer.end("Exporting mesh")
```

### 2.2 GLB出力関数の実装

```python
def export_glb_with_texture(out_mesh_path, out_texture_path, mesh, bake_output):
    """
    GLB形式でUV座標付きメッシュとテクスチャを出力する
    
    Args:
        out_mesh_path: GLBファイルの出力パス
        out_texture_path: テクスチャ画像の出力パス
        mesh: 元のtrimeshオブジェクト
        bake_output: bake_texture関数の出力辞書
    """
    # テクスチャ画像の準備
    texture_array = (bake_output["colors"] * 255.0).astype(np.uint8)
    texture_image = Image.fromarray(texture_array).transpose(Image.FLIP_TOP_BOTTOM)
    
    # テクスチャ画像を別ファイルとして保存
    texture_image.save(out_texture_path)
    
    # UV座標付きメッシュの作成
    # 注：GLBファイルにはUV座標のみを含め、テクスチャ参照は含めない
    textured_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[bake_output["vmapping"]],
        faces=bake_output["indices"],
        vertex_normals=mesh.vertex_normals[bake_output["vmapping"]]
    )
    
    # UV座標をメッシュに設定
    # trimeshのvisualにUV座標を設定（マテリアルなし）
    textured_mesh.visual = trimesh.visual.TextureVisuals(
        uv=bake_output["uvs"],
        material=None  # フロントエンドで別途テクスチャを適用するため
    )
    
    # GLB形式で出力
    textured_mesh.export(out_mesh_path)
```

## 3. データ構造

### 3.1 bake_output辞書の構造（既存）
```python
{
    "vmapping": np.ndarray,    # 頂点マッピング
    "indices": np.ndarray,      # 面インデックス (N, 3)
    "uvs": np.ndarray,          # UV座標 (M, 2)
    "colors": np.ndarray        # テクスチャ色 (H, W, 4)
}
```

### 3.2 出力ファイル構造
```
output/
└── 0/
    ├── input.png           # 入力画像（既存）
    ├── mesh.glb           # UV座標付きメッシュ
    └── texture.png        # テクスチャ画像
```

## 4. エラーハンドリング

### 4.1 例外処理
```python
try:
    export_glb_with_texture(...)
except Exception as e:
    logging.error(f"Failed to export GLB with texture: {e}")
    # フォールバック：OBJ形式で出力を試みる
    logging.info("Falling back to OBJ format...")
    xatlas.export(...)
```

### 4.2 検証処理
- UV座標の範囲チェック（0.0 ～ 1.0）
- 頂点数とUV座標数の整合性確認
- ファイル書き込み権限の確認

## 5. パフォーマンス考慮事項

### 5.1 メモリ使用量
- trimeshオブジェクトの複製を最小限に抑える
- テクスチャ画像は一度だけメモリに保持

### 5.2 処理速度
- 既存のベイキング処理結果を再利用
- 不要な変換処理を避ける

## 6. 後方互換性

### 6.1 既存機能への影響
- OBJ出力時の処理は変更なし
- コマンドラインオプションの互換性維持
- デフォルト動作の維持

### 6.2 移行パス
- 既存ユーザーは何も変更せずに従来通り使用可能
- GLB+テクスチャが必要なユーザーのみ新オプションを使用