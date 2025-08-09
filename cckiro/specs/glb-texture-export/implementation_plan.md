# GLB形式でのテクスチャ付きメッシュ出力機能 - 実装計画書

## 1. 実装タスク一覧

### タスク1: export_glb_with_texture関数の実装
**対象ファイル**: `/home/yikeda/work/TripoSR_fork/run.py`
**実装位置**: bg_removal_and_normalize_image関数の後（56行目付近）

**実装内容**:
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
    # 実装詳細は設計書参照
```

### タスク2: generate_3d_mesh_from_image関数の修正
**対象ファイル**: `/home/yikeda/work/TripoSR_fork/run.py`
**修正範囲**: 107-125行目

**修正内容**:
1. 出力形式による条件分岐を追加
2. GLB形式の場合は新規関数を呼び出し
3. OBJ形式の場合は既存処理を維持

### タスク3: インポート文の追加
**対象ファイル**: `/home/yikeda/work/TripoSR_fork/run.py`
**修正位置**: ファイル先頭のimport部分

**追加内容**:
```python
import trimesh
```
※既にtrimeshはtsr.utilsで使用されているため、追加の依存関係は不要

## 2. 実装手順

### Step 1: 関数の追加（5分）
1. run.pyを開く
2. 56行目付近に`export_glb_with_texture`関数を追加
3. 関数内の実装を完成させる

### Step 2: メイン処理の修正（10分）
1. `generate_3d_mesh_from_image`関数内の107-125行目を修正
2. `if args.model_save_format == "glb":`の条件分岐を追加
3. GLBの場合は新規関数を呼び出し
4. OBJの場合は既存のxatlas.exportを使用

### Step 3: 動作確認（5分）
1. サンプル画像でGLB出力をテスト
2. ファイルが正しく生成されることを確認
3. エラーが発生しないことを確認

## 3. 実装時の注意事項

### 3.1 コーディング規約
- 既存コードのスタイルに合わせる
- 関数にはdocstringを記載
- 変数名は既存コードと一貫性を保つ

### 3.2 エラーハンドリング
- try-except節で例外をキャッチ
- エラー時は適切なログを出力
- 部分的なファイルが残らないようにする

### 3.3 テスト項目
- GLB形式でのファイル出力
- テクスチャ画像の出力
- UV座標の正確性
- 既存機能への影響なし

## 4. 実装チェックリスト

- [ ] export_glb_with_texture関数の実装
- [ ] generate_3d_mesh_from_image関数の修正
- [ ] インポート文の確認
- [ ] GLB出力の動作確認
- [ ] テクスチャ画像出力の確認
- [ ] OBJ出力が影響を受けていないことの確認
- [ ] エラーハンドリングの動作確認
- [ ] 不要なコメントやデバッグコードの削除

## 5. 実装後の確認コマンド

```bash
# GLB + テクスチャ出力のテスト
python run.py examples/chair.png --output-dir output_test/ --model-save-format glb --bake-texture --texture-resolution 2048

# 出力ファイルの確認
ls -la output_test/0/
# 期待される出力:
# - mesh.glb (GLBファイル)
# - texture.png (テクスチャ画像)
# - input.png (入力画像)

# GLBファイルの形式確認
file output_test/0/mesh.glb
# 期待される出力: "glTF binary model"

# OBJ出力が正常に動作することの確認
python run.py examples/chair.png --output-dir output_obj/ --model-save-format obj --bake-texture
```

## 6. リスクと対策

### リスク1: trimeshのGLB出力でUV座標が失われる
**対策**: TextureVisualsオブジェクトを正しく設定し、UV座標が保持されることを確認

### リスク2: メモリ使用量の増加
**対策**: 大きなオブジェクトの複製を避け、参照を使用

### リスク3: 既存機能への影響
**対策**: 条件分岐を明確にし、既存処理は変更しない