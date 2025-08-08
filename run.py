import logging
import os

import numpy as np
import rembg
import torch
import xatlas
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture
from tsr_pipeline.cli import parse_args
from tsr_pipeline.timer import Timer


# グローバルTimerインスタンスを作成
timer = Timer()


def bg_removal_and_normalize_image(image_path, output_dir, image_index, no_remove_bg, foreground_ratio, rembg_session):
    """背景除去と画像正規化処理を行う
    
    Args:
        image_path: 入力画像のパス
        output_dir: 出力ディレクトリ
        image_index: 画像のインデックス
        no_remove_bg: 背景除去をスキップするかどうか
        foreground_ratio: 前景のサイズ比率
        rembg_session: rembgのセッション（背景除去する場合）
    
    Returns:
        前処理済みの画像（PIL.ImageまたはNumPy配列）
    """
    if no_remove_bg:
        # 背景除去なし：単純に画像を読み込んでRGB変換
        return np.array(Image.open(image_path).convert("RGB"))
    else:
        # 背景除去あり：背景除去、前景調整、アルファチャンネル処理
        image = remove_background(Image.open(image_path), rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        
        # アルファチャンネル処理（透明部分をグレーに）
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        
        # 出力ディレクトリの作成と前処理済み画像の保存
        output_path = os.path.join(output_dir, str(image_index))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        image.save(os.path.join(output_path, "input.png"))
        
        return image


def generate_3d_mesh_from_image(image, image_index, model, device, output_dir, args):
    """画像から3Dメッシュを生成する
    
    処理フロー:
    1. 画像 → Triplane（3D潜在表現）を生成
    2. [オプション] 多視点レンダリング
    3. Triplane → 3Dメッシュを抽出
    4. [オプション] テクスチャベイキング
    5. ファイル出力（OBJ/GLB）
    
    Args:
        image: 前処理済み画像
        image_index: 画像インデックス
        model: TSRモデル
        device: 実行デバイス
        output_dir: 出力ディレクトリ
        args: コマンドライン引数
    """
    logging.info(f"Running image {image_index + 1} ...")
    
    # ========== Step 1: 2D画像から3D表現（Triplane）を生成 ==========
    # Transformerモデルで画像をTriplane（3つの平面で表現された3D潜在表現）に変換
    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")
    
    # ========== Step 2: [オプション] 多視点レンダリング ==========
    # 生成した3Dモデルを30の異なる視点から見た画像を作成
    if args.render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        # 各視点の画像を保存（render_000.png 〜 render_029.png）
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, str(image_index), f"render_{ri:03d}.png"))
        # 回転アニメーション動画も生成
        save_video(
            render_images[0], os.path.join(output_dir, str(image_index), f"render.mp4"), fps=30
        )
        timer.end("Rendering")
    
    # ========== Step 3: Triplaneから3Dメッシュを抽出 ==========
    # マーチングキューブアルゴリズムで3D密度場から表面メッシュを生成
    timer.start("Extracting mesh")
    meshes = model.extract_mesh(scene_codes, not args.bake_texture, resolution=args.mc_resolution)
    timer.end("Extracting mesh")
    
    # ========== Step 4 & 5: メッシュの出力（テクスチャ有無で処理分岐）==========
    out_mesh_path = os.path.join(output_dir, str(image_index), f"mesh.{args.model_save_format}")
    
    if args.bake_texture:
        # テクスチャベイキングあり：UV展開してテクスチャアトラスを生成
        out_texture_path = os.path.join(output_dir, str(image_index), "texture.png")
        
        # UV展開とテクスチャ色の計算
        timer.start("Baking texture")
        bake_output = bake_texture(meshes[0], model, scene_codes[0], args.texture_resolution)
        timer.end("Baking texture")
        
        # UV座標付きOBJファイルとテクスチャ画像を出力
        timer.start("Exporting mesh and texture")
        xatlas.export(out_mesh_path, meshes[0].vertices[bake_output["vmapping"]], bake_output["indices"], bake_output["uvs"], meshes[0].vertex_normals[bake_output["vmapping"]])
        Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(out_texture_path)
        timer.end("Exporting mesh and texture")
    else:
        # テクスチャベイキングなし：頂点カラー付きメッシュを直接出力
        timer.start("Exporting mesh")
        meshes[0].export(out_mesh_path)
        timer.end("Exporting mesh")


def main(args):
    """メイン処理
    
    Args:
        args: コマンドライン引数
    """
    # 出力ディレクトリの準備
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # デバイスの設定
    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"
        logging.info(f"CUDA not available, using CPU")
    
    # モデルの初期化
    timer.start("Initializing model")
    model = TSR.from_pretrained(
        args.pretrained_model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device)
    timer.end("Initializing model")
    
    # 画像の前処理
    timer.start("Processing images")
    images = []
    
    # rembgセッションの初期化
    if args.no_remove_bg:
        rembg_session = None
    else:
        rembg_session = rembg.new_session()
    
    # 各画像を背景除去・正規化
    for i, image_path in enumerate(args.image):
        image = bg_removal_and_normalize_image(
            image_path=image_path,
            output_dir=output_dir,
            image_index=i,
            no_remove_bg=args.no_remove_bg,
            foreground_ratio=args.foreground_ratio,
            rembg_session=rembg_session
        )
        images.append(image)
    
    timer.end("Processing images")
    
    # 各画像から3Dメッシュを生成
    for i, image in enumerate(images):
        generate_3d_mesh_from_image(image, i, model, device, output_dir, args)


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )
    
    # コマンドライン引数をパース
    args = parse_args()
    
    # メイン処理を実行
    main(args)
