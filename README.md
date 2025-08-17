## 使い方
### インストール
- Python >= 3.8
- Install CUDA if available
- Install PyTorch according to your platform: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) **[Please make sure that the locally-installed CUDA major version matches the PyTorch-shipped CUDA major version. For example if you have CUDA 11.x installed, make sure to install PyTorch compiled with CUDA 11.x.]**
- Update setuptools by `pip install --upgrade setuptools`
- Install other dependencies by `pip install -r requirements.txt`
The default options takes about **6GB VRAM** for a single image input.

### 動作確認

(1) .objを生成（テクスチャなし）
```sh
python run.py examples/chair.png --output-dir output/
```

(2) .objを生成（テクスチャあり）
```sh
python run.py examples/chair.png --output-dir output/ --model-save-format obj --bake-texture --texture-resolution 2048
```

(3) .glbを生成（テクスチャなし）
```sh
python run.py examples/chair.png --output-dir output/ --model-save-format glb
```

(4) .glbを生成（テクスチャあり）
```sh
python run.py examples/chair.png --output-dir output/ --model-save-format glb --bake-texture --texture-resolution 1024
```

生成されるもの
- `mesh.glb`: A complete GLB file with:
  - UV coordinates (TEXCOORD_0)
  - Normal vectors (NORMAL) 
  - Embedded texture image
  - PBR material settings
- `texture.png`: The texture image as a separate file (for compatibility)


### Local Gradio App
```sh
python gradio_app.py
```

## Citation
```BibTeX
@article{TripoSR2024,
  title={TripoSR: Fast 3D Object Reconstruction from a Single Image},
  author={Tochilkin, Dmitry and Pankratz, David and Liu, Zexiang and Huang, Zixuan and and Letts, Adam and Li, Yangguang and Liang, Ding and Laforte, Christian and Jampani, Varun and Cao, Yan-Pei},
  journal={arXiv preprint arXiv:2403.02151},
  year={2024}
}
```
