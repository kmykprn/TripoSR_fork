FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# パッケージをインストール
RUN apt-get update && apt-get install -y git
WORKDIR /
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# コードをコピー
COPY tsr/ /tsr/
COPY tsr_pipeline/ /tsr_pipeline/
COPY handler.py /handler.py

# Run handler for Runpod
CMD ["python", "-u", "/app/handler.py"]