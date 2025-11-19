FROM mcr.microsoft.com/devcontainers/python:0-3.8

# Instalar dependencias del sistema para OpenCV y librerías científicas
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && apt-get clean

# Crear y activar un entorno virtual (opcional pero recomendado)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar dependencias Python básicas
RUN pip install --upgrade pip setuptools wheel

# Si usas TensorFlow o cv2, puedes fijarlas aquí
# RUN pip install tensorflow opencv-python

WORKDIR /workspace