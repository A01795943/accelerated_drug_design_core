FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# INSTALAR DEPENDENCIAS DEL SIST
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    aria2 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

#INSTALAR DEPENDENCIAS 
RUN pip3 install \
    jedi \
    omegaconf \
    hydra-core \
    icecream \
    pyrsistent \
    pynvml \
    decorator \
    psutil \
    tqdm \
    numpy \
    scipy \
    scikit-learn \
    xgboost \
    networkx \
    requests \
    pillow \
    matplotlib \
    py3Dmol \
    ipywidgets \
    fair-esm

WORKDIR /workspace

# INSTALAR PYTORCH
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# INSTALAR DEPENDENCIAS BÁSICAS PRIMERO 
RUN pip3 install jedi omegaconf hydra-core icecream pyrsistent pynvml decorator psutil

# CLONAR RFDIFFUSION
RUN git clone https://github.com/sokrypton/RFdiffusion.git

# INSTALAR DGL 
RUN pip3 install --no-dependencies dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

# INSTALAR E3NN Y OTRAS DEPENDENCIAS
RUN pip3 install --no-dependencies e3nn==0.5.5 opt_einsum_fx
RUN pip3 install git+https://github.com/NVIDIA/dllogger#egg=dllogger

# INSTALAR SE3TRANSFORMER
RUN cd RFdiffusion/env/SE3Transformer && pip3 install .

# DESCARGAR ANANAS
RUN wget -qnc https://files.ipd.uw.edu/krypton/ananas && chmod +x ananas

# INSTALAR COLABDESIGN
RUN pip3 install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1
RUN ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign

# CREAR DIRECTORIOS
RUN mkdir -p params RFdiffusion/models

# DESCARGAR MODELOS
RUN aria2c -q -x 16 https://files.ipd.uw.edu/krypton/schedules.zip && \
    unzip -q schedules.zip && rm schedules.zip

RUN aria2c -q -x 16 -d RFdiffusion/models http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
RUN aria2c -q -x 16 -d RFdiffusion/models http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt

# DESCARGAR PARÁMETROS ALPHAFOLD
RUN mkdir -p params && \
    aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar && \
    tar --no-same-owner --no-same-permissions -xf alphafold_params_2022-12-06.tar -C params && \
    rm alphafold_params_2022-12-06.tar && \
    touch params/done.txt

##################################
# PYROSETTA - INSTALL DURING BUILD
# (Previously this tarball had to be downloaded manually into notebooks/)
RUN wget -q --no-check-certificate -O /workspace/PyRosetta4.Release.python310.linux.release-387.tar.bz2 \
  https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python310.linux/PyRosetta4.Release.python310.linux.release-387.tar.bz2
RUN cd /workspace && tar -xf PyRosetta4.Release.python310.linux.release-387.tar.bz2
RUN cd /workspace/PyRosetta4.Release.python310.linux.release-387/setup && python3 setup.py install
##################################

# Variables de entorno
ENV DGLBACKEND=pytorch
ENV PYTHONPATH="/workspace/RFdiffusion:${PYTHONPATH}"

# INSTALAR JAX CON GPU
RUN pip uninstall -y jax jaxlib
RUN pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --upgrade --force-reinstall "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Crear directorio de outputs
RUN mkdir -p outputs logs

# Copy repo into container (build from repo root: docker build -f notebooks/Dockerfile -t rfdiffusion .)
COPY . /workspace/repo

# REST API for pipeline
RUN pip3 install "fastapi[standard]" uvicorn

EXPOSE 8000
WORKDIR /workspace/repo/notebooks

# Start API by default (python3 -m so uvicorn is found); override with: docker run -it ... bash
CMD ["python3", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]