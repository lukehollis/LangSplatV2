FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set up a working directory
WORKDIR /app

# Install conda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

# Add conda to PATH
ENV PATH /opt/conda/bin:$PATH

# Copy the environment file and project files
COPY . .

# Create the conda environment
RUN conda env create -f environment_temp.yml

# Install PyTorch
RUN conda run -n langsplat_v2 pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install submodules
ENV TORCH_CUDA_ARCH_LIST="7.5"
RUN conda run -n langsplat_v2 pip install submodules/segment-anything-langsplat submodules/efficient-langsplat-rasterization submodules/simple-knn

# Set the default command to bash
CMD ["/bin/bash"]
