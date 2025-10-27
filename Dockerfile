# Use the official NVIDIA CUDA base image for Ubuntu 20.04
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV TZ=Etc/UTC
ENV DEBIAN_FRONTEND=noninteractive

# Install a modern C++ toolchain (g++-11) to provide the required libstdc++.so.6,
# and other necessary tools like wget.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update && \
    apt-get install -y gcc-11 g++-11 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Download the required SAM model checkpoint before copying project files
RUN mkdir -p ckpts && \
    wget -O ckpts/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH

# Copy the environment file and project files
COPY . .

# Create the conda environment from the temp file
RUN conda env create -f environment_temp.yml

# Install PyTorch and submodules into the created environment
ENV TORCH_CUDA_ARCH_LIST="8.9"
SHELL ["conda", "run", "-n", "langsplat_v2", "/bin/bash", "-c"]
RUN pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install submodules/segment-anything-langsplat submodules/efficient-langsplat-rasterization submodules/simple-knn

# Unset the shell to return to default
SHELL ["/bin/bash", "-c"]

# Set the default command to bash
CMD ["/bin/bash"]