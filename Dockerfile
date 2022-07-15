FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
# Issue https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Install dependencies
RUN apt-get update && apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  curl \
  ca-certificates \
  dumb-init \
  htop \
  sudo \
  git \
  bzip2 \
  libx11-6 \
  locales \
  man \
  nano \
  git \
  procps \
  openssh-client \
  vim.tiny \
  lsb-release \
  python \
  python3-pip \
  libgl1-mesa-glx \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN sed -i "s/# ko_KR.UTF-8/ko_KR.UTF-8/" /etc/locale.gen \
  && locale-gen
ENV LANG=ko_KR.UTF-8

# Create project directory
RUN mkdir /projects

# Create a non-root user
RUN adduser --disabled-password --gecos '' --shell /bin/bash coder \
  && chown -R coder:coder /projects
RUN echo "coder ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-coder

# Install fixuid
ENV ARCH=amd64
RUN curl -fsSL "https://github.com/boxboat/fixuid/releases/download/v0.4.1/fixuid-0.4.1-linux-$ARCH.tar.gz" | tar -C /usr/local/bin -xzf - && \
  chown root:root /usr/local/bin/fixuid && \
  chmod 4755 /usr/local/bin/fixuid && \
  mkdir -p /etc/fixuid && \
  printf "user: coder\ngroup: coder\n" > /etc/fixuid/config.yml

# Install code-server
WORKDIR /tmp
ENV CODE_SERVER_VERSION=3.12.0
RUN curl -fOL https://github.com/cdr/code-server/releases/download/v${CODE_SERVER_VERSION}/code-server_${CODE_SERVER_VERSION}_${ARCH}.deb
RUN dpkg -i ./code-server_${CODE_SERVER_VERSION}_${ARCH}.deb && rm ./code-server_${CODE_SERVER_VERSION}_${ARCH}.deb

COPY ./entrypoint.sh /usr/bin/entrypoint.sh

# Switch to default user
USER coder
ENV USER=coder
ENV HOME=/home/coder
ENV PATH "$PATH:/home/coder/.local/bin"
# WORKDIR /home/coder
# RUN pip install 
WORKDIR /projects
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip install aim dagit dagster opencv-python

EXPOSE 8080 3000
ENTRYPOINT ["/usr/bin/entrypoint.sh", "--bind-addr", "0.0.0.0:8080", "--cert", "--disable-telemetry", "."]
