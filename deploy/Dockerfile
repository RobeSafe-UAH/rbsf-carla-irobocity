FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG http_proxy

RUN apt-get update && apt-get install --reinstall -y locales && locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US
ENV LC_ALL=en_US.UTF-8
ENV TZ=Europe/Madrid

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all

ARG USER
ARG UID
ARG GID

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN groupadd -g $GID $USER && \
    useradd --uid $UID --gid $GID -m $USER && \
    apt-get update && \
    apt-get install -y sudo && \
    echo "$USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER && \
    chmod 0440 /etc/sudoers.d/$USER


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python-is-python3 \
    python3-pip \
    python3-setuptools \
    curl \
    wget \
    git \
    build-essential \
    ca-certificates \
    cmake \
    jupyter-notebook \
    figlet \
    strace \
    xdg-user-dirs \
    software-properties-common \
    # Graphics / OpenGL
    libgl1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglapi-mesa \
    libglu1-mesa \
    libglvnd0 \
    libglx0 \
    libegl1 \
    mesa-utils \
    mesa-vulkan-drivers \
    vulkan-validationlayers \
    # X11
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxcb1 \
    x11-xserver-utils \
    # Fonts
    fonts-liberation \
    fonts-dejavu \
    libfontconfig1 \
    libfreetype6 \
    # Misc
    libqt5gui5 \
    libsm6 \
    libgeos-dev \
    psmisc \
    lsof \
    sysstat \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 Humble
RUN add-apt-repository universe -y
RUN sudo apt update && sudo apt install curl -y
RUN curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.2.0/ros2-apt-source_1.2.0.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
RUN sudo dpkg -i /tmp/ros2-apt-source.deb
RUN apt update && apt upgrade -y
RUN apt install -y ros-humble-desktop ros-humble-tf-transformations ros-humble-vision-msgs python3-colcon-common-extensions

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_LINK_MODE=copy

USER $USER
WORKDIR /workspace

ENV CARLA_ROOT="/workspace/CARLA"
ENV TEAM_CODE_ROOT="/workspace/team_code"

ENV TERM=xterm-256color
RUN echo "PS1='\[\e[1;34m\]\u\[\e[1;37m\]@\[\e[1;33m\]\h\[\e[1;37m\]:\[\e[0;37m\]\w\[\e[1;37m\] >> \[\e[0m\]'" >> ~/.bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "alias RVIZ='rviz2 -d /workspace/rviz_cfg.rviz'" >> ~/.bashrc
RUN echo "alias BUILD='cd /workspace && colcon build && source install/setup.sh'" >> ~/.bashrc

CMD ["/bin/bash"]