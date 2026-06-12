BASE_TAG := base
CARLA_TAG := carla
RACING_TAG := racing
ROS_TAG := ros2_humble
IMAGE_NAME := curso_carla_irobocity
CONTAINER := $(IMAGE_NAME)_container

UID := $(shell id -u)
GID := $(shell id -g)
USER_NAME := $(shell whoami)

define run_docker
	docker run -it --rm \
		--runtime=nvidia \
		--gpus all \
		--net host \
		--ipc host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--name $(CONTAINER) \
		-u $(UID):$(GID) \
		-v $(PWD):/workspace \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=$(DISPLAY) \
		-e TERM=xterm-256color \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		$(IMAGE_NAME):$(CARLA_TAG) \
		bash
endef

build_base:
	docker build . -f deploy/Dockerfile.base \
		-t $(IMAGE_NAME):$(BASE_TAG) \
		--build-arg USER=$(USER_NAME) \
		--build-arg UID=$(UID) \
		--build-arg GID=$(GID)

build_carla: build_base
	docker build . -f deploy/Dockerfile.carla \
		-t $(IMAGE_NAME):$(CARLA_TAG) \
		--build-arg USER=$(USER_NAME) \
		--build-arg UID=$(UID) \
		--build-arg GID=$(GID)

build_racing: build_base
	docker build . -f deploy/Dockerfile.racing \
		-t $(IMAGE_NAME):$(RACING_TAG) \
		--build-arg USER=$(USER_NAME) \
		--build-arg UID=$(UID) \
		--build-arg GID=$(GID)

attach:
	docker exec -it $(IMAGE_NAME)_container /bin/bash -c "bash"

clean:
	rm -rf build/ install/ log/ robocity_carla.egg-info/

run:
	$(call run_docker)

run_ros:
	$(call run_docker,$(ROS_TAG), "bash")
