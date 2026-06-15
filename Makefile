BASE_TAG := base
COURSE_TAG := carla
ROS_TAG := ros2_humble
IMAGE_NAME := curso_uah_irobocity
CONTAINER_NAME := $(IMAGE_NAME)_$(COURSE_TAG)_container

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
		--name $(CONTAINER_NAME) \
		-u $(UID):$(GID) \
		-v $(PWD):/workspace \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-e DISPLAY=$(DISPLAY) \
		-e TERM=xterm-256color \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		$(IMAGE_NAME):$(COURSE_TAG) \
		bash
endef

build_base:
	docker build . -f deploy/Dockerfile.base \
		-t $(IMAGE_NAME):$(BASE_TAG) \
		--build-arg USER=$(USER_NAME) \
		--build-arg UID=$(UID) \
		--build-arg GID=$(GID)

build: build_base
	docker build . -f deploy/Dockerfile.carla \
		-t $(IMAGE_NAME):$(COURSE_TAG) \
		--build-arg USER=$(USER_NAME) \
		--build-arg UID=$(UID) \
		--build-arg GID=$(GID)

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash -c "bash"

clean:
	rm -rf build/ install/ log/ robocity_carla.egg-info/

run:
	$(call run_docker)

run_ros:
	$(call run_docker,$(ROS_TAG), "bash")
