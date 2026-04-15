#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import logging
import json
import time
import shlex
from .sandbox_interface import SandboxInterface

class DockerSandbox(SandboxInterface):

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.container_id = None
        self.container_name = None

    def execute(self, command, timeout=300):
        try:
            resolved_id = self.resolve_container_id()
            if resolved_id:
                full_command = f"docker exec {resolved_id} {command}"
            else:
                full_command = command
            self.logger.info(f"执行命令: {full_command}")
            result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            self.logger.error(f"命令执行超时({timeout}秒): {command}")
            return "", f"命令执行超时({timeout}秒)", 1
        except Exception as e:
            self.logger.error(f"命令执行失败: {str(e)}")
            return "", str(e), 1

    def upload_file(self, local_path, remote_path):
        try:
            resolved_id = self.resolve_container_id()
            if not resolved_id:
                self.logger.error("容器未创建，无法上传文件")
                return False, "", "容器未创建，无法上传文件", 1

            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                output, error, exit_code = self.execute(f"mkdir -p {remote_dir}")

            cmd = f"docker cp {local_path} {resolved_id}:{remote_path}"
            self.logger.info(f"上传文件: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            success = result.returncode == 0
            return success, result.stdout, result.stderr, result.returncode
        except Exception as e:
            self.logger.error(f"文件上传失败: {str(e)}")
            return False, "", str(e), 1

    def download_file(self, remote_path, local_path):
        try:
            resolved_id = self.resolve_container_id()
            if not resolved_id:
                self.logger.error("容器未创建，无法下载文件")
                return False, "", "容器未创建，无法下载文件", 1

            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)

            cmd = f"docker cp {resolved_id}:{remote_path} {local_path}"
            self.logger.info(f"下载文件: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            success = result.returncode == 0
            return success, result.stdout, result.stderr, result.returncode
        except Exception as e:
            self.logger.error(f"文件下载失败: {str(e)}")
            return False, "", str(e), 1

    def set_container(self, *, container_id=None, container_name=None):
        if container_id:
            self.container_id = container_id
        if container_name:
            self.container_name = container_name

    def resolve_container_id(self):
        target = self.container_id or self.container_name
        if not target:
            return None
        cmd = f"docker inspect -f '{{{{.Id}}}}' {target}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            self.container_id = result.stdout.strip()
            return self.container_id
        return None

    def container_exists(self, name):
        cmd = f"docker inspect {name}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0

    def get_container_state(self, name):
        cmd = f"docker inspect -f '{{{{.State.Status}}}}' {name}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return ""

    def is_container_running(self, name):
        return self.get_container_state(name) == "running"

    def start_existing_container(self, name):
        if self.is_container_running(name):
            self.container_name = name
            self.resolve_container_id()
            return True, name, "", 0

        cmd = f"docker start {name}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            self.container_name = name
            self.resolve_container_id()
            return True, result.stdout.strip(), result.stderr.strip(), result.returncode
        return False, result.stdout.strip(), result.stderr.strip(), result.returncode

    def create_container_from_config(self, image_config):
        image_name = image_config.get("image_name")
        if not image_name:
            return False, "", "缺少 image_name", 1

        container_name = image_config.get("container_name") or image_config.get("preferred_container_name")
        envs = image_config.get("environment") or {}
        volumes = image_config.get("volumes") or []
        gpus = image_config.get("gpus", "all")
        network_mode = image_config.get("network_mode", "host")
        ipc_mode = image_config.get("ipc_mode", "host")
        entrypoint = image_config.get("entrypoint") or "/bin/bash"
        keepalive_command = image_config.get("keepalive_command") or ["-lc", "tail -f /dev/null"]

        if isinstance(keepalive_command, str):
            keepalive_command = ["-lc", keepalive_command]

        cmd_parts = ["docker", "run", "-dit"]
        if container_name:
            cmd_parts.extend(["--name", container_name])
        if gpus:
            cmd_parts.extend(["--gpus", str(gpus)])
        if network_mode:
            cmd_parts.extend(["--network", str(network_mode)])
        if ipc_mode:
            cmd_parts.append(f"--ipc={ipc_mode}")
        if entrypoint:
            cmd_parts.extend(["--entrypoint", entrypoint])
        for k, v in envs.items():
            cmd_parts.extend(["-e", f"{k}={v}"])
        for volume in volumes:
            cmd_parts.extend(["-v", volume])
        cmd_parts.append(image_name)
        cmd_parts.extend([str(x) for x in keepalive_command])

        quoted = " ".join(shlex.quote(part) for part in cmd_parts)
        success, output, error, code = self.create_container(quoted)
        if success:
            self.container_name = container_name
        return success, output, error, code

    def check_status(self):
        try:
            resolved_id = self.resolve_container_id()
            if not resolved_id:
                return False, "", "容器ID未设置", 1

            cmd = f"docker inspect -f '{{{{.State.Running}}}}' {resolved_id}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            status = result.returncode == 0 and "true" in result.stdout.lower()
            return status, result.stdout, result.stderr, result.returncode
        except Exception as e:
            self.logger.error(f"状态检查失败: {str(e)}")
            return False, "", str(e), 1

    def create_container(self, docker_run_cmd):
        try:
            self.logger.info(f"创建容器: {docker_run_cmd}")
            result = subprocess.run(
                docker_run_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                self.logger.error(f"容器创建失败: {result.stderr}")
                return False, "", result.stderr, result.returncode

            if result.stdout:
                self.container_id = result.stdout.strip()
                self.logger.info(f"容器创建成功，ID: {self.container_id}")
                return True, self.container_id, "", result.returncode
            else:
                self.logger.error("无法获取容器ID")
                return False, "", "无法获取容器ID", 1
        except Exception as e:
            self.logger.error(f"容器创建异常: {str(e)}")
            return False, "", str(e), 1

    def remove_container(self):
        try:
            if not self.container_id:
                return True, "容器ID为空，无需删除", "", 0

            cmd = f"docker rm -f {self.container_id}"
            self.logger.info(f"删除容器: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            success = result.returncode == 0
            if success:
                self.container_id = None

            return success, result.stdout, result.stderr, result.returncode
        except Exception as e:
            self.logger.error(f"容器删除失败: {str(e)}")
            return False, "", str(e), 1
