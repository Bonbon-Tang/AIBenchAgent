#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import logging
import json
import time
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

    def start_existing_container(self, name):
        cmd = f"docker start {name}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            self.container_name = name
            self.resolve_container_id()
            return True, result.stdout.strip(), result.stderr.strip(), result.returncode
        return False, result.stdout.strip(), result.stderr.strip(), result.returncode

    def check_status(self):
        try:
            if not self.container_id:
                return False, "", "容器ID未设置", 1

            cmd = f"docker inspect -f '{{{{.State.Running}}}}' {self.container_id}"
            output, error, exit_code = self.execute(cmd)

            status = exit_code == 0 and "true" in output.lower()
            return status, output, error, exit_code
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
