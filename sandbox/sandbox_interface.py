#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging

class SandboxInterface(ABC):

    @abstractmethod
    def execute(self, command):
        pass

    @abstractmethod
    def upload_file(self, local_path, remote_path):
        pass

    @abstractmethod
    def download_file(self, remote_path, local_path):
        pass

    @abstractmethod
    def check_status(self):
        pass
