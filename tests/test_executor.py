#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pytest
from utils.llm import create_client
from agent.executor import Executor
import logging
logging.basicConfig(level=logging.INFO)

def test_executor_run():
    client, client_model = create_client("claude-opus-4-6")
    executor = Executor(llm=client, model_name=client_model)

    context = {
  "task_id": "eval_20260316_154521_df8e6734",
  "application_scenario": "language",
  "task_scenario": "operator",
  "task_type": "operator",
  "chip_type": "NVIDIA_H200",
  "card_count": 8,
  "test_case": "gemm",
  "suggestion": "I am done",
  "image_config": {
    "image_name": "registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:operate",
    "start_command": "docker run -d --gpus all --shm-size=10g ",
    "task_command": "cd speed_test/cuda_ops && mkdir -p build && cd build && cmake -DCUDNN_INCLUDE_DIR=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/include -DCUDNN_LIBRARIES=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib/libcudnn.so.9  .. && make && cd ../../ && python test_conv.py conv_f16.csv 16 0",
    "description": "",
    "start_command_hints": "映射/mnt/nvme1n1/dongkaixing到容器中，工作目录切换到/mnt/nvme1n1/dongkaixing/sglang/3rdparty/AIChipBenchmark/operators",
    "task_command_hints": "在容器内先cd到/mnt/nvme1n1/dongkaixing/sglang/3rdparty/AIChipBenchmark/operators目录，再执行编译和测试命令。cmake构建在speed_test/cuda_ops/build目录下。",
    "environment": {},
    "volumes": []
  },
  "local_memory": [],
  "generated_command": "docker run -d --gpus all --shm-size=10g -v /mnt/nvme1n1/dongkaixing:/mnt/nvme1n1/dongkaixing -w /mnt/nvme1n1/dongkaixing/sglang/3rdparty/AIChipBenchmark/operators --name 20231012_123456_abc123 registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:operate tail -f /dev/null",
  "generated_container_name": "20231012_123456_abc123",
  "container_id": "3473fe56bbf3",
  "container_name": "20231012_123456_abc123",
  "docker_command": "docker run -d --gpus all --shm-size=10g -v /mnt/nvme1n1/dongkaixing:/mnt/nvme1n1/dongkaixing -w /mnt/nvme1n1/dongkaixing/sglang/3rdparty/AIChipBenchmark/operators --name 20231012_123456_abc123 registry.h.pjlab.org.cn/ailab-sys-sys_gpu/nemo:operate tail -f /dev/null",
  "attempts": 1
}
    try:
        success, result = executor.run(context)
        print("\n执行结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        assert isinstance(result, dict)
    except Exception as e:
        pytest.fail(f"Executor.run执行失败: {str(e)}")


if __name__ == "__main__":
    test_executor_run()
