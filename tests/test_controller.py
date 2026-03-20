#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pytest
from agent.controller import EvaluationController
import logging
logging.basicConfig(level=logging.INFO)

def test_controller_with_config_path():
    controller = EvaluationController(
        model_name="claude-opus-4-6", skill_config_dir="./config/skills"
    )

    try:
        success, result = controller.run_evaluation(
            config_path="./config/task/h200_language_operator.json"
        )
        print("\n评测结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        assert isinstance(result, dict)
        if success:
            assert "evaluation_results" in result
            print(f"\n评测成功! 容器ID: {result.get('container_id', '未知')}")
        else:
            print(f"\n评测失败: {result.get('error', '未知错误')}")
    except Exception as e:
        pytest.fail(f"EvaluationController执行失败: {str(e)}")


def test_controller_with_config_dict():
    controller = EvaluationController(
        model_name="claude-opus-4-6", skill_config_dir="./config/skills"
    )

    config = {
        "application_scenario": "language",
        "task_scenario": "operator",
        "task_type": "operator",
        "chip_type": "NVIDIA_H200",
        "card_count": 8,
        "test_case": "gemm",
    }

    try:
        success, result = controller.run_evaluation(config=config, interactive=False)
        print("\n评测结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        assert isinstance(result, dict)
        if success:
            assert "evaluation_results" in result
            print(f"\n评测成功! 容器ID: {result.get('container_id', '未知')}")
        else:
            print(f"\n评测失败: {result.get('error', '未知错误')}")
    except Exception as e:
        pytest.fail(f"EvaluationController执行失败: {str(e)}")


if __name__ == "__main__":
    test_controller_with_config_path()
