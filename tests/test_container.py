#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pytest
from utils.llm import create_client
from skills.skill_registry import SkillRegistry
from agent.creator import Creator
import logging
logging.basicConfig(level=logging.INFO)
def test_container():
    client, client_model = create_client("claude-opus-4-6")
    tool_selector = SkillRegistry("./config/skills")
    image_config = tool_selector.get_image_config("language", "NVIDIA_H200", "operator")
    agent = Creator(llm=client, model_name=client_model)
    config = {
            "task_id": "eval_20260316_154521_df8e6734",
            "application_scenario": "language",
            "task_scenario": "operator",
            "task_type": "operator",
            "chip_type": "NVIDIA_H200",
            "card_count": 8,
            "test_case": "gemm",
            "suggestion": "I am done",
            "image_config": image_config,
    }
    try:
        _, result = agent.create_container(config)
        print("\n收集结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        assert isinstance(result, dict)
    except Exception as e:
        pytest.fail(f"执行容器任务失败: {str(e)}")

if __name__ == "__main__":
    test_container()