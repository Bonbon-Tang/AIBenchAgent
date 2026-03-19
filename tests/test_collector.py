#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pytest
from agent.collector import Collector
from utils.llm import create_client
from skills.skill_registry import SkillRegistry

def test_collector_from_config_path():
    client, client_model = create_client("claude-opus-4-6")
    tool_selector = SkillRegistry("./config/skills")
    agent = Collector(llm=client, modelname=client_model, tool_selector=tool_selector)

    try:
        result = agent.load_from_config("./config/task/h200_language_operator.json")
        print("\n收集结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        assert isinstance(result, dict)
        assert "image_config" in result, "应包含image_config"
        for field in Collector.REQUIRED_FIELDS:
            assert field in result, f"缺少必要字段: {field}"
    except Exception as e:
        pytest.fail(f"从配置文件加载失败: {str(e)}")


if __name__ == "__main__":
    test_collector_from_config_path()