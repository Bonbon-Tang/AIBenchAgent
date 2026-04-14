#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import os
import uuid
import logging
import re
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
from adapters.projectten_v2 import ProjectTenV2Adapter


def extract_json_between_markers(llm_output):
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue

    return None

class Collector:

    REQUIRED_FIELDS = [
        "application_scenario",
        "task_scenario",
        "chip_type",
        "card_count",
        "test_case",
    ]

    def __init__(self, llm, modelname, tool_selector):
        if tool_selector is None:
            raise ValueError("tool_selector is required")
        self.llm = llm
        self.model = modelname
        self.tool_selector = tool_selector
        self.logger = logging.getLogger(__name__)
        self.system_prompt = self.get_default_system_prompt()
        self.max_interaction = 10
        self.projectten_v2_adapter = ProjectTenV2Adapter()
    
    def get_default_system_prompt(self) -> str:
        return """
你是一个专业的AI模型评测助手，负责收集用户的评测需求信息。你需要引导用户提供完整的评测配置信息。

当前系统支持的配置选项：

支持的应用场景：
{available_scenarios}

支持的芯片类型：
{available_chips}

支持的任务类型：
{available_tasks}

你需要收集以下信息：
1. application_scenario: 应用场景（需要从支持的场景中选择：{scenario_options}）
2. task_scenario: 任务类型（需要从支持的任务中选择：{task_scenario_options}）
3. chip_type: 芯片类型（需要从支持的芯片中选择：{chip_type_options}）
4. card_count: GPU卡数
5. test_case: 需要测试的模型名或者算子名

请通过对话的方式，逐步引导用户提供这些信息，确保信息完整和准确。
特别注意：应用场景、芯片类型和任务类型必须从支持的列表中选择。

Respond in the format:
THOUGHT:
<THOUGHT>

JSON:
```json
<JSON>
```

JSON的字段必须包含以上5个必须收集的信息，且格式必须严格按照JSON格式。并且增加建议字段：如果信息已经完整获取，此字段设置为"I am done"；否则设置为引导用户补充缺失的信息。
"""

    def get_dynamic_system_prompt(self) -> str:
        template = self.get_default_system_prompt()

        if not self.tool_selector:
            raise ValueError("tool_selector is required")
        
        scenarios = self.tool_selector.get_available_application_scenarios()
        chips = self.tool_selector.get_available_chips()
        tasks = self.tool_selector.get_available_task_types()

        scenario_descriptions = []
        for scenario in scenarios:
            scenario_descriptions.append(f"- {scenario['name']}: {scenario['description']}")
        available_scenarios = '\n'.join(scenario_descriptions)

        chip_descriptions = []
        for chip in chips:
            chip_descriptions.append(f"- {chip['name']}: {chip['description']}")
        available_chips = '\n'.join(chip_descriptions)

        task_descriptions = []
        for task in tasks:
            task_descriptions.append(f"- {task['name']}: {task['description']}")
        available_tasks = '\n'.join(task_descriptions)

        scenario_options = ", ".join([scenario['name'] for scenario in scenarios])
        chip_type_options = ", ".join([chip['name'] for chip in chips])
        task_scenario_options = ", ".join([task['name'] for task in tasks])

        dynamic_prompt = template.format(
            available_scenarios=available_scenarios,
            available_chips=available_chips,
            available_tasks=available_tasks,
            scenario_options=scenario_options,
            chip_type_options=chip_type_options,
            task_scenario_options=task_scenario_options
        )
        
        return dynamic_prompt

    def build_prompt(self, answer: str) -> str:
        prompt = f"""
{self.get_dynamic_system_prompt()}

当前用户的输入：
"""
        prompt += answer
        return prompt
    
    def collect_user_info(self, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = initial_context or {}

        if "task_id" not in context:
            context["task_id"] = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        print("欢迎使用自动化评测Agent系统！")
        print("我将帮助您配置评测任务，请按照提示回答问题。")
        print("信息收集过程中，我会根据您的回答逐步完善配置信息。")
        print()

        if context:
            print("当前已有配置信息：")
            for key, value in context.items():
                if key != "task_id":
                    print(f"- {key}: {value}")
            print()

        msg_history = []

        if context:
            context_str = json.dumps(context, ensure_ascii=False, indent=2)
            msg_history.append({"role": "system", "content": f"当前配置信息：\n{context_str}"})

        msg_history.append({"role": "system", "content": self.get_dynamic_system_prompt()})

        num_interaction = self.max_interaction
        for j in range(num_interaction):
            print(f"交互轮次: {j + 1}/{num_interaction}")

            answer = input("请补充评测要求\n").strip()

            prompt = self.build_prompt(answer)

            from utils.llm import get_response_from_llm
            text, msg_history = get_response_from_llm(
                prompt,
                client=self.llm,
                model=self.model,
                system_message=self.get_dynamic_system_prompt(),
                msg_history=msg_history,
                temperature=0.7,
            )
            json_data = extract_json_between_markers(text)
            self.logger.debug(f"LLM extracted JSON: {json_data}")
            if json_data is None:
                self.logger.warning("无法从LLM输出中提取JSON")
                continue

            if "I am done" in text:
                self.logger.info(f"信息收集在 {j + 1} 轮后完成。")
                self.logger.debug(f"最终配置: {json_data}")
                json_data = self._attach_image_config(json_data)
                return json_data

        print("信息收集不完整，请重新运行。")
        return {}

    def _attach_image_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        task_type = config.get("task_type", config.get("task_scenario"))
        try:
            image_config = self.tool_selector.get_image_config(
                config["application_scenario"], config["chip_type"], task_type
            )
            config["image_config"] = image_config
        except (ValueError, KeyError) as e:
            self.logger.warning(f"获取镜像配置失败: {str(e)}")
        return config

    def load_from_config(self, config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if self.projectten_v2_adapter.is_projectten_v2_config(config):
            config = self.projectten_v2_adapter.normalize(config)
            self.logger.info(f"检测到 ProjectTen v2 配置，已转换为 AIBench 内部配置: {config_path}")

        missing = [field for field in self.REQUIRED_FIELDS if not config.get(field)]
        if missing:
            raise ValueError(
                f"配置文件缺少必要字段: {', '.join(missing)}。"
                f"必须包含: {', '.join(self.REQUIRED_FIELDS)}"
            )

        if "task_type" not in config and "task_scenario" in config:
            config["task_type"] = config["task_scenario"]

        if "task_id" not in config:
            config["task_id"] = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        task_type = config.get("task_type", config.get("task_scenario"))
        if not self.tool_selector.validate_config(
            config["application_scenario"], config["chip_type"], task_type
        ):
            raise ValueError(
                f"不支持的配置组合: 场景={config['application_scenario']}, "
                f"芯片={config['chip_type']}, 任务={task_type}"
            )

        if not config.get("image_config"):
            config = self._attach_image_config(config)

        self.logger.info(f"从配置文件加载评测配置: {config_path}")
        return config
    