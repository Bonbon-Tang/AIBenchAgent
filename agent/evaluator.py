#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import json
from typing import Dict, Any, Optional
from utils.llm import get_response_from_llm, extract_json_between_markers


class Evaluator:

    def __init__(self, llm, model_name, temperature: float = 0.7):
        self.llm = llm
        self.model = model_name
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

    def _get_system_prompt(self, command_type: str = "docker") -> str:
        if command_type == "docker":
            return """你是一名Docker专家。根据失败的Docker命令和错误信息，分析问题原因并给出改进建议。
请输出JSON格式的结果，包含以下字段：
- analysis: 问题的详细分析
- adjusted_command: 调整后的命令（保持相同的结构，只修改有问题的参数）
- is_recoverable: 是否可恢复（true/false）
- suggestion: 简短的改进建议"""
        else:
            return """你是一名Linux/Docker专家。根据失败的容器内执行命令和错误信息，分析问题原因并给出改进建议。
请输出JSON格式的结果，包含以下字段：
- analysis: 问题的详细分析
- adjusted_command: 调整后的命令
- is_recoverable: 是否可恢复（true/false）
- suggestion: 简短的改进建议"""

    def _build_prompt(
        self,
        failed_command: str,
        error_output: str,
        context: Dict[str, Any],
        command_type: str = "docker",
    ) -> str:
        prompt = f"""请分析以下失败的命令并给出改进建议。

失败的命令：
{failed_command}

错误输出：
{error_output}

"""
        if command_type == "docker":
            prompt += f"""配置信息：
- 芯片类型: {context.get('chip_type', '未知')}
- 应用场景: {context.get('application_scenario', '未知')}
- 任务类型: {context.get('task_type', '未知')}
- 卡数量: {context.get('card_count', '未指定')}
- 测试用例: {context.get('test_case', '未指定')}
"""
            if context.get('image_config'):
                prompt += f"""
镜像配置：
- 镜像名称: {context['image_config'].get('image_name', '未指定')}
- 启动命令模板: {context['image_config'].get('start_command', '未指定')}
"""
        else:
            prompt += f"""容器信息：
- 容器ID: {context.get('container_id', '未知')}
- 容器名称: {context.get('container_name', '未知')}
"""
        if context.get('local_memory'):
            prompt += "\n历史尝试记录（请参考这些信息避免重复错误）：\n"
            for i, attempt in enumerate(context['local_memory'][-3:], 1):
                prompt += f"  尝试 {i}:\n"
                prompt += f"    - 命令: {attempt.get('command', '无')}\n"
                prompt += f"    - 错误: {attempt.get('error', '无')[:200]}...\n"

        prompt += """
请以JSON格式返回结果，格式如下：
{
  "analysis": "问题分析",
  "adjusted_command": "调整后的命令",
  "is_recoverable": true,
  "suggestion": "改进建议"
}
"""
        return prompt

    def evaluate(
        self,
        failed_command: str,
        error_output: str,
        context: Dict[str, Any],
        command_type: str = "docker",
        msg_history: list = None,
    ) -> Dict[str, Any]:
        if msg_history is None:
            msg_history = []

        try:
            prompt = self._build_prompt(failed_command, error_output, context, command_type)

            text, msg_history = get_response_from_llm(
                prompt,
                client=self.llm,
                model=self.model,
                system_message=self._get_system_prompt(command_type),
                msg_history=msg_history,
                temperature=self.temperature,
            )

            json_data = extract_json_between_markers(text)

            if json_data is None:
                self.logger.warning("无法从LLM输出中提取JSON，将标记为不可恢复")
                return {
                    "analysis": "无法解析LLM响应",
                    "adjusted_command": failed_command,
                    "is_recoverable": False,
                    "suggestion": "请检查错误日志",
                    "msg_history": msg_history,
                }

            result = {
                "analysis": json_data.get("analysis", ""),
                "adjusted_command": json_data.get("adjusted_command", failed_command),
                "is_recoverable": json_data.get("is_recoverable", False),
                "suggestion": json_data.get("suggestion", ""),
                "msg_history": msg_history,
            }

            self.logger.info(f"评估完成: 可恢复={result['is_recoverable']}, 建议={result['suggestion']}")
            return result

        except Exception as e:
            self.logger.error(f"评估异常: {str(e)}")
            return {
                "analysis": f"评估异常: {str(e)}",
                "adjusted_command": failed_command,
                "is_recoverable": False,
                "suggestion": "评估过程出错，请检查日志",
                "msg_history": msg_history,
            }
