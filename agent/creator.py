#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import random
import string
import logging
from typing import Dict, Any, Tuple, Optional, List
from utils.llm import get_response_from_llm, extract_json_between_markers
from sandbox.docker_sandbox import DockerSandbox
from .base import EvalRetryAgent


class Creator(EvalRetryAgent):

    def get_default_system_prompt(self, context: Dict[str, Any]) -> str:
        application_scenario = context.get('application_scenario', '未知')
        task_scenario = context.get('task_scenario', context.get('task_type', '未知'))
        chip_type = context.get('chip_type', '未知')
        card_count = context.get('card_count', '未指定')
        test_case = context.get('test_case', '未指定')
        image_config = context.get('image_config', {})

        start_command_hints = image_config.get('start_command_hints', '')
        environment = image_config.get('environment', {})
        volumes = image_config.get('volumes', [])

        system_prompt = f"""你是一个专业的Docker容器管理专家，负责根据评测配置信息创建Docker容器。

当前固定配置信息：
- 应用场景: {application_scenario}
- 任务场景: {task_scenario}
- 芯片类型: {chip_type}
- 卡数量: {card_count}
- 测试用例: {test_case}
"""

        if start_command_hints:
            system_prompt += f"\n容器启动提示（请参考以下信息生成更准确的docker run命令）：\n{start_command_hints}\n"

        system_prompt += """
请根据用户提供的配置信息生成一个完整的Docker运行命令，命令中需要注意是否需要包含：
1. GPU设备映射（例如--gpus all或指定具体卡，或昇腾芯片的--device映射）
2. 必要的目录挂载（如代码、数据、输出目录）
3. 环境变量（如模型参数、数据集路径）
4. 容器名称（建议使用时间戳+随机后缀避免冲突）
5. 基础镜像（使用上面指定的镜像名称）
6. 保持容器运行的命令（如tail -f /dev/null）
"""

        if environment:
            env_str = ", ".join(f"{k}={v}" for k, v in environment.items())
            system_prompt += f"\n建议的环境变量: {env_str}\n"

        if volumes:
            system_prompt += f"\n建议的挂载目录: {', '.join(volumes)}\n"

        system_prompt += "\n请以JSON格式返回结果，包含command和container_name字段。\n"

        return system_prompt

    def build_prompt(self, context: Dict[str, Any]) -> str:
        chip_type = context.get('chip_type', '未知')
        application_scenario = context.get('application_scenario', '未知')
        task_scenario = context.get('task_scenario', context.get('task_type', '未知'))
        card_count = context.get('card_count', '未指定')
        test_case = context.get('test_case', '未指定')
        image_config = context.get('image_config', {})
        local_memory = context.get('local_memory', [])

        prompt = """用户配置信息：
"""
        prompt += f"- 芯片类型: {chip_type}\n"
        prompt += f"- 应用场景: {application_scenario}\n"
        prompt += f"- 任务场景: {task_scenario}\n"
        prompt += f"- 卡数量: {card_count}\n"
        prompt += f"- 测试用例: {test_case}\n"

        if image_config:
            prompt += f"- 镜像名称: {image_config.get('image_name', '未指定')}\n"
            prompt += f"- 启动命令模板: {image_config.get('start_command', '未指定')}\n"

            if image_config.get('start_command_hints'):
                prompt += f"- 容器启动提示: {image_config['start_command_hints']}\n"
            if image_config.get('environment'):
                env_str = ", ".join(f"{k}={v}" for k, v in image_config['environment'].items())
                prompt += f"- 建议环境变量: {env_str}\n"
            if image_config.get('volumes'):
                prompt += f"- 建议挂载目录: {', '.join(image_config['volumes'])}\n"

        for key, value in context.items():
            if key not in ['chip_type', 'application_scenario', 'task_scenario', 'task_type',
                          'card_count', 'test_case', 'image_config', 'local_memory']:
                if key not in ['task_id', 'suggestion']:
                    prompt += f"- {key}: {value}\n"

        if local_memory:
            prompt += "\n历史尝试记录（请参考这些信息避免重复错误）：\n"
            for i, attempt in enumerate(local_memory, 1):
                prompt += f"  尝试 {i}:\n"
                prompt += f"    - Docker命令: {attempt.get('docker_command', '无')}\n"
                prompt += f"    - 分析: {attempt.get('analysis', '无')}\n"
                prompt += f"    - 错误信息: {attempt.get('error', '无')[:200]}...\n"

        prompt += """
注意：请使用镜像配置中提供的启动命令模板作为基础，但是根据需要，添加或者调整参数。

请以JSON格式返回结果，必须包含command和container_name字段。
JSON:
```json
<JSON>
```
"""
        return prompt

    def execute_container_creation(self, context: Dict[str, Any], msg_history: List = None) -> Tuple[bool, Dict[str, Any]]:
        if msg_history is None:
            msg_history = []

        try:
            prompt = self.build_prompt(context)

            self.logger.debug(f"生成容器创建命令的提示词: {prompt[:200]}...")
            text, msg_history = get_response_from_llm(
                prompt,
                client=self.llm,
                model=self.model,
                system_message=self.get_default_system_prompt(context),
                msg_history=msg_history,
                temperature=0.7,
            )
            self.logger.debug(f"LLM response text: {text[:300]}...")

            json_data = extract_json_between_markers(text)

            if json_data is None:
                self.logger.error("无法从LLM输出中提取JSON")
                return False, {"error": "无法从LLM输出中提取JSON", "msg_history": msg_history}

            success, result = self.parse_response(json_data, context)

            if success:
                result["msg_history"] = msg_history
                return True, result
            else:
                self.logger.warning(f"命令执行失败: {result.get('error', str(result))}")
                result["msg_history"] = msg_history
                return False, result

        except Exception as e:
            self.logger.error(f"执行异常: {str(e)}")
            return False, {"error": f"执行异常: {str(e)}", "msg_history": msg_history}

    def validate_docker_command(self, command: str, container_name: str) -> Tuple[bool, str]:
        if not command.strip().startswith("docker run"):
            return False, "命令必须以'docker run'开头"

        import re
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9_.-]*$', container_name):
            return False, f"容器名称包含非法字符: {container_name}"

        dangerous_options = [
            "--privileged",
            "--cap-add=ALL",
            "--security-opt",
            "--device=/dev",
            "--volume=/:/host",
            "--network=host"
        ]

        for option in dangerous_options:
            if option in command:
                return False, f"命令包含危险选项: {option}"

        injection_chars = [";", "&&", "||", "`", "$(", ">", "<", "|"]
        for char in injection_chars:
            if char in command and not (char in container_name or char in "docker run"):
                return False, f"命令可能包含shell注入字符: {char}"

        return True, ""

    def parse_response(self, response: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        try:
            if "error" in response:
                return False, {"error": response.get("error", "未知错误")}

            command = response.get("command", "")
            container_name = response.get("container_name", "")
            self.logger.debug(f"command: {command}, container_name: {container_name}")

            if not command or not container_name:
                return False, {"error": "响应缺少command或container_name字段"}

            is_safe, safety_error = self.validate_docker_command(command, container_name)
            if not is_safe:
                return False, {"error": f"命令安全性验证失败: {safety_error}", "command": command}

            context["generated_command"] = command
            context["generated_container_name"] = container_name

            try:
                self.logger.info(f"执行Docker命令: {command}")
                output, error_output, exit_code = self.sandbox.execute(command)

                if exit_code == 0:
                    if "docker run" in command:
                        get_id_cmd = f"docker ps -q -f name={container_name}"
                        container_id_output, container_error, exit_code_id = self.sandbox.execute(get_id_cmd)

                        if exit_code_id == 0 and container_id_output.strip():
                            container_id = container_id_output.strip()
                            context["container_id"] = container_id
                            context["container_name"] = container_name
                            context["docker_command"] = command

                            self.logger.info(f"容器创建成功，ID: {container_id}")
                            return True, context
                        else:
                            return False, {"error": f"容器创建成功但无法获取容器ID: {container_id_output}", "container_name": container_name}
                    else:
                        context["docker_command"] = command
                        return True, context
                else:
                    error_msg = f"Docker命令执行失败: {error_output if error_output else output}"
                    context["docker_command"] = command
                    context["error_output"] = error_output
                    context["std_output"] = output
                    return False, {"error": error_msg, "exit_code": exit_code}

            except Exception as e:
                error_msg = f"执行Docker命令异常: {str(e)}"
                context["docker_command"] = command
                context["exception"] = str(e)
                return False, {"error": error_msg}

        except Exception as e:
            self.logger.error(f"响应解析失败: {str(e)}")
            return False, {"error": f"响应解析失败: {str(e)}"}

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        required_fields = ['chip_type', 'application_scenario', 'task_type']
        for field in required_fields:
            if field not in config:
                return False, f"缺少必要字段: {field}"
            if not config[field]:
                return False, f"字段不能为空: {field}"

        valid_chips = ["NVIDIA_H200", "Ascend_910B"]
        if config.get('chip_type') not in valid_chips:
            return False, f"不支持的芯片类型: {config.get('chip_type')}"

        if 'card_count' in config and config['card_count']:
            try:
                card_count = int(config['card_count'])
                if card_count < 0:
                    return False, f"卡数量必须为非负整数: {config['card_count']}"
            except ValueError:
                return False, f"卡数量必须是整数: {config['card_count']}"
        return True, "validate_config passed"

    def create_container(self, config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        is_valid, validation_error = self.validate_config(config)
        if not is_valid:
            self.logger.error(f"配置验证失败: {validation_error}")
            return False, {"error": f"配置验证失败: {validation_error}"}

        chip_type = config.get('chip_type')
        application_scenario = config.get('application_scenario')
        task_type = config.get('task_type')

        image_config = config.get('image_config')
        if not image_config:
            self.logger.error("配置中缺少image_config")
            return False, {"error": "配置中缺少image_config，请确保Collector已提供镜像配置"}
        self.logger.info(f"获取到镜像配置: {image_config.get('image_name', '未知')}")

        # Creator 特有的回调
        def prepare_context(ctx, local_memory):
            context = ctx.copy()
            context.update({
                'chip_type': chip_type,
                'application_scenario': application_scenario,
                'task_type': task_type,
                'image_config': image_config,
                'local_memory': local_memory.copy(),
            })
            return context

        def get_failed_command(attempt_ctx, result):
            return attempt_ctx.get('generated_command', '')

        def on_success(result, attempt):
            self.logger.info(f"容器创建成功，容器ID: {result.get('container_id', '未知')}")
            result.update({
                'chip_type': chip_type,
                'application_scenario': application_scenario,
                'task_type': task_type,
                'image_config': image_config,
                'attempts': attempt,
            })
            return result

        def on_failure_record(attempt, cmd, error, eval_result):
            record = {
                'attempt': attempt,
                'docker_command': cmd,
                'error': error,
            }
            if eval_result:
                record.update({
                    'analysis': eval_result.get('analysis', ''),
                    'suggestion': eval_result.get('suggestion', ''),
                    'adjusted_command': eval_result.get('adjusted_command', ''),
                    'is_recoverable': eval_result.get('is_recoverable', False),
                })
            else:
                record.update({'analysis': '', 'suggestion': ''})
            return record

        def on_unrecoverable(error, analysis, ctx, attempt, memory):
            return {
                "error": f"容器创建失败，不可恢复: {error}",
                "analysis": analysis,
                "chip_type": chip_type,
                "application_scenario": application_scenario,
                "task_type": task_type,
                "attempts": attempt,
                "local_memory": memory,
            }

        def on_max_retries(ctx, attempt, memory):
            return {
                "error": f"容器创建失败，已达到最大尝试次数 {self.max_retries}",
                "chip_type": chip_type,
                "application_scenario": application_scenario,
                "task_type": task_type,
                "image_config": image_config,
                "attempts": attempt,
                "local_memory": memory,
            }

        def build_feedback(cmd, error, analysis, suggestion):
            return (
                f"上一次生成的命令执行失败。\n"
                f"失败命令: {cmd}\n"
                f"错误信息: {error}\n"
                f"错误分析: {analysis}\n"
                f"改进建议: {suggestion}\n"
                f"请根据以上反馈重新生成正确的Docker命令。"
            )

        return self.execute_with_retry(
            action_fn=self.execute_container_creation,
            context=config,
            command_type="docker",
            prepare_context=prepare_context,
            get_failed_command=get_failed_command,
            on_success=on_success,
            on_failure_record=on_failure_record,
            on_unrecoverable=on_unrecoverable,
            on_max_retries=on_max_retries,
            build_feedback=build_feedback,
            retry_delay=2.0,
        )

    def execute_in_container(
        self,
        container_id: str,
        task_command: str,
        context: Dict[str, Any],
        command_type: str = "bash"
    ) -> Tuple[bool, Dict[str, Any]]:
        """在容器内执行命令，委托给基类 retry_command。"""
        return self.retry_command(
            container_id=container_id,
            initial_command=task_command,
            context=context,
            command_type="exec",
        )

    def cleanup_container(self) -> bool:
        try:
            if self.sandbox.container_id:
                self.logger.info("正在清理容器...")
                success = self.sandbox.remove_container()
                if success:
                    self.logger.info("容器清理完成")
                return success
            return True
        except Exception as e:
            self.logger.error(f"容器清理失败: {str(e)}")
            return False