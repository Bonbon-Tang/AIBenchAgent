#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Dict, Any, Tuple, List
from .base import EvalRetryAgent
from .templates import build_service_eval_script


class Executor(EvalRetryAgent):

    def get_default_system_prompt(self, context: Dict[str, Any]) -> str:
        application_scenario = context.get('application_scenario', '未知')
        task_scenario = context.get('task_scenario', context.get('task_type', '未知'))
        chip_type = context.get('chip_type', '未知')
        card_count = context.get('card_count', '未指定')
        test_case = context.get('test_case', '未指定')
        image_config = context.get('image_config', {})

        task_command_hints = image_config.get('task_command_hints', '')

        system_prompt = f"""你是一个专业的AI模型评测专家，负责根据评测配置信息生成评测脚本。

当前固定配置信息：
- 应用场景: {application_scenario}
- 任务场景: {task_scenario}
- 芯片类型: {chip_type}
- 卡数量: {card_count}
- 测试用例: {test_case}
"""

        if task_command_hints:
            system_prompt += f"\n评测脚本提示（请参考以下信息生成更准确的评测脚本）：\n{task_command_hints}\n"

        system_prompt += """
请根据用户提供的配置信息生成一个完整的评测脚本，命令中需要注意是否需要包含：
1. 数据准备（如数据集路径检查、下载）
2. 运行评测命令（例如训练脚本、推理脚本）
3. 结果输出格式约定（如JSON文件、关键指标打印）
4. 错误处理和日志记录

脚本必须以 #!/bin/bash 开头，并且创建 /workspace/results 目录用于存放结果。
最终结果需写入 result.json 文件。

如果 image_config 中包含 service_profile：
- 优先采用“tangyufeng 风格”两阶段流程：先确保工作容器环境可用，再在容器内启动/检查模型服务，最后对 OpenAI 兼容接口做评测。
- 优先检查 healthcheck_path 对应接口是否已经可用；如果已可用，不要重复启动服务。
- 如果未就绪，再按 service_profile 中的 env 和 serve_command 启动服务。
- 结果中至少包含 smoke_passed、success_count、fail_count、avg_latency_ms、p95_latency_ms。

请以JSON格式返回结果，包含 script_content 和 script_name 字段。
"""
        return system_prompt

    def build_prompt(self, context: Dict[str, Any]) -> str:
        chip_type = context.get('chip_type', '未知')
        application_scenario = context.get('application_scenario', '未知')
        task_scenario = context.get('task_scenario', context.get('task_type', '未知'))
        card_count = context.get('card_count', '未指定')
        test_case = context.get('test_case', '未指定')
        image_config = context.get('image_config', {})
        local_memory = context.get('local_memory', [])

        prompt = "用户配置信息：\n"
        prompt += f"- 芯片类型: {chip_type}\n"
        prompt += f"- 应用场景: {application_scenario}\n"
        prompt += f"- 任务场景: {task_scenario}\n"
        prompt += f"- 卡数量: {card_count}\n"
        prompt += f"- 测试用例: {test_case}\n"

        if image_config:
            prompt += f"- 镜像名称: {image_config.get('image_name', '未指定')}\n"
            prompt += f"- 任务命令模板: {image_config.get('task_command', '未指定')}\n"

            if image_config.get('task_command_hints'):
                prompt += f"- 评测脚本提示: {image_config['task_command_hints']}\n"
            if image_config.get('service_profile'):
                prompt += f"- 服务配置: {image_config['service_profile']}\n"
            if image_config.get('container_name'):
                prompt += f"- 优先复用容器: {image_config.get('container_name')}\n"

        skip_keys = {
            'chip_type', 'application_scenario', 'task_scenario', 'task_type',
            'card_count', 'test_case', 'image_config', 'local_memory',
            'task_id', 'suggestion', 'container_id', 'container_name',
            'script_content', 'script_name', 'script_path',
            'generated_command', 'generated_container_name',
            'docker_command','attempts',
        }
        for key, value in context.items():
            if key not in skip_keys:
                prompt += f"- {key}: {value}\n"

        if local_memory:
            prompt += "\n历史尝试记录（请参考这些信息避免重复错误）：\n"
            for i, attempt in enumerate(local_memory, 1):
                prompt += f"  尝试 {i}:\n"
                prompt += f"    - 脚本名称: {attempt.get('script_name', '无')}\n"
                prompt += f"    - 分析: {attempt.get('analysis', '无')}\n"
                prompt += f"    - 错误信息: {attempt.get('error', '无')[:200]}...\n"

        prompt += """
请根据配置信息生成一个完整的评测脚本。

脚本要求：
1. 以 #!/bin/bash 开头
2. 创建 /workspace/results 目录（mkdir -p /workspace/results）
3. 包含数据准备、评测执行、结果收集三个阶段

请以JSON格式返回结果，必须包含 script_content 和 script_name 字段。
JSON:
```json
<JSON>
```
"""
        return prompt

    def validate_script(self, script_content: str) -> Tuple[bool, str]:
        if not script_content or not script_content.strip():
            return False, "脚本内容为空"

        required_elements = [
            ("#!/bin/bash", "缺少 shebang 行 (#!/bin/bash)"),
            ("mkdir -p /workspace/results", "缺少创建结果目录命令 (mkdir -p /workspace/results)"),
            ("result.json", "缺少结果文件引用 (result.json)"),
        ]

        for element, error_msg in required_elements:
            if element not in script_content:
                return False, error_msg

        return True, ""

    def generate_script(self, context: Dict[str, Any], msg_history: List = None) -> Tuple[bool, Dict[str, Any]]:
        if msg_history is None:
            msg_history = []

        image_config = context.get('image_config', {})
        if image_config.get('service_profile'):
            script_name, script_content = build_service_eval_script(context)
            is_valid, validation_error = self.validate_script(script_content)
            if not is_valid:
                return False, {
                    'error': f'模板脚本验证失败: {validation_error}',
                    'msg_history': msg_history,
                }
            self.logger.info(f"使用 service_profile 固定模板生成评测脚本: {script_name}")
            return True, {
                'script_content': script_content,
                'script_name': script_name,
                'attempts': 1,
                'msg_history': msg_history,
            }

        max_format_retries = 3
        format_errors = []

        for attempt in range(1, max_format_retries + 1):
            self.logger.info(f"尝试生成评测脚本，第 {attempt} 次尝试...")

            try:
                prompt = self.build_prompt(context)
                self.logger.debug(f"attempt {attempt}, prompt: {prompt[:200]}...")

                from utils.llm import get_response_from_llm, extract_json_between_markers
                text, msg_history = get_response_from_llm(
                    prompt,
                    client=self.llm,
                    model=self.model,
                    system_message=self.get_default_system_prompt(context),
                    msg_history=msg_history,
                    temperature=0.7,
                )

                json_data = extract_json_between_markers(text)

                if json_data is None:
                    error_msg = "无法从LLM输出中提取JSON"
                    self.logger.error(error_msg)
                    format_errors.append(error_msg)
                    msg_history.append({"role": "user", "content": f"你的输出格式不正确：{error_msg}。请严格按照JSON格式返回结果，包含 script_content 和 script_name 字段。"})
                    continue

                script_content = json_data.get('script_content', '')
                script_name = json_data.get('script_name', 'evaluation_script.sh')

                is_valid, validation_error = self.validate_script(script_content)
                if not is_valid:
                    error_msg = f"脚本验证失败: {validation_error}"
                    self.logger.warning(error_msg)
                    format_errors.append(error_msg)
                    msg_history.append({"role": "user", "content": f"生成的脚本验证失败：{validation_error}。请修正脚本内容后重新返回。"})
                    continue

                self.logger.info(f"评测脚本生成成功: {script_name}")
                return True, {
                    'script_content': script_content,
                    'script_name': script_name,
                    'attempts': attempt,
                    'msg_history': msg_history,
                }

            except Exception as e:
                error_msg = f"生成脚本异常: {str(e)}"
                self.logger.error(error_msg)
                format_errors.append(error_msg)

                if attempt >= max_format_retries:
                    break

                time.sleep(1)

        return False, {
            "error": f"脚本生成失败，已达到最大格式重试次数 {max_format_retries}",
            "format_errors": format_errors,
            "msg_history": msg_history,
        }

    def upload_script_to_container(
        self,
        container_id: str,
        script_content: str,
        script_name: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        if not container_id:
            return False, {"error": "容器ID不能为空"}

        self.sandbox.container_id = container_id

        try:
            mkdir_cmd = f"mkdir -p /workspace/scripts"
            output, stderr, exit_code = self.sandbox.execute(mkdir_cmd)
            if exit_code != 0:
                return False, {"error": f"创建脚本目录失败: {stderr}"}

            local_script_path = os.path.join(os.getcwd(), "scripts", script_name)
            self.logger.debug(f"local_script_path: {local_script_path}")
            os.makedirs(os.path.dirname(local_script_path), exist_ok=True)
            with open(local_script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            remote_path = f"/workspace/scripts/{script_name}"
            success, upload_output, upload_error, upload_exit = self.sandbox.upload_file(
                local_script_path, remote_path
            )

            if not success:
                return False, {"error": f"上传脚本失败: {upload_error}"}

            chmod_cmd = f"chmod +x {remote_path}"
            output, stderr, exit_code = self.sandbox.execute(chmod_cmd)
            if exit_code != 0:
                return False, {"error": f"设置脚本权限失败: {stderr}"}

            self.logger.info(f"脚本已上传到容器: {remote_path}")
            return True, {
                "local_path": local_script_path,
                "remote_path": remote_path,
            }

        except Exception as e:
            return False, {"error": f"上传脚本异常: {str(e)}"}

    def execute_script_in_container(
        self,
        container_id: str,
        script_path: str,
        context: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """在容器内执行脚本，委托给基类 retry_command。"""
        return self.retry_command(
            container_id=container_id,
            initial_command=f"bash {script_path}",
            context=context,
            command_type="exec",
        )

    def run(self, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        container_id = context.get('container_id')
        if not container_id:
            return False, {"error": "缺少 container_id"}
        self.logger.debug(f"container_id: {container_id}")

        def action_fn(run_context, msg_history):

            self.logger.info("步骤1: 生成评测脚本...")
            success, gen_result = self.generate_script(run_context, msg_history=msg_history)
            msg_history_out = gen_result.pop("msg_history", msg_history)
            self.logger.debug(f"gen_result: {gen_result}")

            if not success:
                self.logger.error(f"脚本生成失败: {gen_result.get('error', '')}")
                return False, {
                    "error": gen_result.get('error', '脚本生成失败'),
                    "msg_history": msg_history_out,
                    "_skip_evaluator": True,
                    "_retry_delay": 1,
                }

            script_content = gen_result['script_content']
            script_name = gen_result['script_name']


            self.logger.info("步骤2: 上传脚本到容器...")
            success, upload_result = self.upload_script_to_container(
                container_id, script_content, script_name
            )
            if not success:
                error_msg = f"脚本上传失败: {upload_result.get('error', '')}"
                self.logger.error(error_msg)
                msg_history_out.append({
                    "role": "user",
                    "content": f"脚本上传到容器失败：{error_msg}。请检查脚本内容是否有问题并重新生成。"
                })
                return False, {
                    "error": error_msg,
                    "msg_history": msg_history_out,
                    "_skip_evaluator": True,
                    "_retry_delay": 1,
                    "_script_name": script_name,
                    "_script_content": script_content,
                }

            remote_path = upload_result['remote_path']

            self.logger.info("步骤3: 在容器内执行脚本...")
            exec_command = f"bash {remote_path}"
            try:
                self.sandbox.container_id = container_id
                output, stderr, exit_code = self.sandbox.execute(exec_command)
            except Exception as e:
                output, stderr, exit_code = "", str(e), -1

            if exit_code == 0:
                self.logger.info("脚本执行成功")
                return True, {
                    "script_content": script_content,
                    "script_name": script_name,
                    "remote_path": remote_path,
                    "output": output,
                    "msg_history": msg_history_out,
                }

            error_msg = f"脚本执行失败，退出码={exit_code}: {stderr if stderr else output}"
            return False, {
                "error": error_msg,
                "msg_history": msg_history_out,
                "_script_name": script_name,
                "_script_content": script_content,
            }

        # Executor 特有的回调
        def get_failed_command(attempt_ctx, result):
            return result.get('_script_content', '')

        def on_success(result, attempt):
            result['attempts'] = attempt
            return result

        def on_failure_record(attempt, cmd, error, eval_result):
            record = {
                'attempt': attempt,
                'script_name': '',
                'script_content': cmd,
                'error': error,
            }
            if eval_result:
                record.update({
                    'analysis': eval_result.get('analysis', ''),
                    'suggestion': eval_result.get('suggestion', ''),
                    'is_recoverable': eval_result.get('is_recoverable', False),
                })
            else:
                record.update({
                    'analysis': '脚本格式生成失败，需重新生成',
                    'suggestion': '',
                })
            return record

        def on_unrecoverable(error, analysis, ctx, attempt, memory):
            return {
                "error": f"脚本执行失败，不可恢复: {error}",
                "analysis": analysis,
                "script_content": memory[-1].get('script_content', '') if memory else '',
                "script_name": memory[-1].get('script_name', '') if memory else '',
                "attempts": attempt,
                "local_memory": memory,
            }

        def on_max_retries(ctx, attempt, memory):
            return {
                "error": f"脚本执行失败，已达到最大尝试次数 {self.max_retries}",
                "attempts": attempt,
                "local_memory": memory,
            }

        def build_feedback(cmd, error, analysis, suggestion):
            return (
                f"上一次生成的脚本执行失败。\n"
                f"错误信息: {error}\n"
                f"错误分析: {analysis}\n"
                f"改进建议: {suggestion}\n"
                f"请根据以上反馈重新生成修正后的评测脚本。"
            )

        return self.execute_with_retry(
            action_fn=action_fn,
            context=context,
            command_type="exec",
            get_failed_command=get_failed_command,
            on_success=on_success,
            on_failure_record=on_failure_record,
            on_unrecoverable=on_unrecoverable,
            on_max_retries=on_max_retries,
            build_feedback=build_feedback,
            retry_delay=2.0,
        )