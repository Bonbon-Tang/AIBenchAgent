#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import logging
from typing import Dict, Any, Tuple, List, Callable, Optional
from sandbox.docker_sandbox import DockerSandbox
from .evaluator import Evaluator


class EvalRetryAgent:
    """提供统一的ReAct反馈循环Agent类。"""

    def __init__(self, llm, model_name, evaluator=None, max_retries=10):
        self.llm = llm
        self.model = model_name
        self.max_retries = max_retries
        self.sandbox = DockerSandbox()
        self.logger = logging.getLogger(self.__class__.__module__)
        self.logger.setLevel(logging.INFO)

        if evaluator is None:
            self.evaluator = Evaluator(llm, model_name)
        else:
            self.evaluator = evaluator

    def execute_with_retry(
        self,
        action_fn: Callable[[Dict[str, Any], List], Tuple[bool, Dict[str, Any]]],
        context: Dict[str, Any],
        *,
        command_type: str = "docker",
        prepare_context: Optional[Callable[[Dict[str, Any], List], Dict[str, Any]]] = None,
        get_failed_command: Optional[Callable[[Dict[str, Any], Dict[str, Any]], str]] = None,
        on_success: Optional[Callable[[Dict[str, Any], int], Dict[str, Any]]] = None,
        on_failure_record: Optional[Callable] = None,
        on_unrecoverable: Optional[Callable] = None,
        on_max_retries: Optional[Callable] = None,
        build_feedback: Optional[Callable[[str, str, str, str], str]] = None,
        retry_delay: float = 2.0,
    ) -> Tuple[bool, Dict[str, Any]]:
        if prepare_context is None:
            prepare_context = lambda ctx, mem: {**ctx, 'local_memory': mem.copy()}

        if get_failed_command is None:
            get_failed_command = lambda ctx, res: ctx.get('generated_command', '')

        if on_success is None:
            on_success = lambda result, attempt: {**result, 'attempts': attempt}

        if on_failure_record is None:
            def on_failure_record(attempt, cmd, error, eval_result):
                if eval_result:
                    return {
                        'attempt': attempt, 'command': cmd, 'error': error,
                        'analysis': eval_result.get('analysis', ''),
                        'suggestion': eval_result.get('suggestion', ''),
                        'adjusted_command': eval_result.get('adjusted_command', ''),
                        'is_recoverable': eval_result.get('is_recoverable', False),
                    }
                return {'attempt': attempt, 'command': cmd, 'error': error,
                        'analysis': '', 'suggestion': ''}

        if on_unrecoverable is None:
            def on_unrecoverable(error, analysis, ctx, attempt, memory):
                return {"error": f"执行失败，不可恢复: {error}",
                        "analysis": analysis, "attempts": attempt,
                        "local_memory": memory}

        if on_max_retries is None:
            max_retries_ref = self.max_retries
            def on_max_retries(ctx, attempt, memory):
                return {"error": f"执行失败，已达到最大尝试次数 {max_retries_ref}",
                        "attempts": attempt, "local_memory": memory}

        if build_feedback is None:
            def build_feedback(cmd, error, analysis, suggestion):
                return (
                    f"上一次生成的命令执行失败。\n"
                    f"失败命令: {cmd}\n"
                    f"错误信息: {error}\n"
                    f"错误分析: {analysis}\n"
                    f"改进建议: {suggestion}\n"
                    f"请根据以上反馈重新生成正确的命令。"
                )

        local_memory: List[Dict] = []
        msg_history: List[Dict] = []
        evaluator_msg_history: List[Dict] = []
        attempt = 0

        while attempt < self.max_retries:
            attempt += 1
            self.logger.info(f"第 {attempt} 次尝试...")

            attempt_context = prepare_context(context, local_memory)
            success, result = action_fn(attempt_context, msg_history)
            msg_history = result.pop("msg_history", msg_history)

            if success:
                self.logger.info(f"第 {attempt} 次尝试成功")
                return True, on_success(result, attempt)

            error_msg = result.get('error', str(result))
            failed_command = get_failed_command(attempt_context, result)
            self.logger.warning(f"第 {attempt} 次尝试失败: {error_msg}")

            if result.get('_skip_evaluator'):
                local_memory.append(on_failure_record(attempt, failed_command, error_msg, None))
                time.sleep(result.get('_retry_delay', 1))
                continue

            if attempt < self.max_retries:
                self.logger.info("请求Evaluator评估...")
                eval_result = self.evaluator.evaluate(
                    failed_command=failed_command,
                    error_output=error_msg,
                    context=attempt_context,
                    command_type=command_type,
                    msg_history=evaluator_msg_history,
                )
                evaluator_msg_history = eval_result.pop("msg_history", evaluator_msg_history)
                print(eval_result)

                is_recoverable = eval_result.get('is_recoverable', False)
                analysis = eval_result.get('analysis', '')
                suggestion = eval_result.get('suggestion', '')

                self.logger.warning(f"评估结果: 可恢复={is_recoverable}, 建议={suggestion}")

                local_memory.append(on_failure_record(attempt, failed_command, error_msg, eval_result))

                if not is_recoverable:
                    self.logger.error("Evaluator认为此错误不可恢复，停止重试")
                    return False, on_unrecoverable(error_msg, analysis, context, attempt, local_memory)

                feedback = build_feedback(failed_command, error_msg, analysis, suggestion)
                msg_history.append({"role": "user", "content": feedback})

                time.sleep(retry_delay)
            else:
                local_memory.append(on_failure_record(attempt, failed_command, error_msg, None))

        self.logger.error(f"执行失败，已达到最大尝试次数 {self.max_retries}")
        return False, on_max_retries(context, attempt, local_memory)

    def retry_command(
        self,
        container_id: str,
        initial_command: str,
        context: Dict[str, Any],
        command_type: str = "exec",
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        在容器内执行命令，失败时让 Evaluator 调整命令后重试。

        合并了 Creator.execute_in_container 和
        Executor.execute_script_in_container 的逻辑。
        """
        if not container_id:
            return False, {"error": "容器ID不能为空"}

        self.sandbox.container_id = container_id
        local_memory = context.get('local_memory', [])
        evaluator_msg_history: List[Dict] = []
        attempt = 0
        last_command = None
        last_error = None
        exec_command = initial_command

        while attempt < self.max_retries:
            attempt += 1
            self.logger.info(f"尝试在容器内执行命令，第 {attempt} 次尝试...")

            if attempt > 1:
                self.logger.info("请求Evaluator评估...")
                eval_result = self.evaluator.evaluate(
                    failed_command=last_command,
                    error_output=last_error,
                    context=context,
                    command_type=command_type,
                    msg_history=evaluator_msg_history,
                )
                evaluator_msg_history = eval_result.pop("msg_history", evaluator_msg_history)

                is_recoverable = eval_result.get('is_recoverable', False)
                adjusted_command = eval_result.get('adjusted_command', last_command)

                self.logger.info(f"评估结果: 可恢复={is_recoverable}")

                if not is_recoverable:
                    self.logger.error("Evaluator认为此错误不可恢复，停止重试")
                    return False, {
                        "error": last_error,
                        "analysis": eval_result.get('analysis', ''),
                        "attempts": attempt - 1,
                        "local_memory": local_memory,
                    }

                exec_command = adjusted_command

            try:
                self.logger.info(f"执行命令: {exec_command}")
                output, error_output, exit_code = self.sandbox.execute(exec_command)

                if exit_code == 0:
                    self.logger.info("命令执行成功")
                    return True, {
                        "output": output,
                        "exit_code": exit_code,
                        "attempts": attempt,
                        "command": exec_command,
                    }
                else:
                    error_msg = f"命令执行失败，退出码={exit_code}: {error_output if error_output else output}"
                    last_command = exec_command
                    last_error = error_msg

                    local_memory.append({
                        'attempt': attempt,
                        'command': exec_command,
                        'error': error_msg,
                    })

                    if attempt >= self.max_retries:
                        self.logger.error("已达到最大尝试次数")
                        return False, {
                            "error": error_msg,
                            "exit_code": exit_code,
                            "attempts": attempt,
                            "local_memory": local_memory,
                        }

                    self.logger.warning(f"第 {attempt} 次尝试失败: {error_msg}")
                    time.sleep(2)

            except Exception as e:
                error_msg = f"执行命令异常: {str(e)}"
                last_command = exec_command
                last_error = error_msg

                local_memory.append({
                    'attempt': attempt,
                    'command': exec_command,
                    'error': error_msg,
                })

                self.logger.error(error_msg)

                if attempt >= self.max_retries:
                    return False, {
                        "error": error_msg,
                        "attempts": attempt,
                        "local_memory": local_memory,
                    }

                time.sleep(2)

        return False, {
            "error": f"命令执行失败，已达到最大尝试次数 {self.max_retries}",
            "attempts": attempt,
            "local_memory": local_memory,
        }