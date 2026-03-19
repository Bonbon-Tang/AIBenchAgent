#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import json
from datetime import datetime
from typing import Dict, Any, Tuple
from utils.llm import create_client
from .collector import Collector
from .creator import Creator
from .evaluator import Evaluator
from .executor import Executor
from skills.skill_registry import SkillRegistry


class EvaluationController:

    def __init__(self, model_name: str = "claude-opus-4-6", skill_config_dir: str = None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.llm, self.model = create_client(model_name)

        self.tool_registry = SkillRegistry(config_dir=skill_config_dir)

        self.evaluator = Evaluator(self.llm, self.model)
        self.collector = Collector(
            llm=self.llm, modelname=self.model, tool_selector=self.tool_registry
        )
        self.creator = Creator(
            llm=self.llm,
            model_name=self.model,
            evaluator=self.evaluator,
        )
        self.executor = Executor(
            llm=self.llm,
            model_name=self.model,
            evaluator=self.evaluator,
        )

        self.max_retries = 3

    def run_evaluation(self, config: Dict[str, Any] = None, config_path: str = None, interactive: bool = True) -> Tuple[bool, Dict[str, Any]]:
        context = {"timestamp": datetime.now().isoformat()}

        try:
            self.logger.info("阶段1: 收集用户评测配置信息")
            print("\n" + "=" * 50)
            print("阶段1: 收集用户评测配置信息")
            print("=" * 50)

            if config_path:
                try:
                    user_config = self.collector.load_from_config(config_path)
                    print(f"已从配置文件加载: {config_path}")
                except (FileNotFoundError, ValueError) as e:
                    return False, {"error": f"配置文件加载失败: {str(e)}"}
            elif config:
                user_config = config
            elif interactive:
                user_config = self.collector.collect_user_info(context)
            else:
                return False, {"error": "非交互模式必须提供配置信息（config或config_path）"}

            if not user_config:
                self.logger.error("用户配置信息为空")
                return False, {"error": "用户配置信息为空"}

            context.update(user_config)
            self.logger.info(f"用户配置: {json.dumps(context, ensure_ascii=False, default=str)}")
            print(f"\n已收集配置信息: {json.dumps(user_config, ensure_ascii=False, indent=2, default=str)}")

            is_valid, validation_msg = self._validate_config(context)
            if not is_valid:
                return False, {"error": f"配置验证失败: {validation_msg}"}

            self.logger.info("阶段2: 创建Docker容器")
            print("\n" + "=" * 50)
            print("阶段2: 创建Docker容器")
            print("=" * 50)

            if "task_type" not in context and "task_scenario" in context:
                context["task_type"] = context["task_scenario"]

            success, container_result = self.creator.create_container(context)
            if not success:
                error_msg = container_result.get("error", "容器创建失败")
                self.logger.error(f"容器创建失败: {error_msg}")
                return False, {"error": error_msg, "stage": "create_container", **container_result}

            container_id = container_result.get("container_id", "")
            container_name = container_result.get("container_name", "")
            context.update(container_result)
            print(f"\n容器创建成功! ID: {container_id}, Name: {container_name}")

            self.logger.info("阶段3: 生成并执行评测脚本")
            print("\n" + "=" * 50)
            print("阶段3: 生成并执行评测脚本")
            print("=" * 50)

            success, exec_result = self.executor.run(context)
            if not success:
                error_msg = exec_result.get("error", "脚本执行失败")
                stage = exec_result.get("stage", "execute")
                self.logger.error(f"脚本执行失败 (阶段: {stage}): {error_msg}")
                self._cleanup(container_id)
                return False, {"error": error_msg, "stage": stage, **exec_result}

            context.update(exec_result)
            print(f"\n评测脚本执行成功!")
            print(f"脚本: {exec_result.get('script_name', '')}")
            print(f"输出: {exec_result.get('output', '')[:500]}")

            self.logger.info("阶段4: 收集评测结果")
            print("\n" + "=" * 50)
            print("阶段4: 收集评测结果")
            print("=" * 50)

            result_data = self._collect_results(container_id, context)
            context["evaluation_results"] = result_data

            self._cleanup(container_id)

            print(f"\n评测完成!")
            print(f"结果: {json.dumps(result_data, ensure_ascii=False, indent=2, default=str)}")

            return True, context

        except Exception as e:
            self.logger.error(f"评测流程异常: {str(e)}")
            container_id = context.get("container_id")
            if container_id:
                self._cleanup(container_id)
            return False, {"error": f"评测流程异常: {str(e)}"}

    def _validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        required_fields = ["application_scenario", "chip_type"]
        for field in required_fields:
            if field not in config or not config[field]:
                return False, f"缺少必要字段: {field}"

        if not config.get("task_scenario") and not config.get("task_type"):
            return False, "缺少任务场景(task_scenario)或任务类型(task_type)"

        task_type = config.get("task_type", config.get("task_scenario"))
        is_valid = self.tool_registry.validate_config(
            config["application_scenario"], config["chip_type"], task_type
        )
        if not is_valid:
            return False, (
                f"不支持的配置组合: 场景={config['application_scenario']}, "
                f"芯片={config['chip_type']}, 任务={task_type}"
            )

        return True, ""

    def _collect_results(self, container_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        results = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "application_scenario": context.get("application_scenario"),
                "task_type": context.get("task_type", context.get("task_scenario")),
                "chip_type": context.get("chip_type"),
                "card_count": context.get("card_count"),
                "test_case": context.get("test_case"),
            },
            "script_output": context.get("output", ""),
        }

        if container_id:
            try:
                sandbox = self.creator.sandbox
                sandbox.container_id = container_id
                output, stderr, exit_code = sandbox.execute(
                    f"docker exec {container_id} cat /workspace/results/result.json"
                )
                if exit_code == 0 and output.strip():
                    result_json = json.loads(output.strip())
                    results["metrics"] = result_json
            except Exception as e:
                self.logger.warning(f"读取result.json失败: {str(e)}")
                results["metrics"] = {}

        return results

    def _cleanup(self, container_id: str):
        if container_id:
            try:
                self.creator.sandbox.container_id = container_id
                self.creator.cleanup_container()
                self.logger.info(f"容器 {container_id} 已清理")
            except Exception as e:
                self.logger.warning(f"容器清理失败: {str(e)}")
