#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import json
import argparse
from utils.logging_config import setup_logging
from agent.controller import EvaluationController


def main():
    parser = argparse.ArgumentParser(description="自动化评测Agent系统")

    parser.add_argument("--config", type=str, help="评测配置文件路径(JSON)")
    parser.add_argument(
        "--no-interactive", action="store_true",
        help="非交互模式，必须通过--config或命令行参数提供配置",
    )

    parser.add_argument("--chip", type=str, help="芯片类型，如 NVIDIA_H200, Ascend_910B")
    parser.add_argument("--scenario", type=str, help="应用场景，如 language, vision_multimodal")
    parser.add_argument("--task", type=str, help="任务类型，如 operator, training, inference")
    parser.add_argument("--card-count", type=int, default=8, help="卡数量(默认8)")
    parser.add_argument("--test-case", type=str, default="", help="测试用例名称")

    parser.add_argument("--model", type=str, default="claude-opus-4-6", help="LLM模型名称")
    parser.add_argument("--skill-config", type=str, default="./config/skills", help="技能配置目录")

    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    for d in ["scripts", "results", "logs", "data"]:
        os.makedirs(d, exist_ok=True)

    controller = EvaluationController(
        model_name=args.model,
        skill_config_dir=args.skill_config,
    )

    interactive = not args.no_interactive
    config = None
    config_path = None

    if args.config:
        config_path = args.config
        interactive = False
        print(f"使用配置文件: {config_path}")
    elif args.chip and args.scenario and args.task:
        config = {
            "chip_type": args.chip,
            "application_scenario": args.scenario,
            "task_type": args.task,
            "task_scenario": args.task,
            "card_count": args.card_count,
            "test_case": args.test_case,
        }
        interactive = False
        print(f"使用命令行配置: 芯片={args.chip}, 场景={args.scenario}, 任务={args.task}")
    elif args.no_interactive:
        print("错误: 非交互模式必须通过 --config 或 --chip/--scenario/--task 提供配置")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  自动化评测Agent启动")
    print("=" * 60)

    success, result = controller.run_evaluation(
        config=config, config_path=config_path, interactive=interactive
    )

    if success:
        print("\n" + "=" * 60)
        print("  评测完成!")
        print("=" * 60)

        result_path = os.path.join("results", f"eval_result.json")
        try:
            serializable = {k: v for k, v in result.items() if k != "image_config"}
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)
            print(f"结果已保存到: {result_path}")
        except Exception as e:
            print(f"保存结果失败: {e}")

        sys.exit(0)
    else:
        print(f"\n评测失败: {result.get('error', '未知错误')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
