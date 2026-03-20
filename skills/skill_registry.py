#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


@dataclass
class SkillConfig:
    image_name: str
    start_command: str
    task_command: str
    description: str = ""
    start_command_hints: str = ""
    task_command_hints: str = ""
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[str] = field(default_factory=list)
    source: str = "builtin"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_name": self.image_name,
            "start_command": self.start_command,
            "task_command": self.task_command,
            "description": self.description,
            "start_command_hints": self.start_command_hints,
            "task_command_hints": self.task_command_hints,
            "environment": self.environment,
            "volumes": self.volumes,
        }

    def to_prompt(self) -> str:
        lines = [
            f"- 镜像: {self.image_name}",
            f"- 启动命令模板: {self.start_command}",
            f"- 评测命令模板: {self.task_command}",
        ]
        if self.description:
            lines.append(f"- 说明: {self.description}")
        if self.environment:
            env_str = ", ".join(f"{k}={v}" for k, v in self.environment.items())
            lines.append(f"- 环境变量: {env_str}")
        if self.volumes:
            lines.append(f"- 挂载目录: {', '.join(self.volumes)}")
        if self.start_command_hints:
            lines.append(f"- 容器启动提示: {self.start_command_hints}")
        if self.task_command_hints:
            lines.append(f"- 评测脚本提示: {self.task_command_hints}")
        return "\n".join(lines)


class SkillRegistry:

    def __init__(self, config_dir: str = None):
        self.logger = logging.getLogger(__name__)

        self._chips = {
            "NVIDIA_H200": "NVIDIA H200 GPU，适用于大规模训练和推理",
            "Ascend_910B": "华为昇腾910B，适用于AI训练和推理",
        }

        self._scenarios = {
            "language": "语言场景，包括文本生成、文本分类、机器翻译等",
            "vision_multimodal": "视觉和多模态场景，包括图像分类、目标检测、图像生成等",
            "audio": "音频场景，包括语音识别、语音合成、音频分类等",
            "scientific_computing": "科学计算场景，包括数值模拟、数据分析、物理仿真等",
            "retrieval_matching": "检索匹配场景，包括语义搜索、推荐系统、相似度匹配等",
        }

        self._task_types = {
            "operator": "算子任务，主要进行矩阵运算、卷积等基础算子性能测试",
            "training": "模型训练任务，包括从头训练深度学习模型",
            "inference": "模型推理任务，对已训练模型进行推理性能测试",
            "fine_tuning": "模型微调任务，在预训练模型基础上进行微调",
            "reinforcement_training": "强化学习训练任务，使用强化学习算法训练智能体",
        }

        self._skills: Dict[Tuple[str, str, str], SkillConfig] = {}

        self._load_builtin_skills()

        if config_dir:
            self.load_from_directory(config_dir)


    def register(self, chip: str, scenario: str, task: str, config: SkillConfig):
        self._skills[(chip, scenario, task)] = config

    def get_skill(self, chip: str, scenario: str, task: str) -> SkillConfig:
        key = (chip, scenario, task)
        if key not in self._skills:
            raise ValueError(
                f"不支持的技能组合: 芯片={chip}, 场景={scenario}, 任务={task}"
            )
        return self._skills[key]

    def get_skill_prompt(self, chip: str, scenario: str, task: str) -> str:
        skill = self.get_skill(chip, scenario, task)
        return skill.to_prompt()


    def get_available_chips(self) -> List[Dict[str, str]]:
        return [{"name": n, "description": d} for n, d in self._chips.items()]

    def get_available_task_types(self) -> List[Dict[str, str]]:
        return [{"name": n, "description": d} for n, d in self._task_types.items()]

    def get_available_application_scenarios(self) -> List[Dict[str, str]]:
        return [{"name": n, "description": d} for n, d in self._scenarios.items()]

    def get_supported_tasks_for_chip(self, chip_type: str, application_scenario: str = None) -> List[str]:
        tasks = []
        for (chip, scenario, task) in self._skills:
            if chip == chip_type:
                if application_scenario is None or scenario == application_scenario:
                    if task not in tasks:
                        tasks.append(task)
        return tasks

    def validate_config(self, application_scenario: str, chip_type: str, task_type: str) -> bool:
        return (chip_type, application_scenario, task_type) in self._skills


    def get_image_config(self, application_scenario: str, chip_type: str, task_type: str) -> Dict[str, Any]:
        if chip_type not in self._chips:
            raise ValueError(f"不支持的芯片类型: {chip_type}")
        has_scenario = any(
            c == chip_type and s == application_scenario
            for (c, s, _) in self._skills
        )
        if not has_scenario:
            raise ValueError(f"芯片 {chip_type} 不支持应用场景: {application_scenario}")
        has_task = any(
            c == chip_type and s == application_scenario and t == task_type
            for (c, s, t) in self._skills
        )
        if not has_task:
            raise ValueError(
                f"芯片 {chip_type} 在应用场景 {application_scenario} 下不支持任务类型: {task_type}"
            )
        skill = self.get_skill(chip_type, application_scenario, task_type)
        return skill.to_dict()


    def load_from_directory(self, config_dir: str):
        config_path = Path(config_dir)
        if not config_path.exists():
            self.logger.warning(f"配置目录不存在: {config_dir}")
            return

        loaded_count = 0
        for chip_dir in config_path.iterdir():
            if not chip_dir.is_dir():
                continue
            chip = chip_dir.name
            for scenario_dir in chip_dir.iterdir():
                if not scenario_dir.is_dir():
                    continue
                scenario = scenario_dir.name
                for task_file in scenario_dir.iterdir():
                    if task_file.suffix == ".json":
                        task = task_file.stem
                        skill = self._load_skill_from_json(task_file, chip)
                        if skill:
                            self.register(chip, scenario, task, skill)
                            loaded_count += 1
                    elif task_file.suffix == ".md":
                        task = task_file.stem
                        skill = self._load_skill_from_markdown(task_file, chip)
                        if skill:
                            self.register(chip, scenario, task, skill)
                            loaded_count += 1

        self.logger.info(f"从 {config_dir} 加载了 {loaded_count} 个技能配置")

    def _load_skill_from_json(self, filepath: Path, chip: str) -> Optional[SkillConfig]:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "start_command" not in data:
                data["start_command"] = _CHIP_DEFAULTS.get(chip, {}).get(
                    "start_command", ""
                )

            return SkillConfig(
                image_name=data["image_name"],
                start_command=data["start_command"],
                task_command=data["task_command"],
                description=data.get("description", ""),
                start_command_hints=data.get("start_command_hints", ""),
                task_command_hints=data.get("task_command_hints", ""),
                environment=data.get("environment", {}),
                volumes=data.get("volumes", []),
                source=str(filepath),
            )
        except Exception as e:
            self.logger.error(f"加载技能配置失败 {filepath}: {e}")
            return None

    def _load_skill_from_markdown(self, filepath: Path, chip: str) -> Optional[SkillConfig]:
        try:
            content = filepath.read_text(encoding="utf-8")
            parts = content.split("---", 2)
            if len(parts) < 3:
                self.logger.warning(f"Markdown文件缺少frontmatter: {filepath}")
                return None

            import yaml
            metadata = yaml.safe_load(parts[1])

            body = parts[2].strip()

            if "start_command" not in metadata:
                metadata["start_command"] = _CHIP_DEFAULTS.get(chip, {}).get(
                    "start_command", ""
                )

            return SkillConfig(
                image_name=metadata["image_name"],
                start_command=metadata["start_command"],
                task_command=metadata["task_command"],
                description=metadata.get("description", ""),
                start_command_hints=metadata.get("start_command_hints", ""),
                task_command_hints=body if body else metadata.get("task_command_hints", ""),
                environment=metadata.get("environment", {}),
                volumes=metadata.get("volumes", []),
                source=str(filepath),
            )
        except ImportError:
            self.logger.warning(f"Markdown加载需要pyyaml库，跳过: {filepath}")
            return None
        except Exception as e:
            self.logger.error(f"加载Markdown技能配置失败 {filepath}: {e}")
            return None


    def _load_builtin_skills(self):
        self._register_nvidia_h200_skills()
        self._register_ascend_910b_skills()

    def _register_nvidia_h200_skills(self):
        chip = "NVIDIA_H200"
        start_cmd = _CHIP_DEFAULTS[chip]["start_command"]
        default_img = _CHIP_DEFAULTS[chip]["default_image"]

        skills = {
            ("language", "operator"): (
                None,
                "python benchmark_language_ops.py --test_attention --test_transformer --batch_size 512",
                "H200有141GB HBM3e显存，建议使用--shm-size=10g。多卡场景需映射所有GPU(--gpus all)。",
                "GEMM/Attention算子建议使用cuBLAS/cuDNN库。支持FP16/BF16/FP32精度对比。"
                "batch_size=512可充分利用H200 141GB HBM3e显存。",
            ),
            ("language", "training"): (
                "huggingface/transformers-pytorch-gpu:latest",
                "python train_language_model.py --model_type gpt2 --dataset wikitext --batch_size 16 --epochs 10",
                "训练任务需要挂载数据目录和输出目录。建议使用--shm-size=16g以支持多进程DataLoader。",
                "推荐HuggingFace Transformers框架。支持混合精度训练(AMP)和gradient checkpointing。",
            ),
            ("language", "inference"): (
                "huggingface/transformers-pytorch-gpu:latest",
                "python inference_language.py --model_path /models/ --text_input 'Hello world' --max_length 100",
                "推理任务需要挂载模型目录到容器内/models/路径。",
                "推理测试关注throughput(tokens/s)和latency(ms)。建议使用vLLM或TensorRT-LLM加速。",
            ),
            ("language", "fine_tuning"): (
                "huggingface/transformers-pytorch-gpu:latest",
                "python finetune_language.py --model_name bert-base-uncased --dataset /data/ --learning_rate 2e-5 --epochs 5",
                "微调需要挂载预训练模型和数据目录。建议使用--shm-size=16g。",
                "微调建议使用LoRA/QLoRA以节省显存。数据集放在/data/目录下。",
            ),
            ("language", "reinforcement_training"): (
                "rayproject/ray:latest-gpu",
                "python rl_train_language.py --algorithm PPO --env text_generation --total_timesteps 500000",
                "Ray框架需要较大共享内存，建议--shm-size=16g。",
                "强化学习训练使用Ray/RLlib框架。RLHF场景建议使用DeepSpeed-Chat。",
            ),
            ("vision_multimodal", "operator"): (
                None,
                "python benchmark_vision_ops.py --test_conv --test_pooling --batch_size 256",
                "算子测试需要GPU支持，使用--gpus all。",
                "Conv2D/Pooling算子测试。建议使用cuDNN自动调优(cudnn.benchmark=True)。",
            ),
            ("vision_multimodal", "training"): (
                "pytorch/pytorch:latest",
                "python train_vision.py --model resnet50 --dataset imagenet --batch_size 32 --epochs 100",
                "需要挂载ImageNet数据集目录到/data/imagenet/。建议使用--shm-size=16g。",
                "视觉模型训练推荐PyTorch框架。ImageNet数据集需提前下载到/data/imagenet/。",
            ),
            ("vision_multimodal", "inference"): (
                "tensorflow/tensorflow:latest-gpu",
                "python inference_vision.py --model_path /models/ --image_path /images/ --batch_size 16",
                "需要挂载模型目录和图片目录到容器内。",
                "推理测试关注FPS和latency。建议使用TensorRT进行模型优化。",
            ),
            ("vision_multimodal", "fine_tuning"): (
                "pytorch/pytorch:latest",
                "python finetune_vision.py --model vit_base_patch16 --dataset flowers102 --lr 1e-4",
                "微调需要挂载预训练模型和数据目录。",
                "ViT微调建议使用timm库。小数据集可使用数据增强提升效果。",
            ),
            ("vision_multimodal", "reinforcement_training"): (
                "stablebaselines/stable-baselines3:latest",
                "python rl_train_vision.py --algorithm SAC --env robotic_vision --total_timesteps 1000000",
                "环境渲染需要GPU支持，使用--gpus all。",
                "视觉强化学习使用Stable-Baselines3框架。环境渲染需要GPU支持。",
            ),
            ("audio", "operator"): (
                None,
                "python benchmark_audio_ops.py --test_fft --test_mfcc --batch_size 128",
                "音频算子测试需要GPU支持。",
                "FFT/MFCC算子测试。音频预处理建议使用torchaudio。",
            ),
            ("audio", "training"): (
                "pytorch/pytorch:latest",
                "python train_audio.py --model wav2vec2 --dataset librispeech --batch_size 8",
                "需要挂载LibriSpeech数据集目录。建议使用--shm-size=8g。",
                "语音模型训练推荐wav2vec2/HuBERT。LibriSpeech数据集需提前下载。",
            ),
            ("audio", "inference"): (
                "pytorch/pytorch:latest",
                "python inference_audio.py --model_path /models/ --audio_path /audio/ --sample_rate 16000",
                "需要挂载模型目录和音频目录到容器内。",
                "音频推理测试关注RTF(Real-Time Factor)。采样率统一为16kHz。",
            ),
            ("audio", "fine_tuning"): (
                "huggingface/transformers-pytorch-gpu:latest",
                "python finetune_audio.py --model whisper-small --dataset common_voice --epochs 10",
                "需要挂载预训练模型和Common Voice数据集目录。",
                "Whisper微调使用HuggingFace Transformers。Common Voice数据集支持多语言。",
            ),
            ("audio", "reinforcement_training"): (
                "openai/gym:0.26.2",
                "python rl_train_audio.py --algorithm A2C --env speech_recognition --total_timesteps 300000",
                "使用--gpus all确保GPU可用。",
                "音频强化学习场景较少，可用于语音交互Agent训练。",
            ),
            ("scientific_computing", "operator"): (
                None,
                "python scientific_benchmark.py --test_matrix_ops --test_fft --precision double",
                "科学计算需要GPU支持，使用--gpus all。",
                "科学计算通常需要FP64精度。矩阵运算建议使用cuSOLVER/cuSPARSE。",
            ),
            ("scientific_computing", "training"): (
                "pytorch/pytorch:latest",
                "python train_scientific.py --model physics_net --dataset simulation_data --batch_size 8",
                "需要挂载模拟数据目录。",
                "物理模拟网络训练。建议使用Physics-Informed Neural Networks(PINNs)。",
            ),
            ("scientific_computing", "inference"): (
                "tensorflow/tensorflow:latest-gpu",
                "python inference_scientific.py --model_path /models/ --input_data /data/ --batch_size 4",
                "需要挂载模型和输入数据目录。",
                "科学计算推理关注数值精度和计算吞吐量。",
            ),
            ("scientific_computing", "fine_tuning"): (
                "pytorch/pytorch:latest",
                "python finetune_scientific.py --model climate_model --dataset weather_data --epochs 15",
                "需要挂载预训练模型和气象数据目录。",
                "科学模型微调注意数据归一化和物理约束。",
            ),
            ("scientific_computing", "reinforcement_training"): (
                "rayproject/ray:latest-gpu",
                "python rl_train_scientific.py --algorithm PPO --env molecular_dynamics --total_timesteps 800000",
                "分子动力学环境计算密集，建议使用--shm-size=16g。",
                "分子动力学强化学习。环境状态空间可能很大，注意内存使用。",
            ),
            ("retrieval_matching", "operator"): (
                None,
                "python benchmark_retrieval_ops.py --test_similarity --test_embedding --batch_size 1024",
                "向量检索算子需要GPU加速，使用--gpus all。",
                "相似度计算和embedding算子测试。建议使用FAISS GPU加速。",
            ),
            ("retrieval_matching", "training"): (
                "huggingface/sentence-transformers:latest",
                "python train_retrieval.py --model all-mpnet-base-v2 --dataset msmarco --batch_size 32",
                "需要挂载MSMARCO数据集目录。建议使用--shm-size=8g。",
                "检索模型训练使用Sentence-Transformers框架。MSMARCO是标准检索基准。",
            ),
            ("retrieval_matching", "inference"): (
                "huggingface/sentence-transformers:latest",
                "python inference_retrieval.py --model_path /models/ --query 'search query' --top_k 10",
                "需要挂载模型目录到容器内。",
                "检索推理关注recall@k和latency。建议使用ANN索引加速。",
            ),
            ("retrieval_matching", "fine_tuning"): (
                "huggingface/transformers-pytorch-gpu:latest",
                "python finetune_retrieval.py --model sentence-bert --dataset custom_qa --epochs 8",
                "需要挂载预训练模型和QA数据集目录。",
                "检索模型微调使用对比学习损失(contrastive loss)。",
            ),
            ("retrieval_matching", "reinforcement_training"): (
                "stablebaselines/stable-baselines3:latest",
                "python rl_train_retrieval.py --algorithm DQN --env recommendation --total_timesteps 400000",
                "使用--gpus all确保GPU可用。",
                "推荐系统强化学习。奖励函数需要根据业务指标设计。",
            ),
        }

        for (scenario, task), (image, task_cmd, start_hints, task_hints) in skills.items():
            self.register(chip, scenario, task, SkillConfig(
                image_name=image or default_img,
                start_command=start_cmd,
                task_command=task_cmd,
                start_command_hints=start_hints,
                task_command_hints=task_hints,
            ))

    def _register_ascend_910b_skills(self):
        chip = "Ascend_910B"
        start_cmd = _CHIP_DEFAULTS[chip]["start_command"]
        image = _CHIP_DEFAULTS[chip]["default_image"]

        skills = {
            ("language", "operator"): (
                "python ascend_language_ops.py --test_nlp_ops --device_target Ascend --batch_size 256",
                "昇腾需要映射NPU设备(--device=/dev/davinci0)。多卡场景需映射多个davinci设备。",
                "昇腾算子使用MindSpore框架。需指定--device_target Ascend。CANN 6.3提供算子加速。",
            ),
            ("language", "training"): (
                "python train_language.py --device_target Ascend --model_type bert --dataset_path /data/ --epochs 20",
                "训练任务需要挂载数据目录。昇腾设备需通过--device映射。",
                "昇腾训练使用MindSpore。自动混合精度通过mindspore.amp实现。",
            ),
            ("language", "inference"): (
                "python inference_language.py --device_target Ascend --model_path /models/ --text 'Hello world' --max_length 100",
                "推理任务需要挂载模型目录。昇腾设备需通过--device映射。",
                "昇腾推理可使用MindSpore Lite或ACL进行模型转换优化。",
            ),
            ("language", "fine_tuning"): (
                "python finetune_language.py --device_target Ascend --pretrained_model /pretrained/ --dataset /data/ --epochs 10",
                "微调需要挂载预训练模型和数据目录。昇腾设备需通过--device映射。",
                "昇腾微调建议使用MindFormers框架，支持主流大模型。",
            ),
            ("language", "reinforcement_training"): (
                "python rl_train_language.py --device_target Ascend --algorithm DQN --env text_generation",
                "昇腾设备需通过--device映射。",
                "昇腾强化学习可使用MindSpore Reinforcement框架。",
            ),
            ("vision_multimodal", "operator"): (
                "python ascend_vision_ops.py --test_cv_ops --device_target Ascend --batch_size 128",
                "昇腾需要映射NPU设备。",
                "昇腾视觉算子测试。AICPU和AICore算子性能可能不同。",
            ),
            ("vision_multimodal", "training"): (
                "python train_vision.py --device_target Ascend --model resnet50 --dataset_path /data/ --epochs 50",
                "需要挂载数据目录。昇腾设备需通过--device映射。",
                "昇腾视觉训练使用MindSpore ModelZoo中的预置模型。",
            ),
            ("vision_multimodal", "inference"): (
                "python inference_vision.py --device_target Ascend --model_path /models/ --image_path /images/ --batch_size 16",
                "需要挂载模型和图片目录。昇腾设备需通过--device映射。",
                "昇腾视觉推理建议使用OM模型格式(ATC工具转换)。",
            ),
            ("vision_multimodal", "fine_tuning"): (
                "python finetune_vision.py --device_target Ascend --pretrained_model /pretrained/ --dataset /data/ --epochs 15",
                "需要挂载预训练模型和数据目录。昇腾设备需通过--device映射。",
                "昇腾视觉微调支持迁移学习，冻结backbone微调head。",
            ),
            ("vision_multimodal", "reinforcement_training"): (
                "python rl_train_vision.py --device_target Ascend --algorithm PPO --env robotic_vision",
                "昇腾设备需通过--device映射。环境渲染可能需要CPU辅助。",
                "昇腾视觉强化学习场景，环境渲染可能需要CPU辅助。",
            ),
            ("audio", "operator"): (
                "python ascend_audio_ops.py --test_audio_ops --device_target Ascend --batch_size 64",
                "昇腾需要映射NPU设备。",
                "昇腾音频算子测试，FFT等操作可能在AICPU上执行。",
            ),
            ("audio", "training"): (
                "python train_audio.py --device_target Ascend --model audio_net --dataset_path /data/ --epochs 30",
                "需要挂载数据目录。昇腾设备需通过--device映射。",
                "昇腾音频训练，注意音频预处理步骤可能需要CPU完成。",
            ),
            ("audio", "inference"): (
                "python inference_audio.py --device_target Ascend --model_path /models/ --audio_path /audio/",
                "需要挂载模型和音频目录。昇腾设备需通过--device映射。",
                "昇腾音频推理，实时性要求高的场景建议使用流式推理。",
            ),
            ("audio", "fine_tuning"): (
                "python finetune_audio.py --device_target Ascend --pretrained_model /pretrained/ --dataset /data/ --epochs 12",
                "需要挂载预训练模型和数据目录。昇腾设备需通过--device映射。",
                "昇腾音频微调，支持多语言语音模型适配。",
            ),
            ("audio", "reinforcement_training"): (
                "python rl_train_audio.py --device_target Ascend --algorithm A2C --env speech_processing",
                "昇腾设备需通过--device映射。",
                "昇腾音频强化学习，适用于语音对话系统训练。",
            ),
            ("scientific_computing", "operator"): (
                "python ascend_scientific_ops.py --test_math_ops --device_target Ascend --precision float32",
                "昇腾需要映射NPU设备。",
                "昇腾科学计算算子，注意FP64支持可能受限，建议使用FP32。",
            ),
            ("scientific_computing", "training"): (
                "python train_scientific.py --device_target Ascend --model scientific_net --dataset_path /data/ --epochs 40",
                "需要挂载数据目录。昇腾设备需通过--device映射。",
                "昇腾科学计算训练，大规模矩阵运算可使用HCCL通信库加速。",
            ),
            ("scientific_computing", "inference"): (
                "python inference_scientific.py --device_target Ascend --model_path /models/ --input_data /data/",
                "需要挂载模型和数据目录。昇腾设备需通过--device映射。",
                "昇腾科学计算推理，关注数值精度和计算一致性。",
            ),
            ("scientific_computing", "fine_tuning"): (
                "python finetune_scientific.py --device_target Ascend --pretrained_model /pretrained/ --dataset /data/ --epochs 20",
                "需要挂载预训练模型和数据目录。昇腾设备需通过--device映射。",
                "昇腾科学模型微调，注意物理约束的保持。",
            ),
            ("scientific_computing", "reinforcement_training"): (
                "python rl_train_scientific.py --device_target Ascend --algorithm SAC --env physics_simulation",
                "昇腾设备需通过--device映射。物理仿真环境计算密集。",
                "昇腾科学计算强化学习，物理仿真环境可能计算密集。",
            ),
            ("retrieval_matching", "operator"): (
                "python ascend_retrieval_ops.py --test_similarity_ops --device_target Ascend --batch_size 512",
                "昇腾需要映射NPU设备。",
                "昇腾检索算子测试，向量相似度计算可使用专用算子加速。",
            ),
            ("retrieval_matching", "training"): (
                "python train_retrieval.py --device_target Ascend --model retrieval_net --dataset_path /data/ --epochs 25",
                "需要挂载数据目录。昇腾设备需通过--device映射。",
                "昇腾检索模型训练，对比学习建议使用大batch_size。",
            ),
            ("retrieval_matching", "inference"): (
                "python inference_retrieval.py --device_target Ascend --model_path /models/ --query 'search query'",
                "需要挂载模型目录。昇腾设备需通过--device映射。",
                "昇腾检索推理，向量检索建议使用ANN索引。",
            ),
            ("retrieval_matching", "fine_tuning"): (
                "python finetune_retrieval.py --device_target Ascend --pretrained_model /pretrained/ --dataset /data/ --epochs 15",
                "需要挂载预训练模型和数据目录。昇腾设备需通过--device映射。",
                "昇腾检索模型微调，使用领域数据提升检索质量。",
            ),
            ("retrieval_matching", "reinforcement_training"): (
                "python rl_train_retrieval.py --device_target Ascend --algorithm DQN --env recommendation_system",
                "昇腾设备需通过--device映射。",
                "昇腾推荐系统强化学习，在线学习需要注意exploration策略。",
            ),
        }

        for (scenario, task), (task_cmd, start_hints, task_hints) in skills.items():
            self.register(chip, scenario, task, SkillConfig(
                image_name=image,
                start_command=start_cmd,
                task_command=task_cmd,
                start_command_hints=start_hints,
                task_command_hints=task_hints,
            ))


_CHIP_DEFAULTS = {
    "NVIDIA_H200": {
        "start_command": "docker run -d --gpus all",
        "default_image": "nvidia/cuda:12.2-runtime-ubuntu20.04",
    },
    "Ascend_910B": {
        "start_command": "docker run -d --device=/dev/davinci0",
        "default_image": "mindspore/mindspore:2.0-cann6.3",
    },
}
