#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_PATH = ROOT / 'config' / 'projectten_v2_mapping.json'
DEFAULT_ASSETS_PATH = ROOT / 'config' / 'projectten_assets.local.json'


class ProjectTenV2Adapter:
    def __init__(self, mapping_path: str = None, assets_path: str = None):
        self.mapping_path = Path(mapping_path) if mapping_path else DEFAULT_MAPPING_PATH
        self.assets_path = Path(assets_path) if assets_path else DEFAULT_ASSETS_PATH
        self.mapping = self._load_json(self.mapping_path)
        self.assets = self._load_json(self.assets_path)

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def is_projectten_v2_config(config: Dict[str, Any]) -> bool:
        required = ['task', 'scenario', 'chips', 'chip_num']
        return all(k in config for k in required)

    def normalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_projectten_v2_config(config):
            return config

        task = config.get('task')
        scenario = config.get('scenario')
        chips = config.get('chips')
        chip_num = config.get('chip_num', 1)
        image_id = config.get('image_id')
        tool_id = config.get('tool_id')

        route_key = f'{task}:{scenario}'
        route = (self.mapping.get('routes') or {}).get(route_key)
        if not route:
            raise ValueError(f'未找到 ProjectTen v2 到 AIBench 的路由映射: {route_key}')

        chip_type = (self.mapping.get('chips') or {}).get(chips)
        if not chip_type:
            raise ValueError(f'未找到 chips 到 AIBench chip_type 的映射: {chips}')

        normalized = dict(config)
        normalized.update({
            'projectten_task': task,
            'projectten_scenario': scenario,
            'projectten_chips': chips,
            'projectten_chip_num': chip_num,
            'projectten_image_id': image_id,
            'projectten_tool_id': tool_id,
            'chip_type': chip_type,
            'application_scenario': route['application_scenario'],
            'task_type': route['task_type'],
            'task_scenario': route.get('task_scenario', route['task_type']),
            'card_count': chip_num,
            'test_case': route.get('default_test_case', 'projectten_default'),
        })

        image_config = self._build_image_config(normalized)
        if image_config:
            normalized['image_config'] = image_config

        return normalized

    def _build_image_config(self, normalized: Dict[str, Any]) -> Dict[str, Any]:
        task_type = normalized.get('task_type')
        application_scenario = normalized.get('application_scenario')
        chip_type = normalized.get('chip_type')
        image_id = normalized.get('projectten_image_id')
        tool_id = normalized.get('projectten_tool_id')

        route_key = f"{normalized.get('projectten_task')}:{normalized.get('projectten_scenario')}"
        route = (self.mapping.get('routes') or {}).get(route_key, {})

        result: Dict[str, Any] = {
            'projectten_source': True,
            'projectten_task': normalized.get('projectten_task'),
            'projectten_scenario': normalized.get('projectten_scenario'),
            'projectten_image_id': image_id,
            'projectten_tool_id': tool_id,
            'chip_type': chip_type,
            'application_scenario': application_scenario,
            'task_type': task_type,
        }

        images = (self.assets.get('images') or {})
        tools = (self.assets.get('tools') or {})
        image_meta = images.get(str(image_id), {}) if image_id is not None else {}
        tool_meta = tools.get(str(tool_id), {}) if tool_id is not None else {}

        result.update(route.get('image_config_defaults', {}))
        result.update(image_meta)

        env = {}
        env.update(route.get('image_config_defaults', {}).get('environment', {}))
        env.update(image_meta.get('environment', {}))
        if env:
            result['environment'] = env

        volumes = []
        for source in [route.get('image_config_defaults', {}), image_meta]:
            for volume in source.get('volumes', []) or []:
                if volume not in volumes:
                    volumes.append(volume)
        if volumes:
            result['volumes'] = volumes

        if tool_meta.get('test_case') and normalized.get('test_case') == 'projectten_default':
            normalized['test_case'] = tool_meta['test_case']

        if tool_meta.get('task_command') and not result.get('task_command'):
            result['task_command'] = tool_meta['task_command']
        if tool_meta.get('task_command_hints'):
            existing = result.get('task_command_hints', '')
            hint = tool_meta['task_command_hints']
            result['task_command_hints'] = f'{existing}\n{hint}'.strip() if existing else hint

        return result
