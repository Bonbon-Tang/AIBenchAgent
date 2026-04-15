#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from typing import Any, Dict, Tuple


def build_service_eval_script(context: Dict[str, Any]) -> Tuple[str, str]:
    image_config = context.get('image_config', {})
    service_profile = image_config.get('service_profile', {})
    base_url = service_profile.get('base_url', image_config.get('base_url', 'http://127.0.0.1:18080')).rstrip('/')
    healthcheck_path = service_profile.get('healthcheck_path', '/v1/models')
    chat_path = service_profile.get('chat_path', '/v1/chat/completions')
    serve_command = service_profile.get('serve_command', '')
    env_map = service_profile.get('env', {}) or {}
    model_id = image_config.get('model_id', '')
    container_name = image_config.get('container_name', context.get('container_name', ''))
    test_case = context.get('test_case', 'llm_throughput')

    env_exports = "\n".join([f"export {k}={json.dumps(str(v))}" for k, v in env_map.items()])

    script = f'''#!/bin/bash
set -euo pipefail

mkdir -p /workspace/results
START_TS=$(date +%s)
BASE_URL={json.dumps(base_url)}
HEALTH_URL="$BASE_URL{healthcheck_path}"
CHAT_URL="$BASE_URL{chat_path}"
MODEL_ID={json.dumps(model_id)}
TEST_CASE={json.dumps(test_case)}
CONTAINER_NAME={json.dumps(container_name)}

{env_exports}

health_ok=0
if curl -fsS "$HEALTH_URL" >/tmp/aibench_healthcheck.json 2>/tmp/aibench_healthcheck.err; then
  health_ok=1
fi

if [ "$health_ok" -ne 1 ]; then
  if [ -n {json.dumps(serve_command)} ]; then
    nohup bash -lc {json.dumps(serve_command)} >/tmp/aibench_vllm.log 2>&1 &
  fi

  for i in $(seq 1 60); do
    if curl -fsS "$HEALTH_URL" >/tmp/aibench_healthcheck.json 2>/tmp/aibench_healthcheck.err; then
      health_ok=1
      break
    fi
    sleep 5
  done
fi

if [ "$health_ok" -ne 1 ]; then
  python3 - <<'PY'
import json
from pathlib import Path
Path('/workspace/results').mkdir(parents=True, exist_ok=True)
Path('/workspace/results/result.json').write_text(json.dumps({{
  'status': 'failed',
  'smoke_passed': False,
  'success_count': 0,
  'fail_count': 1,
  'avg_latency_ms': None,
  'p95_latency_ms': None,
  'error': 'service healthcheck failed',
}}, ensure_ascii=False, indent=2), encoding='utf-8')
PY
  exit 1
fi

python3 - <<'PY'
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

base_url = {json.dumps(base_url)}
chat_url = base_url + {json.dumps(chat_path)}
model_id = {json.dumps(model_id)}
health_payload = Path('/tmp/aibench_healthcheck.json').read_text(encoding='utf-8', errors='ignore') if Path('/tmp/aibench_healthcheck.json').exists() else ''

prompts = [
    '请用一句话介绍 Transformer。',
    '请简述大模型镜像评测为什么要同时看可用性和性能。',
    '请输出一个三点列表，说明在 H200 上部署 vLLM 需要关注什么。',
    '请用两句话解释什么是 KV Cache。',
    '请简要说明为什么 host network 适合内网 GPU 评测。',
]

latencies = []
success_count = 0
fail_count = 0
samples = []

for prompt in prompts:
    payload = json.dumps({{
        'model': model_id,
        'messages': [{{'role': 'user', 'content': prompt}}],
        'temperature': 0,
        'max_tokens': 128,
    }}).encode('utf-8')
    req = urllib.request.Request(chat_url, data=payload, headers={{'Content-Type': 'application/json'}})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode('utf-8', errors='ignore')
            latency_ms = round((time.time() - start) * 1000, 2)
            latencies.append(latency_ms)
            success_count += 1
            samples.append({{'prompt': prompt, 'latency_ms': latency_ms, 'response_preview': body[:300]}})
    except Exception as e:
        latency_ms = round((time.time() - start) * 1000, 2)
        fail_count += 1
        samples.append({{'prompt': prompt, 'latency_ms': latency_ms, 'error': str(e)}})

avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None
if latencies:
    ordered = sorted(latencies)
    idx = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered) + 0.5)) - 1))
    p95 = ordered[idx]
else:
    p95 = None

result = {{
    'status': 'completed' if success_count > 0 else 'failed',
    'workspace_mode': 'reuse_or_workspace_container',
    'container_name': {json.dumps(container_name)},
    'base_url': base_url,
    'model_id': model_id,
    'test_case': {json.dumps(test_case)},
    'smoke_passed': success_count > 0,
    'success_count': success_count,
    'fail_count': fail_count,
    'avg_latency_ms': avg_latency,
    'p95_latency_ms': p95,
    'samples': samples,
    'healthcheck_preview': health_payload[:500],
}}
Path('/workspace/results').mkdir(parents=True, exist_ok=True)
Path('/workspace/results/result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
PY
'''
    return 'service_eval.sh', script
