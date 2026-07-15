# AIBenchAgent
人工智能软硬件验证平台 评测智能体

> 项目处于快速演进阶段，功能和接口可能随时调整。

## 为什么用 AIBenchAgent

- **零人工介入**：从配置收集、环境创建、脚本生成到结果分析，全流程由 Agent 自动完成，消除手工环节，提高评测效率
- **多芯片覆盖**：适配 NVIDIA GPU、华为昇腾等不同硬件平台，一次定义、多端运行
- **场景丰富**：算子、训练、推理、微调、强化学习——开箱即用
- **自我修复**：执行失败时自动诊断原因并重试，无需人工排查，提升成功率
- **持续进化**：Agent 积累每次执行的成功与失败经验，自动优化后续评测策略（规划中）
- **性能调优闭环**：未来支持根据评测结果自动生成调优建议并回归验证，形成"评测→诊断→优化→再评测"闭环（规划中）

## 快速开始

```bash
pip install -r requirements.txt

# 交互模式
python agent.py

# 指定配置文件
python agent.py --config config/task/h200_language_operator.json
```

## 910B 执行节点 HTTP 服务

`serve.py` 将 AIBenchAgent 作为 ProjectTen 的异步执行节点运行。该入口不调用
LLM；管理节点提交任务后，服务在本机启动 Docker 容器，并持久化任务状态、日志和
结果。

### 启动

服务必须部署在能够直接访问 Ascend 设备和 Docker daemon 的 910B 宿主机：

```bash
python3 -m pip install -r requirements.txt

export AIBENCH_DATA_DIR=/var/lib/aibench/jobs
export AIBENCH_MAX_WORKERS=2
export AIBENCH_ALLOWED_VOLUME_ROOTS=/models,/opt/aibench/workloads,/usr/local/Ascend

python3 serve.py --host 0.0.0.0 --port 8080 \
  --data-dir /var/lib/aibench/jobs --workers 2
```

部署前检查：

```bash
docker --version
npu-smi info
ls -l /dev/davinci0 /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
curl http://127.0.0.1:8080/health
```

生产环境应通过防火墙或反向代理只允许 ProjectTen 管理节点访问该端口。

### API

```text
POST /api/v1/jobs                 提交任务，返回 job_id
GET  /api/v1/jobs/{job_id}        查询状态和进度
GET  /api/v1/jobs/{job_id}/result 获取完成结果
GET  /api/v1/jobs/{job_id}/logs   获取容器日志
POST /api/v1/jobs/{job_id}/cancel 取消任务并删除容器
```

请求示例见：

```text
config/task/projectten_910b_service_job.json
```

提交任务：

```bash
curl -sS -X POST http://127.0.0.1:8080/api/v1/jobs \
  -H 'Content-Type: application/json' \
  --data @config/task/projectten_910b_service_job.json
```

容器执行契约：

1. Agent 将每个任务的独立目录挂载为 `/workspace/results`。
2. `image.command` 必须执行完整评测并退出，不能只启动永久驻留服务。
3. 成功退出前必须写 `/workspace/results/result.json`。
4. 结果至少包含 `status` 和 `metrics` 对象，例如：

```json
{
  "status": "completed",
  "metrics": {
    "throughput": 31.8,
    "throughput_unit": "tokens/s",
    "avg_latency_ms": 412.3,
    "p95_latency_ms": 471.6
  }
}
```

缺失/非法的结果文件、非零退出码、超时以及结果声明失败都会使 Job 失败，不会生成
模拟成功结果。任务数据保存在：

```text
<AIBENCH_DATA_DIR>/<job_id>/request.json
<AIBENCH_DATA_DIR>/<job_id>/state.json
<AIBENCH_DATA_DIR>/<job_id>/result.json
<AIBENCH_DATA_DIR>/<job_id>/container.log
```

Ascend 默认映射 `device_ids` 对应的 `/dev/davinciN`，以及
`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`。`gpus` 默认为
`null`，不会给 Ascend 容器添加 NVIDIA 的 `--gpus all`。任务提交者只能挂载
`AIBENCH_ALLOWED_VOLUME_ROOTS` 中声明的宿主机目录。

## ProjectTen v2 → AIBenchAgent（H200 / tangyufeng）执行说明

这条链路面向当前的真实使用方式：

```text
ProjectTen 输出 v2 JSON
→ AIBenchAgent 在开发机更新代码/整理任务
→ scp 到内网 GPU 机器
→ 优先复用 H200 上已有 tangyufeng 工作容器
→ 在容器内启动/检查 vLLM 服务
→ 完成接口评测并输出 result.json
```

### 1. 当前默认设计

针对以下 ProjectTen v2 任务：

```json
{
  "task": "model_deployment",
  "scenario": "llm",
  "chips": "nvidia_h200",
  "chip_num": 1,
  "image_id": 70,
  "tool_id": 27
}
```

AIBenchAgent 当前会自动转换为 H200 的 `tangyufeng` 风格执行方式：

- 优先复用容器：`tangyufeng`
- 镜像：`registry.h.pjlab.org.cn/ailab-pj/vllm:0.16.2rc2.g21dfb842d.cu128`
- 网络：`host`
- IPC：`host`
- 挂载：`/mnt/nvme1n1:/mnt/nvme1n1`
- 服务地址：`http://127.0.0.1:18080`
- 健康检查：`/v1/models`
- 推理接口：`/v1/chat/completions`

相关本地映射文件：

- `config/projectten_v2_mapping.json`
- `config/projectten_assets.local.json`

### 2. 推荐执行场景

适合以下环境：

- ProjectTen 负责输出标准 v2 JSON
- 开发机可以联网拉代码
- 内网 GPU 机器不方便联网 / 不方便现场安装依赖
- H200 上已有长期驻留工作容器 `tangyufeng`
- 模型目录位于：`/mnt/nvme1n1/...`

### 3. 在开发机准备

确保本地代码为最新：

```bash
git pull
```

准备一个 ProjectTen v2 配置文件，或直接使用仓库中的 demo：

```bash
config/task/projectten_v2_llm_h200_demo.json
```

示例：

```json
{
  "task": "model_deployment",
  "scenario": "llm",
  "chips": "nvidia_h200",
  "chip_num": 1,
  "image_id": 70,
  "tool_id": 27,
  "name": "ProjectTen-v2-llm-demo"
}
```

如果需要把代码传到内网 GPU 机器，推荐直接 scp 整个目录（当前版本尚未收口为 bundle）：

```bash
scp -r AIBenchAgent <user>@<gpu-host>:/path/to/
```

### 4. 在 H200 / GPU 机器执行前检查

先检查基础环境：

```bash
docker --version
nvidia-smi
python3 --version
```

再检查 `tangyufeng` 容器是否存在：

```bash
docker ps -a | grep tangyufeng
```

如果 `tangyufeng` 已存在，AIBenchAgent 当前会优先尝试复用它。

如果容器内服务尚未启动，当前固定模板会按 `service_profile`：

- 先检查 `http://127.0.0.1:18080/v1/models`
- 若不可用，再在容器内执行 `vllm serve ...`
- 然后调用 `/v1/chat/completions` 做样例推理测试
- 统计成功数、失败数、平均延迟、P95
- 输出到 `/workspace/results/result.json`

### 5. 执行方式

在 AIBenchAgent 目录下执行：

```bash
python3 agent.py --config config/task/projectten_v2_llm_h200_demo.json
```

如果你已经拿到了 ProjectTen 导出的 v2 JSON，也可以直接传它：

```bash
python3 agent.py --config /path/to/projectten_task.json
```

### 6. 当前 H200 模型镜像评测行为

当输入命中：

- `task=model_deployment`
- `scenario=llm`
- `chips=nvidia_h200`

AIBenchAgent 当前保留原有测试主流程：

- Collector 加载配置
- Creator 处理容器（优先复用 `tangyufeng`）
- Executor 执行评测脚本
- Controller 收集结果

但对于包含 `service_profile` 的任务，Executor 会优先使用固定模板脚本，而不是完全依赖 LLM 自由生成。这样做的目的是：

- 保留 AIBenchAgent 主流程
- 降低 H200 / 内网环境下脚本漂移风险
- 更稳定地复用 `tangyufeng` 和 `http://127.0.0.1:18080`

### 7. 输出结果

执行完成后，AIBenchAgent 会尝试读取：

```text
/workspace/results/result.json
```

当前结果中重点字段包括：

- `status`
- `smoke_passed`
- `success_count`
- `fail_count`
- `avg_latency_ms`
- `p95_latency_ms`
- `base_url`
- `model_id`
- `container_name`

### 8. 常见问题

#### H200 上没有 `pip`

当前推荐直接使用：

```bash
python3 agent.py --config ...
```

对于当前 `service_profile + tangyufeng` 路径，AIBenchAgent 已改成优先走低依赖固定模板执行，不再要求 GPU 机先具备 `backoff` / LLM 依赖链才能启动。

如果 GPU 机没有 `pip/pip3`，不要优先折腾宿主机 Python 环境；优先保证：

- Docker 可用
- `tangyufeng` 容器可用
- `python3` 可执行

#### `tangyufeng` 已存在但没挂载 `/mnt/nvme1n1`

Docker 已创建容器不能追加挂载。需要：

1. 确认旧容器是否已经挂载
2. 如果没挂载，删除旧容器后重建
3. 使用：

```bash
-v /mnt/nvme1n1:/mnt/nvme1n1
```

#### 18080 接口不可用

先在宿主机验证：

```bash
curl http://127.0.0.1:18080/v1/models
```

如果不通，优先检查：

- `tangyufeng` 是否启动
- 容器内模型目录是否存在
- `FLASHINFER_DISABLE_VERSION_CHECK=1` 是否已设置
- GPU 是否被其他任务占满

#### 当前版本是否完全离线 bundle 化

还没有。当前版本先优先打通：

- ProjectTen v2 JSON
- AIBenchAdapter 转换
- `tangyufeng` 复用
- H200 固定模板评测

后续可以继续演进为：

- 开发机构建离线执行包
- scp 单个 bundle 到 GPU 机
- GPU 机一键执行

## 许可

内部项目，暂不对外开放。
