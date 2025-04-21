# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import sys
import subprocess
from datetime import datetime
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.wrappers.history_wrapper import HistoryWrapper
import time

# 配置部分
CHECK_INTERVAL = 10  # 检查间隔（秒）
GPU_USAGE_THRESHOLD = 99  # GPU 使用率阈值（%）
MEMORY_USAGE_THRESHOLD = 500  # 显存使用量阈值（MB）
MEMORY_CAPACITY = 24268  # 显存总容量（MB）

def check_gpu_status():
    try:
        # 获取 nvidia-smi 的输出
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits", 
            shell=True
        )
        gpu_status_list = output.decode("utf-8").strip().split("\n")
        
        idle_gpu = None
        status_lines = []

        for i, gpu_status in enumerate(gpu_status_list):
            utilization, memory = map(int, gpu_status.split(","))
            memory_degree = round(memory / MEMORY_CAPACITY * 100, 2)
            status_line = f"GPU {i}: 使用率 {utilization}%，显存占用 {memory_degree}%（{memory} MB）"
            status_lines.append(status_line)

            # 判断是否空闲
            if utilization < GPU_USAGE_THRESHOLD and memory < MEMORY_USAGE_THRESHOLD:
                idle_gpu = i

        # 清屏并打印状态
        sys.stdout.write("\033c")  # 清屏
        print("当前显卡使用情况:")
        for line in status_lines:
            print(line)

        return idle_gpu
    except Exception as e:
        print(f"检测显卡状态时发生错误: {e}")
        return None

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env = HistoryWrapper(env)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    print("显卡监控已启动，正在检测显卡状态...")
    
    while True:
        gpu_id = check_gpu_status()
        if gpu_id is not None:
            print(f"\n检测到显卡 {gpu_id} 空闲，开始训练！")
            train(args)
            break
        else:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 无空闲显卡，继续检测...")
        time.sleep(CHECK_INTERVAL)