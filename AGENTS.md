## 目标

这几天我负责调研实现sgl上的 dsv4 的hybrid hisparse

hybrid hisparse 可参考 vllm 中的当前分支。native hisparse 中，把 dsa attn 的非 topk 部分均存储在 host memory 中，计算出 topk 后再从 host memory 中取出。 

hybrid hisparse 则在 gpu 压力不大时，默认全部存储在 kv cache 中，gpu 压力大时，才将非 topk 部分存储在 host memory 中。

实现过程中，你可以参考
+ vllm 的当前分支
+ sgl 的 pr #35488

## 环境说明

目前我在自己的 mac 上开发. gpuq 是内网服务器的 gpu 资源调度器，给出 gpuq 命令，才能在内网服务器上运行。可查看../gpuq

内网使用的是 conda 管理环境，使用 /home/jovyan/whw/whw_dev 这个conda环境

在内网中，家目录在`/home/jovyan/whw/`，打算使用 `~/models/DeepSeek-V4-Flash-0731` 模型和 `~/datasets/gsm8k` 数据集作为准确度验证的基准。

my_scripts 放测试脚本
my_development 记录开发过程

有时候我可用H20开发(141G)，有时候可用H100(80G)，并且hisparse默认需要PD分离场景

目前任务是最快验证deepseekV4的功能，不需要做兼容设计