# Gather (tensormap_and_ringbuffer)

纯 gather 通信算子，使用 tensormap_and_ringbuffer 运行时（设备端编排）。

流程：WindowMemCopyIn → CommBarrier → TGATHER → WindowMemCopyOut (root only)

与 host_build_graph/gather 的区别：
- 使用 `aicpu_orchestration_entry` 设备端编排
- 内核使用 TensorData 接口（buffer.addr）
- 非 root rank 使用 ZeroBuffer 初始化输出以通过校验

## 运行

```bash
cd simpler-PTO
python examples/scripts/multi_card_run_example.py \
    -k examples/tensormap_and_ringbuffer/gather/kernels \
    -g examples/tensormap_and_ringbuffer/gather/golden.py
```

或使用便捷脚本：
```bash
./run_tensormap.sh gather 2 0
```

需要设置 `PTO_COMM_ISA_ROOT` 指向 pto-comm-isa 根目录，以及多卡 HCCL 环境。
