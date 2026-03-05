# libhccl_helper

C++ 辅助库，与 pto-comm-isa 相同方式链接（ascendcl、hcomm、runtime），供 Python 通过 ctypes 调用。不依赖 Python 侧直接加载 libacl.so/libhccl.so。

## 编译

在已 source CANN 环境（与跑 pto-comm-isa comm case 相同）下：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
cd examples/scripts/hccl_helper
mkdir build && cd build
cmake ..
make
```

生成 `libhccl_helper.so`。运行多卡脚本前同样需要 `source .../setenv.bash`，以便运行时找到 ascendcl、hcomm、runtime 等依赖。

## 依赖

- CANN：需设置 `ASCEND_HOME_PATH`（一般由 `setenv.bash` 设置）
- 与 pto-comm-isa 的 a2a3 comm 用例相同

## Python 使用

`hccl_bindings.py` 会优先在以下位置查找 `libhccl_helper.so`：

- `examples/scripts/hccl_helper/build/libhccl_helper.so`
- `examples/scripts/build/libhccl_helper.so`
- `examples/scripts/libhccl_helper.so`

找到则使用 C++ 实现；未找到则报错并提示先编译本库。
