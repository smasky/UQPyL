# 2026-09-18 本机开发环境配置

后续更新：完成 [MARS 编译兼容性修复](0918-mars-gcc15-fix.md) 后，`py312` 已升级为 GCC/G++ 15.2，并严格重建可编辑安装。下文 GCC 11.2 的安装经过是修复前记录，修复后的源码无需降级编译器。

- 源码基线：`dev` / `f07f058`。
- 环境：`/home/wmtsky/anaconda3/envs/py312`，Python 3.12.0。
- 已安装 conda GCC/G++ 11.2，以及可编辑模式的 `UQPyL[viz]`、pytest、pytest-cov、build、Cython、pybind11。
- 本次运行依赖：NumPy 2.5.3、SciPy 1.18.1。
- 默认源安装的 GCC 15.2 在编译 MARS 的 Cython 生成代码时因不兼容指针类型报错；当前源没有 GCC 13，改用 GCC 11.2 后构建成功。未修改源码或关闭编译诊断。

本机使用：

```bash
conda activate py312
python -m pytest -q
```

环境重建的主要安装命令：

```bash
conda install -n py312 -y 'gcc_linux-64=11.2' 'gxx_linux-64=11.2'
conda run -n py312 python -m pip install -e '.[viz]' pytest pytest-cov build Cython pybind11
```

验证：`pip check` 通过；10 个原生扩展全部成功导入；确认 UQPyL 从当前仓库加载。
完整测试：**1345 passed，4 warnings，22.86 秒**，4 条均为既有 FAST 辅助频率复用警告。
[测试日志](verification/0918-new-machine-py312-pytest.txt)。

本次验证为本机可编辑安装，不代表远程 CI 或独立 wheel 验证。文档构建依赖本次未安装。
