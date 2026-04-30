# AGENTS.md

This file provides guidance to codex when working in this repository.

## 沟通约定

- 与用户讨论架构、方案、文档时，默认使用中文。
- 代码注释与标识符风格，默认遵循当前代码文件已有风格。
- 新增代码默认使用驼峰风格命名；只有在需要兼容第三方接口、Python 特定约定或历史 YAML 字段时例外。
- 与 codex 相关的草稿、计划、记忆文件，如果将来需要新增，统一放在 `.codex/` 下；但要先确认是否真的需要保留，避免再次堆积历史文档。
- 目前不需要向后兼容，正在开发阶段。
- `Problem.evaluate` 的正式返回协议现在是 `Eval`，不是 `dict`。后续如果在触达文件中遇到 `res['objs']` / `res['cons']` 这类旧用法，顺手改成 `res.objs` / `res.cons`，不需要专门做一次全仓库扫描。
- `problem` 模块旧 transform 接口不再作为正式接口维护。后续如果在触达文件中扫到旧名字，顺手替换即可，不需要专门做一次全仓库扫描：
  `_transform_unit_X` -> `unit_to_space`
  `_transform_to_I_D` -> `apply_var_type`
  `_transform_int_var` -> `cast_int_vars`
  `_transform_discrete_var` -> `map_discrete_vars`
