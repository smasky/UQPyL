# UQPyL 架构评审建议

## 总体评价

从 `UQPyL.zip` 的静态结构来看，项目已经有比较清晰的领域模块划分：

```text
problem / doe / optimization / surrogate / analysis / inference / calibration / viz
```

这说明项目的主线是明确的：定义问题、采样、优化、代理模型、分析、推断、校准和可视化。

但目前更大的问题不在于“算法是否够多”，而在于：

- 模块依赖方向还不够稳定；
- 运行时基础设施重复；
- 配置体系不统一；
- `Problem` 抽象还没有完全收敛；
- 发布包和工程化结构不够干净；
- 可选依赖、native extension、持久化等机制还没有形成稳定规范。

一句话总结：**领域模块拆分方向是对的，但现在像是“功能包已经分好，基础设施还没抽出来”。**

---

## 1. 模块依赖方向存在潜在循环

目前看到一些比较危险的依赖关系：

```text
optimization/base.py  -> doe.LHS
surrogate/GPR、KRG    -> optimization.AlgorithmABC / GA
optimization/expensive -> surrogate
problem/mop/*          -> optimization.core
```

这会带来几个问题：

1. 包导入顺序变脆弱；
2. 可选模块不容易拆分；
3. 优化算法和代理模型互相绑定；
4. 单元测试难以隔离；
5. 后续插件化会比较困难。

建议把依赖方向收敛为：

```text
core
  ↓
problem / space / eval
  ↓
doe / optimization / surrogate / analysis / inference / calibration
  ↓
viz
```

其中：

- `core` 只提供基础协议、配置、运行上下文、结果对象；
- `problem` 只定义问题和评价协议；
- `doe / optimization / surrogate` 可以依赖 `core` 和 `problem`；
- `surrogate` 不应该直接依赖具体优化算法；
- `viz` 可以作为最外层模块，依赖其他模块，但其他模块不要依赖 `viz`。

### 建议做法

把 `surrogate -> optimization.GA` 这种直接依赖改成协议注入：

```python
class OptimizerProtocol(Protocol):
    def run(self, problem, seed: int | None = None) -> Result:
        ...
```

代理模型内部如果需要优化超参数，不要硬编码 `GA`，而是接收一个 optimizer：

```python
model = GPR(optimizer=GA(...))
```

或者通过 registry 获取：

```python
optimizer = registry.get_optimizer("GA")
```

---

## 2. runtime / result / storage / verbose 重复严重

现在多个模块中都出现了类似概念：

```text
analysis/runtime
inference/runtime
optimization/runtime
```

里面都有相似的：

- `Result`
- `State`
- `SqliteStorage`
- `Reader`
- `Verbose`

这说明这些能力本质上是跨模块基础设施，不应该分别散落在各业务模块里。

长期下去会导致：

1. 每个模块的日志格式不同；
2. 每个模块的结果对象不同；
3. 每个模块的存储 schema 不同；
4. 回放、恢复、可视化、分析历史结果会很麻烦；
5. 新增模块时会继续复制一套 runtime。

### 建议抽象为统一运行时层

建议新增：

```text
core/runtime/
  context.py
  result.py
  state.py
  storage.py
  events.py
  verbose.py
```

职责可以这样划分：

```text
RunContext
  - run_id
  - seed
  - work_dir
  - rng
  - storage
  - reporter

BaseResult
  - summary
  - metrics
  - artifacts
  - history reference

StorageBackend
  - save_metadata()
  - save_iteration()
  - save_artifact()
  - load_result()

Reporter
  - on_start()
  - on_iter()
  - on_finish()
```

各算法模块只需要产生自己的领域状态，例如：

```python
OptimizationState
InferenceState
AnalysisState
```

不要让每个模块都自己处理 sqlite、日志、路径、输出格式。

---

## 3. `Problem` 抽象不够稳定

现在 `ProblemBase` 主要围绕优化问题设计，例如：

- `objs`
- `cons`
- `nObj`
- `nCon`
- `optType`

但 `ModelProblem` 又引入了：

- `sim`
- `nOutput`

并且存在初始化后再改写 `nObj / optType / opt` 的情况。

这说明当前 `ProblemBase` 对两类问题的统一表达不够自然：

1. 优化型问题；
2. 仿真 / 模型型问题。

### 当前风险

如果继续把所有问题都塞进同一个 `ProblemBase`，后面会出现越来越多特殊字段：

```text
objs / cons / sim / outputs / metrics / likelihood / posterior / constraints
```

最后 `ProblemBase` 会变成一个大而全但不稳定的接口。

### 建议方案一：分层抽象

```text
ProblemBase
  OptimizationProblem
  SimulationProblem
  CalibrationProblem
```

例如：

```python
class ProblemBase:
    def evaluate(self, x):
        raise NotImplementedError


class OptimizationProblem(ProblemBase):
    def evaluate(self, x) -> OptimizationEval:
        ...


class SimulationProblem(ProblemBase):
    def evaluate(self, x) -> SimulationEval:
        ...
```

### 建议方案二：统一输出 channel

如果想保持统一接口，可以把输出定义成 channel：

```python
@dataclass
class EvalResult:
    channels: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
```

例如：

```python
EvalResult(
    channels={
        "objs": objs,
        "cons": cons,
        "sim": sim,
    }
)
```

这样优化、仿真、推断、校准都可以用同一个评价结果对象，但不用在 `ProblemBase` 里硬编码所有字段。

---

## 4. 参数系统不统一

目前至少存在两套参数体系：

### `core.Params`

偏简单 dict 风格，同时有多个兼容方法：

```python
setParaVal
setParaValue
getParaValue
setPara
```

### `surrogate.Setting`

功能更复杂，包含：

- 上下界；
- 参数类型；
- choices；
- log scale；
- owner；
- 是否可优化等。

这会造成：

1. 优化算法、代理模型、分析方法的配置风格不一致；
2. 配置序列化困难；
3. 自动调参困难；
4. GUI 或配置文件生成困难；
5. 参数校验逻辑会重复。

### 建议统一配置模型

可以抽象成：

```python
@dataclass
class ParameterSpec:
    name: str
    value: Any
    lower: float | None = None
    upper: float | None = None
    kind: Literal["float", "int", "choice", "bool", "constant"] = "float"
    choices: list[Any] | None = None
    log_scale: bool = False
    owner: str | None = None
    description: str | None = None
```

然后所有模块都使用统一接口：

```python
params.get("max_iter")
params.set("max_iter", 100)
params.validate()
params.to_dict()
params.from_dict(...)
```

### 重要原则

配置和运行状态要分开：

```text
Config / Params:
  - 用户输入
  - 算法超参数
  - 可序列化

State:
  - 当前迭代
  - 当前种群
  - 当前损失
  - 当前模型状态

Result:
  - 最终输出
  - 历史摘要
  - artifacts
```

不要让 `params / setting / state / result` 混在一个对象里。

---

## 5. 抽象基类约束不够强

当前一些类虽然名字叫 `ABC`，但并没有真正形成强约束。

例如：

```python
class AlgorithmABC:
    def run(...):
        pass
```

或者：

```python
class InferenceABC:
    def run(...):
        pass
```

如果子类忘记实现核心方法，不会在实例化阶段报错，而是运行到某个分支才出问题。

### 建议改法

使用 `abc.ABC` 和 `@abstractmethod`：

```python
from abc import ABC, abstractmethod

class AlgorithmABC(ABC):
    @abstractmethod
    def run(self, problem, seed: int | None = None):
        raise NotImplementedError
```

对于不想强继承的场景，可以用 `Protocol`：

```python
class SamplerProtocol(Protocol):
    def sample(self, n: int, bounds) -> np.ndarray:
        ...
```

### 建议优先补强的接口

```text
AlgorithmABC.run()
InferenceABC.run()
ProblemBase.evaluate()
ProblemBase.get_optimum()
StorageBackend.save/load()
SurrogateModel.fit/predict()
```

---

## 6. 包入口隐藏错误

`UQPyL/__init__.py` 里有大量类似逻辑：

```python
try:
    surrogate = importlib.import_module(...)
except Exception:
    surrogate = None
```

这个问题比较严重，因为它会吞掉真实错误。

例如：

- 依赖缺失；
- 语法错误；
- 内部 import 错误；
- native extension 加载失败；
- 拼写错误；
- 循环导入。

最后用户只看到某个模块是 `None`，但不知道为什么。

### 建议改法

只捕获明确的可选依赖异常：

```python
try:
    import UQPyL.surrogate as surrogate
except ModuleNotFoundError as e:
    warnings.warn(f"Optional module surrogate is unavailable: {e}")
    surrogate = None
```

开发模式下最好直接抛错：

```python
if os.environ.get("UQPYL_DEV"):
    raise
```

更好的方式是做 lazy import 或 registry，不要在根入口 eager import 所有模块。

---

## 7. 发布包结构不干净

压缩包中包含一些不应该进入源码包的内容：

```text
__pycache__/
*.pyc
*.pyd
*.c
*.pyx
*.pxd
```

尤其是 Windows `.pyd` 文件和 Cython 生成产物混在源码包里，会影响跨平台安装和维护。

### 风险

1. Linux/macOS 用户无法直接使用 `.pyd`；
2. Python 版本不一致时 native extension 会失效；
3. 源码包和构建产物混在一起；
4. CI/CD 难以管理；
5. 发布到 PyPI 后容易出现平台兼容问题。

### 建议项目结构

```text
UQPyL/
  pyproject.toml
  README.md
  LICENSE
  src/
    UQPyL/
      __init__.py
      core/
      problem/
      doe/
      optimization/
      surrogate/
      analysis/
      inference/
      calibration/
      viz/
  tests/
  docs/
  examples/
```

构建产物单独放：

```text
dist/
  *.whl
  *.tar.gz
```

### `.gitignore` 建议

```gitignore
__pycache__/
*.py[cod]
*.pyd
*.so
*.dll
*.dylib
build/
dist/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
```

如果 Cython 是必须的，建议用标准构建流程管理：

```text
pyproject.toml
setup.py / setup.cfg
cibuildwheel
```

---

## 8. 持久化方案风险较大

当前 sqlite 存储中存在：

- pickle 保存 problem；
- pickle 保存 artifacts；
- repr(value) 保存参数；
- reader 中再反解析。

这短期方便，但长期风险很大。

### 主要风险

1. `pickle` 不安全；
2. Python 版本、类路径变化后无法兼容；
3. 数据库不可读性差；
4. schema 迁移困难；
5. 不利于跨语言或工具读取；
6. 不利于长期实验归档。

### 建议分层存储

```text
SQLite:
  - run_id
  - metadata
  - schema_version
  - package_version
  - timestamps
  - artifact paths
  - summary metrics

JSON:
  - config
  - params
  - problem metadata

NPZ / HDF5 / Parquet:
  - large arrays
  - population history
  - samples
  - posterior traces

Pickle:
  - 仅作为可选 cache
  - 不作为稳定存档格式
```

### schema 建议

至少加一个 metadata 表：

```sql
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);
```

里面保存：

```text
schema_version
package_version
created_at
python_version
platform
```

---

## 9. 算法实例状态太重，不利于复用和并发

很多算法对象会持有运行时状态：

```text
self.problem
self.rng
self.storage
self.storageCtx
self.state
self.result
self.runId
```

这会导致：

1. 同一个算法实例重复运行容易污染状态；
2. 并发运行困难；
3. 嵌套运行风险高；
4. surrogate 内部调用 optimizer 时会修改 optimizer 状态；
5. debug 时很难区分“配置”和“运行结果”。

### 建议原则

算法实例只保存配置：

```python
class GA:
    def __init__(self, pop_size, max_iter, ...):
        self.config = ...
```

每次运行时创建独立上下文：

```python
def run(self, problem, seed=None):
    ctx = RunContext(problem=problem, seed=seed)
    state = OptimizationState()
    ...
    return result
```

也就是说：

```text
Algorithm object:
  - config only

RunContext:
  - seed
  - rng
  - storage
  - reporter
  - run_id

State:
  - mutable iteration state

Result:
  - final output
```

---

## 10. 历史记录可能爆内存

`OptHistory` 会保存每一代完整 population，包括：

```text
decs
objs
cons
```

如果问题规模较大，或者迭代次数多，这会非常占内存，也会带来很大的 sqlite I/O 压力。

### 建议增加 history 策略

```python
history = "none"      # 不保存历史
history = "summary"   # 只保存指标摘要
history = "best"      # 只保存每代最优
history = "full"      # 保存完整 population
```

还可以支持：

```python
save_every = 10
keep_last = 5
```

默认建议使用：

```python
history="summary"
```

只有用户明确需要时才保存完整 population。

---

## 11. 命名和 API 风格不统一

当前存在一些命名不一致问题：

```text
nObj / nOutput
nCon / nCons
maxIter / maxIters
setParaVal / setParaValue / setPara
camelCase / snake_case 混用
guass_kernel.py 拼写错误
weightEnsemble 类名小写开头
```

这些不是最核心的架构问题，但会影响长期维护、文档和用户体验。

### 建议统一风格

Python 公共 API 建议统一为 snake_case：

```python
max_iter
n_obj
n_con
set_param()
get_param()
get_optimum()
```

旧接口可以暂时保留，但加 deprecation warning：

```python
def setParaValue(...):
    warnings.warn(
        "setParaValue is deprecated; use set_param instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.set_param(...)
```

建议做一个 2.x -> 3.x 的 API 迁移计划。

---

## 12. optional dependency / plugin 机制还没有成型

像这些模块：

```text
MARS
SVR
lasso
native extension
```

应该作为可选依赖，而不是默认全部导入。

### 建议使用 extras

```bash
pip install UQPyL[mars]
pip install UQPyL[svr]
pip install UQPyL[all]
```

在 `pyproject.toml` 中定义：

```toml
[project.optional-dependencies]
mars = ["..."]
svr = ["..."]
all = ["..."]
```

同时结合 registry：

```python
from UQPyL.registry import get_surrogate

model = get_surrogate("GPR")
```

这样核心包可以保持轻量，重依赖模块按需启用。

---

## 13. `viz` 应该作为外层模块

可视化模块建议只依赖结果对象，不要反向影响算法逻辑。

理想结构是：

```text
optimization.run() -> Result
viz.plot_result(result)
```

而不是：

```text
optimization -> viz
```

这样可以保证核心算法可以在无图形环境中运行，例如服务器、CI、HPC 环境。

---

## 14. 测试体系缺失

目前没有看到稳定的测试结构。对这种算法库来说，至少需要几类测试：

```text
tests/
  test_problem/
  test_doe/
  test_optimization/
  test_surrogate/
  test_storage/
  test_runtime/
```

### 建议优先补的测试

1. 每个 `Problem` 能否正常 evaluate；
2. 每个算法能否在小问题上跑通；
3. 每个 surrogate 能否 fit/predict；
4. storage 是否能保存并读取结果；
5. seed 是否可复现；
6. optional dependency 缺失时是否优雅失败；
7. 根入口 import 是否正常。

可以先从 smoke test 做起：

```python
def test_import():
    import UQPyL


def test_ga_runs_on_simple_problem():
    ...
```

---

## 15. 文档和 examples 需要明确 public API

当前模块比较多，但外部用户真正应该怎么用，还需要通过文档收敛。

建议文档结构：

```text
docs/
  getting-started.md
  problem.md
  doe.md
  optimization.md
  surrogate.md
  storage.md
  api-reference.md
  migration.md

examples/
  01_define_problem.py
  02_run_ga.py
  03_fit_gpr.py
  04_expensive_optimization.py
  05_save_and_reload_result.py
```

文档中要明确：

- 哪些 API 是 public；
- 哪些是 internal；
- 哪些模块需要额外依赖；
- 结果如何保存和复现；
- 版本升级怎么迁移。

---

# 建议重构路线

## 第一阶段：止血

优先解决会直接影响使用和维护的问题。

1. 清理发布包：
   - 删除 `__pycache__`
   - 删除 `.pyc`
   - 删除不该进入源码包的 `.pyd`
   - 清理构建产物

2. 修复明显 broken import：
   - 检查 `surrogate/ensemble/weightEnsemble.py`
   - 检查所有相对导入
   - 检查 `problem/__init__.py`

3. 改掉根入口的宽泛异常吞掉：

   ```python
   except Exception:
       module = None
   ```

4. 给核心基类补真正的 abstract method：

   ```text
   AlgorithmABC
   InferenceABC
   ProblemBase
   SurrogateABC
   ```

5. 统一基础命名：
   - 新 API 用 snake_case；
   - 旧 API 保留 warning。

---

## 第二阶段：架构收敛

重点抽出公共基础设施。

1. 新增 `core.runtime`：
   - `RunContext`
   - `BaseResult`
   - `StorageBackend`
   - `Reporter`
   - `EventHook`

2. 统一 `Params` 和 `Setting`：
   - 建立 `ParameterSpec`
   - 建立 `ParameterSet`
   - 支持 validate / serialize / optimize spec

3. 重构 `Problem` 输出协议：
   - 分成 `OptimizationProblem / SimulationProblem`
   - 或统一 `EvalResult.channels`

4. 解耦 `surrogate` 和 `optimization`：
   - 使用 `OptimizerProtocol`
   - 使用 dependency injection
   - 使用 registry

5. 重构持久化：
   - sqlite 只存 metadata 和索引；
   - 大数组存 npz/hdf5；
   - 配置存 json；
   - pickle 仅作为 cache。

---

## 第三阶段：工程化

1. 增加 `pyproject.toml`；
2. 使用 `src/` layout；
3. 增加测试目录；
4. 增加 CI；
5. optional dependency 做 extras；
6. native extension 用标准 wheel 构建；
7. 增加 docs 和 examples；
8. 增加 migration guide。

---

# 推荐目标架构

可以考虑演进成下面这种结构：

```text
src/UQPyL/
  __init__.py

  core/
    config.py
    parameter.py
    registry.py
    protocol.py
    runtime/
      context.py
      result.py
      state.py
      storage.py
      reporter.py
      events.py

  problem/
    base.py
    optimization.py
    simulation.py
    eval.py
    benchmark/

  doe/
    base.py
    lhs.py
    sobol.py
    random.py

  optimization/
    base.py
    result.py
    single_objective/
    multi_objective/
    expensive/

  surrogate/
    base.py
    result.py
    gpr.py
    krg.py
    rbf.py
    ensemble/

  analysis/
    base.py
    sensitivity/
    reliability/

  inference/
    base.py
    mcmc/
    likelihood/

  calibration/
    base.py

  viz/
    optimization.py
    surrogate.py
    analysis.py
```

关键依赖方向：

```text
core <- problem <- algorithms/modules <- viz
```

不要出现：

```text
surrogate <-> optimization
analysis <-> optimization
problem -> optimization
```

---

# 最终建议

当前最值得优先做的不是继续增加算法，而是先稳定三件事：

## 1. 稳定 core

包括：

- runtime；
- result；
- storage；
- config；
- protocol；
- registry。

## 2. 稳定 Problem / Eval 协议

明确：

- 什么是优化问题；
- 什么是仿真问题；
- evaluate 返回什么；
- objs / cons / sim 如何表达；
- 多输出如何表达。

## 3. 稳定发布和依赖体系

包括：

- `pyproject.toml`；
- extras；
- wheel 构建；
- 测试；
- CI；
- 文档；
- examples。

如果这三块不先收敛，后面每新增一个算法，都会继续复制一套：

```text
params / state / result / storage / verbose / import handling
```

长期维护成本会越来越高。
