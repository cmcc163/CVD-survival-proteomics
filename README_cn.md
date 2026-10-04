# 基于血浆蛋白组的心血管生存预测

本项目包含UK Biobank中 Total CVD、ASCVD 和 HF 蛋白组生存预测的代码与训练模型。

English documentation: [README.md](README.md)

## 分析流程

1. 使用重复 LassoNet 筛选稳定蛋白。
2. 使用 Optuna 调参并训练五折集成模型。
3. 生成集成预测和折外预测。
4. 评估区分度、校准度和临床效用。
5. 完成模型解释和精简面板分析。

模型包括 Cox、XGBoost、MLP、TabNet、NODE、FT-Transformer、SAINT 和 TabPFN。

## 环境安装

```bash
conda env create -f environment.yml
conda activate cvd-survival
python examples/run_synthetic_workflow.py
```

发布环境使用Python 3.10.1，R依赖见 `requirements-r.txt`。

## 输入和配置

复制 `config/paths.example.yml` 为 `config/paths.local.yml`，填写本机输入和输出路径。

每个结局文件需要包含 `eid`、`Ethnic`、`Is_Incident`、生存时间和模型变量。临床变量为：

`age`、`sex`、`ever_smoked`、`Diabetes_baseline`、`Cholesterol_treatment`、`hdl_cholesterol`、`non_hdl_cholesterol`、`hypertension_treatment`、`average_SBP`、`eGFR` 和 `BMI`。

缺少 `non_hdl_cholesterol` 时，由总胆固醇和HDL胆固醇计算。蛋白面板文件使用 `Protein_Name` 列。

## 运行

```bash
# 查看全部分析步骤
python run.py --help

# 训练一个模型组合
python run.py train --config config/paths.local.yml \
  --disease-type ASCVD --data-type all --model-name SAINT \
  --protein-path features/lassonet/ASCVD/lassonet_protein.csv

# 生成五折集成预测
python run.py predict --config config/paths.local.yml \
  --disease-type ASCVD --data-type all --model-name SAINT \
  --protein-path features/lassonet/ASCVD/lassonet_protein.csv
```

论文下游分析位于 `analysis/`。MAPLE/SuSiE核心代码后续补入 `genetics/maple_susie/`。

## 发布文件

- `features/lassonet/`：蛋白筛选频率和最终面板。
- `artifacts/models/`：主分析五折模型。
- `artifacts/optuna/`：对应的Optuna数据库。
- `artifacts/rp_saint/`：RP-SAINT权重、预处理器、Breslow对象和核对信息。

当前包含360个主分析折模型、72个Optuna数据库和15个RP-SAINT折模型。RP-SAINT预测与历史结果在浮点精度范围内一致。模型文件使用Git LFS管理。

## 检查

```bash
python -m pytest -q
python -m tools.validate_release_runtime --artifact-dir artifacts
```

引用信息见 `CITATION.cff`，代码使用MIT许可证。
