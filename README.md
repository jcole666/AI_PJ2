# 快速开始指南

本仓库包含一个数学求解模型，可使用提供的脚本进行训练和测试。请按照以下步骤配置环境、训练模型，并在指定测试集上进行测试。

## 环境配置

### 准备环境

如果你使用的是 ModelScope（魔搭），则无需额外安装环境。否则，可以运行以下命令安装所需依赖：

```bash
pip install transformers modelscope peft swanlab
```

## 训练

执行以下命令开始训练：

```bash
python qwen_ft.py
```

- 开始训练会自动下载`Qwen2.5-0.5B`模型，请确保你的设备有几个GB的空间
- 训练需要登陆`swanlab`，请根据指引完成注册或登录

## 测试

训练完成后，可以运行以下命令在指定测试集上测试模型：

```bash
python infer.py
```

该命令会生成 `submit.csv` 文件，可直接提交到 [DataFountain](https://www.datafountain.cn/competitions/467/submits) 竞赛平台。
