# 人工智能与交通大数据实战课程手册 · TensorFlow 版

本手册源自北京交通大学在中国大学 MOOC 开设的《人工智能与交通大数据实战》。感谢张金雷老师及课程团队提供原课程讲解、课件与教学案例。Shiqi Li 完成手册整理和后续开发。

**[在线阅读手册](https://richardlsq.github.io/traffic-ai-handbook-tensorflow/)** · **[下载完整课程 ZIP](https://github.com/Richardlsq/traffic-ai-handbook-tensorflow/releases/latest/download/traffic-ai-handbook-tensorflow.zip)**

[PyTorch 版手册](https://richardlsq.github.io/traffic-ai-handbook/) · [原课程](https://www.icourse163.org/course/NJTU-1471607173) · [张金雷的 GitHub](https://github.com/JinleiZhangBJTU) · [Shiqi Li 的 GitHub](https://github.com/Richardlsq)

打开 [课程手册](人工智能与交通大数据实战-课程代码手册.html) 即可离线学习。正文以张金雷老师的《Python 基础与进阶》《人工智能基础模型简介实战》《人工智能在轨道交通刷卡数据的应用综述及实战》课件为主线。三份独立课件共 245 页，逐页讲义按原页码讲解概念、公式、图表、例题与实验。在讲解旁展开「查看本步代码」，可阅读原 Notebook 单元、脚本片段及带预期输出的教学微例。逐页复查补入了原课件图形和曲线、可阅读的参数及数据表、分步公式与例题复算。在对应步骤下展开程序即可查看计算过程。按章展开「配套实验」可运行完整流程。两版共用教学正文，实验代码对应各自的框架。

## 配套实验环境

1. 完整解压下载包，保留目录结构。在解压后的课程目录打开 PowerShell。
2. 准备 Windows x64 的 Python 3.12，用 `py -3.12 -V` 确认解释器可用。
3. 执行下面的命令。它会在课程目录创建 `.venv`，联网安装固定版本的 CPU 依赖（含 TensorFlow 2.21.0），注册 Jupyter 内核，并运行七个课程单元的快速验收。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\课程初始化.ps1 -SmokeTest
```

检查通过时输出 `PASS`。验收产物按运行时间保存在 `课程输出/快速验收/`。只安装环境时可以省略 `-SmokeTest`。已有 Python 3.12 安装时，可用 `-BasePython` 显式传入解释器路径。

## 运行教学示例

整包里的 `解压代码/` 是整理并验证过的教学代码，包含 LSTM 测试与 MSE 计算、深圳路速与地铁客流模型验证流程的修正。

各章的配套实验提供可复制的示例、命令和检查结果。课程教学内容可直接在网页阅读，实验所用的 Notebook、训练代码和数据在完整 ZIP 内。

在 PowerShell 中从对应 Notebook 的目录启动 JupyterLab，内核选择 **Python (Traffic AI · TensorFlow)**。手册的「环境与运行入口」列出全部 Notebook 路径；完整训练命令及预期结果分别列在各章。已有环境可以执行：

```powershell
python -B -X utf8 '课程工具/快速验收.py' env
python -B -X utf8 '课程工具/快速验收.py' all
```

- `解压代码/` 含 15 份代码 Notebook、两个 TensorFlow 图模型项目，以及实验所需的 CSV、Excel 和 MNIST 数据。地铁刷卡预处理使用合成事件。
- `课程工具/` 含依赖清单、快速验收和地铁误差分析。路速和地铁的快速验收分别从头训练一轮，权重与预测由程序在本地生成。预测用本版训练生成的 `best.weights.h5`，其中包含最佳轮次、切分、缩放参数和数据指纹。旧版 `.pth` 权重对应原版代码。
- `课程输出/` 由运行命令创建。连续两次运行会分别保存结果。

## 数据来源

配套 CSV、Excel 与 MNIST 数据已放在运行所需的位置。课程课件按章引用题名及页码。

| 数据集或样例 | 包内文件 | 来源 | GitHub 与文献 |
| --- | --- | --- | --- |
| 北京地铁 MetroBJ2016 | `in_15min.csv`、`out_15min.csv`、`adjacency.csv` | 张金雷等发布的北京地铁站点客流与邻接数据。包内预测输入与作者仓库同名文件逐字节一致。 | [作者仓库](https://github.com/JinleiZhangBJTU/ResNet-LSTM-GCN/tree/78b257bb362d1d0b48bce6fda6250ed67b2e8381) · [论文](https://doi.org/10.1109/TITS.2020.3000761) |
| 深圳出租车路速 SZ-taxi | `sz_speed-论文数据.csv`、`sz_adj1.csv` | 来源为 Ling Zhao 等发布的 T-GCN 数据。邻接矩阵逐字节一致，包内速度表去掉了上游的路段编号表头，数值在舍入精度内一致（最大绝对差小于 10⁻⁷）。 | [作者数据仓库](https://github.com/lehaifeng/T-GCN/tree/e256572736291433f8fe5c5d432b0100184805d5/data) · [论文](https://arxiv.org/abs/1811.05320) |
| MNIST 手写数字 | `4 个 IDX 文件` | 由 Yann LeCun、Corinna Cortes、Christopher J. C. Burges 整理。包内文件与 CVDF 镜像解压后的四份 IDX 数据逐字节一致。 | [原始说明](https://yann.lecun.com/exdb/mnist/) · [CVDF GitHub 镜像](https://github.com/cvdfoundation/mnist) |
| 课程配套客流表 | `15min_in.csv（CNN）`、`in_15min.csv（Python）`、`xizhimen.csv`、`LSTM.csv` | 来自课程代码附件。CNN 与 Python 的两个多站表分别有 275、276 条站点序列与 MetroBJ2016 进站矩阵一致，另含机场线站点。西直门序列与作者矩阵相应行一致。LSTM 教学序列由课程提供。 | [MetroBJ2016 作者仓库](https://github.com/JinleiZhangBJTU/ResNet-LSTM-GCN/tree/78b257bb362d1d0b48bce6fda6250ed67b2e8381) · [CNN 客流表](https://github.com/Richardlsq/traffic-ai-handbook-tensorflow/blob/tensorflow-v1.0.0/%E8%A7%A3%E5%8E%8B%E4%BB%A3%E7%A0%81/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%9F%BA%E7%A1%803--%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%BB%A3%E7%A0%81/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%9F%BA%E7%A1%803.3%2B%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/CNN/15min_in.csv) · [Python 客流样表](https://github.com/Richardlsq/traffic-ai-handbook-tensorflow/tree/tensorflow-v1.0.0/%E8%A7%A3%E5%8E%8B%E4%BB%A3%E7%A0%81/Python%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86%E4%BB%A3%E7%A0%81/Python%E4%BB%A3%E7%A0%81/python1.8-1.9) · [LSTM 教学序列](https://github.com/Richardlsq/traffic-ai-handbook-tensorflow/blob/tensorflow-v1.0.0/%E8%A7%A3%E5%8E%8B%E4%BB%A3%E7%A0%81/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%9F%BA%E7%A1%802--%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%BB%A3%E7%A0%81/LSTM.csv) |
| Python 教学文件与分时段客运量报表 | `test.csv`、`test.txt`、`table.xlsx`、`test1.CSV`、`test2.CSV`、`2017-09-26 和 2017-09-27 的两个 Excel 报表` | 取自课程 Python 基础知识代码附件，用于文件读写、索引和跨日表格整理。 | [文件读写样例](https://github.com/Richardlsq/traffic-ai-handbook-tensorflow/tree/tensorflow-v1.0.0/%E8%A7%A3%E5%8E%8B%E4%BB%A3%E7%A0%81/Python%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86%E4%BB%A3%E7%A0%81/Python%E4%BB%A3%E7%A0%81/Python1.3-1.4) · [Pandas 样表与报表](https://github.com/Richardlsq/traffic-ai-handbook-tensorflow/tree/tensorflow-v1.0.0/%E8%A7%A3%E5%8E%8B%E4%BB%A3%E7%A0%81/Python%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86%E4%BB%A3%E7%A0%81/Python%E4%BB%A3%E7%A0%81/Python1.6-Pandas) |
| 合成刷卡事件 | `样例_0101.csv`、`样例_0102.csv` | 随手册生成的两日合成事件，用于演示刷卡记录分箱及 OD 聚合。 | [两天样例](https://github.com/Richardlsq/traffic-ai-handbook-tensorflow/tree/tensorflow-v1.0.0/%E8%A7%A3%E5%8E%8B%E4%BB%A3%E7%A0%81/%E6%A1%88%E4%BE%8B%E5%AE%9E%E6%88%981--%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E4%B8%8E%E8%BD%A8%E9%81%93%E4%BA%A4%E9%80%9A%E4%BB%A3%E7%A0%81/subway%20flow%20prediction_for%20book/data_preprocess/data) |
| 代码生成的练习数据 | `双月形样本`、`正弦序列`、`合成图` | 反向传播、时序窗口和节点分类的练习数据由配套代码在运行时生成。 | [双月形样本代码](https://github.com/Richardlsq/traffic-ai-handbook-tensorflow/blob/tensorflow-v1.0.0/%E8%A7%A3%E5%8E%8B%E4%BB%A3%E7%A0%81/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%9F%BA%E7%A1%801--%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95%E4%BB%A3%E7%A0%81/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%9F%BA%E7%A1%803.1%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E7%AE%97%E6%B3%95.ipynb) · [正弦序列代码](https://github.com/Richardlsq/traffic-ai-handbook-tensorflow/blob/tensorflow-v1.0.0/%E8%A7%A3%E5%8E%8B%E4%BB%A3%E7%A0%81/TensorFlow%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86%E4%BB%A3%E7%A0%81/TensorFlow%E4%BB%A3%E7%A0%81/TensorFlow2.2%E6%95%B0%E6%8D%AE%E6%A8%A1%E5%9D%97.ipynb) · [合成图代码](https://github.com/Richardlsq/traffic-ai-handbook-tensorflow/blob/tensorflow-v1.0.0/%E8%A7%A3%E5%8E%8B%E4%BB%A3%E7%A0%81/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%9F%BA%E7%A1%804--%E5%9B%BE%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%BB%A3%E7%A0%81/3.4.3/example.py) |

北京地铁案例使用 2016 年 2 月 29 日至 4 月 1 日的 25 个工作日。每天 05:00–23:00 按 15 分钟聚合，进站和出站矩阵各为 276 站 × 1,800 时间片，邻接矩阵为 276 × 276。课件页码见手册中的「北京地铁客流数据（MetroBJ2016）」。
