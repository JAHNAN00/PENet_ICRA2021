# PENet 的 NYU 随机权重适配测速

原项目仅提供 KITTI loader；这里是统一输入下的结构效率测试，不是原论文 KITTI 复现或已训练 NYU 模型。

从本仓库根目录执行，使用独立推理环境，不需要训练入口的 OpenCV/绘图库：

```bash
conda env create -f environment.yaml
conda activate PENet
mkdir -p data
ln -s ../../data/nyudepthv2_h5 data/nyudepthv2_h5
python -m pytest tests/test_inference.py -q
python scripts/benchmark_nyu.py --output outputs/benchmark_nyu_new_run
```

已有环境/软链接时不要重复创建；输出目录必须尚不存在。搬到其他目录时只调整软链接；`images/` 原有插图不动。

## 模型与适配

- 完整 `PENet_C2`（非 ENet、非训练用慢速实现），默认 xyz 编码、dilation_rate=2；3/5/7 三种 kernel，dilation 2 和 1 各 6 次传播，参数与固定 CSPN 核不变。
- 外部统一真实 NYU 228×304、500 点、seed=2023、RGB ImageNet 归一化/确定性采样。RGB/深度在纯推理 wrapper 内右补 16、下补 28 个零像素至 256×320（满足 5 次下采样），预测裁回 228×304；**补边和裁剪计入时间、显存及 MACs 输入尺寸**。
- 使用 NYU RGB 标定 fx=582.62448、fy=582.69103、cx=313.04476、cy=238.44390，缩放 0.5 后中心裁剪偏移 left=8/top=6，得到当前 K。保持作者 CoordConv 的 endpoint-normalized 坐标约定、GeometryFeature 的通道/运算规则；K/position 提前放 GPU，不在计时中加载或传输。
- xyz 的硬编码 352×1216 尺寸新增显式 `geometry_size` 选项；旧 KITTI 路径默认值不变。C2 构造时不再硬编码 `.cuda()` 创建固定核，而由整个模型统一搬到 GPU。
- RGB 预处理不同于作者原 KITTI uint8 输入。这些适配有明确局限，不报告 RMSE，不宣称 NYU 预测几何/精度的论文复现。

## 测量与记录

RTX 4060 Ti，FP32、batch=1、TF32/autocast/compile/CUDA graphs 关闭，cuDNN benchmark=False、deterministic=False，CPU threads=1。样本 0、326、653 各预热 100 次、CUDA events 逐帧同步计时 1000 次；合并平均/P95，FPS=1000/平均毫秒。记录同步墙钟时间，显存另测单次前向峰值 allocated。

排除数据加载/H2D、真值、损失、指标、日志、可视化。参数包含冻结核，共享参数去重；FP32 state_dict 文件含 buffers 和序列化开销。MACs 统计卷积（含转置卷积）与 CSPN einsum 累加，不含 unfold、补边、归一化、逐元素操作，不能当完整 FLOPs。

输出保存 `results.json`、逐次延迟/GPU 状态、配置、随机 state_dict、源码哈希/完整快照、Git 差异和环境清单；三个真实样本与完整原前向在同一适配输入上须 allclose(rtol=1e-5, atol=1e-6)。短验证可加 `--warmup 2 --iterations 5 --groups 1`，不能作为正式结果。
