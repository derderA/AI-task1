# AI Task1 README

## 1. 项目简介

本项目实现了“基于CLIP系列的文本图像匹配及其应用”，包括两个核心模块：

1. 文本图像匹配
   - 使用 CLIP 完成 caption 到图片的检索；
   - 支持 `raw`、`manual_compressed`、`deepseek_compressed`、`ollama_compressed` 四种文本策略。
2. 车牌识别
   - 对固定尺寸车牌图片做字符切分；
   - 融合 CLIP 图像特征与字形特征进行模板匹配，输出车牌号码。

项目同时提供 Flask 前端界面，可在一个页面中切换使用这两个模块。

## 2. 目录结构

```text
~/AI-task1
├── CLIP-pretrained-model/
│   └── codes/
│       ├── clip/
│       ├── result/
│       ├── static/
│       ├── templates/
│       ├── deepseek_config.py
│       ├── deepseek_config.py.example
│       ├── evaluate_result.py
│       ├── plate_recognition.py
│       ├── text_image_retrieval.py
│       ├── ui_app.py
│       └── ViT-B-32.pt
├── dataset1/
│   ├── caption.json
│   ├── caption_with_keywords_and_image.json
│   └── images/
├── 车牌识别/
│   ├── Character_templates/
│   └── License_plate/
├── fill_report_template.py
├── run_retrieval_ui.sh
├── run_task1_tests.sh
├── README.md
└── 实验总结.md
```

## 3. 运行环境

- 操作系统：Linux
- Python 环境：`myenv-3.9`
- 推荐做法：先激活对应环境，再直接使用 `python`

主要依赖包括：

- `torch`
- `Pillow`
- `flask`
- `requests`
- `numpy`

如果使用：

- `deepseek_compressed`，需要配置 DeepSeek API Key；
- `ollama_compressed`，需要本地启动 Ollama 服务，默认地址为 `http://localhost:11434`。

## 4. 源程序清单

### 4.1 顶层脚本与文档

| 文件 | 内容简介 |
| --- | --- |
| `run_task1_tests.sh` | 一键执行文本检索、准确率评估和车牌识别测试。 |
| `run_retrieval_ui.sh` | 启动 Flask Web 前端。 |
| `fill_report_template.py` | 用于将实验内容填入实验报告模板。 |
| `实验总结.md` | 当前实验流程、结果和分析总结。 |
| `实验指导.pdf` | 原始实验要求文档。 |
| `实验报告模板_2026.docx` | 课程实验报告模板。 |

### 4.2 文本图像匹配相关文件

| 文件 | 内容简介 |
| --- | --- |
| `CLIP-pretrained-model/codes/text_image_retrieval.py` | 文本图像匹配主程序；负责 caption 读取、压缩、CLIP 编码、图像检索、结果保存。 |
| `CLIP-pretrained-model/codes/evaluate_result.py` | 对检索结果 JSON 进行准确率评估。 |
| `CLIP-pretrained-model/codes/deepseek_config.py` | DeepSeek API Key 本地配置文件。 |
| `CLIP-pretrained-model/codes/deepseek_config.py.example` | DeepSeek 配置示例文件。 |
| `dataset1/caption.json` | 文本检索输入 caption 数据。 |
| `dataset1/caption_with_keywords_and_image.json` | 文本检索评估用标注文件。 |
| `dataset1/images/` | 文本图像检索使用的图库。 |

### 4.3 车牌识别相关文件

| 文件 | 内容简介 |
| --- | --- |
| `CLIP-pretrained-model/codes/plate_recognition.py` | 车牌识别主程序；负责字符切分、白色前景紧致裁剪、CLIP+字形特征融合匹配、结果评估。 |
| `车牌识别/Character_templates/` | 车牌字符模板库，包括省份简称模板和数字字母模板。 |
| `车牌识别/License_plate/` | 待识别车牌图片数据集。 |

### 4.4 Web 前端相关文件

| 文件 | 内容简介 |
| --- | --- |
| `CLIP-pretrained-model/codes/ui_app.py` | Flask 应用入口；整合文本图像检索与车牌识别。 |
| `CLIP-pretrained-model/codes/templates/index.html` | 前端页面模板。 |
| `CLIP-pretrained-model/codes/static/style.css` | 前端样式文件。 |

### 4.5 CLIP 模型与底层代码

| 文件 | 内容简介 |
| --- | --- |
| `CLIP-pretrained-model/codes/ViT-B-32.pt` | CLIP 模型权重文件。 |
| `CLIP-pretrained-model/codes/clip/clip.py` | CLIP 模型加载和预处理逻辑。 |
| `CLIP-pretrained-model/codes/clip/model.py` | CLIP 模型结构定义。 |
| `CLIP-pretrained-model/codes/clip/simple_tokenizer.py` | CLIP 文本分词器实现。 |
| `CLIP-pretrained-model/codes/clip/__init__.py` | `clip` 包初始化文件。 |
| `bpe_simple_vocab_16e6.txt/bpe_simple_vocab_16e6.txt` | CLIP 分词所需 BPE 词表。 |

### 4.6 结果与缓存目录

| 路径 | 内容简介 |
| --- | --- |
| `CLIP-pretrained-model/codes/result/` | 保存检索结果、压缩结果、中间缓存和车牌识别结果。 |
| `CLIP-pretrained-model/codes/accuracy.txt` | 最近一次文本检索准确率输出。 |

## 5. 如何运行

### 5.1 一键运行测试

```bash
~/AI-task1/run_task1_tests.sh
```

如果想显式指定 Python：

```bash
PYTHON_BIN=python ~/AI-task1/run_task1_tests.sh
```

该脚本默认执行：

1. `manual_compressed` 文本检索；
2. 检索准确率评估；
3. 车牌识别批量测试。

### 5.2 运行文本图像匹配

#### 1）人工规则缩句

```bash
python \
  ~/AI-task1/CLIP-pretrained-model/codes/text_image_retrieval.py \
  --device cpu \
  --strategy manual_compressed \
  --output_path ~/AI-task1/CLIP-pretrained-model/codes/result/caption_result_manual.json \
  --compression_output_path ~/AI-task1/CLIP-pretrained-model/codes/result/manual_compressions.json
```

#### 2）DeepSeek 缩句

先在 `CLIP-pretrained-model/codes/deepseek_config.py` 中填写 API Key，或通过环境变量/命令行参数传入。

```bash
python \
  ~/AI-task1/CLIP-pretrained-model/codes/text_image_retrieval.py \
  --device cpu \
  --strategy deepseek_compressed \
  --output_path ~/AI-task1/CLIP-pretrained-model/codes/result/caption_result_deepseek_v5.json \
  --compression_output_path ~/AI-task1/CLIP-pretrained-model/codes/result/deepseek_compressions_v5.json
```

#### 3）Ollama 缩句

请先确保本地 Ollama 服务可用。

```bash
python \
  ~/AI-task1/CLIP-pretrained-model/codes/text_image_retrieval.py \
  --device cpu \
  --strategy ollama_compressed \
  --output_path ~/AI-task1/CLIP-pretrained-model/codes/result/caption_result_ollama_v5.json \
  --compression_output_path ~/AI-task1/CLIP-pretrained-model/codes/result/ollama_compressions_v5.json
```

#### 4）原始文本对照组

```bash
python \
  ~/AI-task1/CLIP-pretrained-model/codes/text_image_retrieval.py \
  --device cpu \
  --strategy raw \
  --output_path ~/AI-task1/CLIP-pretrained-model/codes/result/caption_result_raw.json \
  --compression_output_path ~/AI-task1/CLIP-pretrained-model/codes/result/raw_texts.json
```

### 5.3 评估文本检索结果

```bash
cd ~/AI-task1/CLIP-pretrained-model/codes
python evaluate_result.py \
  --result_path ~/AI-task1/CLIP-pretrained-model/codes/result/caption_result_manual.json \
  --ground_truth_path ~/AI-task1/dataset1/caption_with_keywords_and_image.json
```

### 5.4 运行车牌识别

```bash
python \
  ~/AI-task1/CLIP-pretrained-model/codes/plate_recognition.py \
  --device cpu \
  --shape_weight 0.2 \
  --plate_dir ~/AI-task1/车牌识别/License_plate \
  --template_dir ~/AI-task1/车牌识别/Character_templates \
  --weight_path ~/AI-task1/CLIP-pretrained-model/codes/ViT-B-32.pt \
  --output_path ~/AI-task1/CLIP-pretrained-model/codes/result/plate_result.json \
  --evaluate_by_filename
```

### 5.5 启动 Web 前端

```bash
~/AI-task1/run_retrieval_ui.sh
```

浏览器访问：

```text
http://127.0.0.1:7860
```

页面中可以：

1. 选择文本压缩策略进行图像检索；
2. 上传图片动态加入图库；
3. 上传车牌图片进行识别。

## 6. 当前实验结果

### 6.1 文本图像匹配

| 策略 | 正确数 | 准确率 |
| --- | --- | --- |
| `raw` | 38/60 | 63.33% |
| `manual_compressed` | 44/60 | 73.33% |
| `deepseek_compressed` | 54/60 | 90.00% |
| `ollama_compressed` | 44/60 | 73.33% |

当前最优结果是 `deepseek_compressed`，对应文件：

- `CLIP-pretrained-model/codes/result/caption_result_deepseek_v5.json`
- `CLIP-pretrained-model/codes/result/deepseek_compressions_v5.json`

### 6.2 车牌识别

- 优化前整牌精确匹配率：`785/1010 = 77.72%`
- 优化后整牌精确匹配率：`1010/1010 = 100.00%`
- 结果文件：`CLIP-pretrained-model/codes/result/plate_result.json`

本轮车牌识别的改进点主要有 3 个：

1. 在固定框裁剪之后，继续基于白色字符区域做紧致裁剪，减少蓝色背景对模板匹配的干扰。
2. 在原有 CLIP 图像特征之外，新增字符二值字形特征，并与 CLIP 相似度加权融合，提升对易混淆字符的区分能力。
3. 针对最后残留的 `1/L` 混淆，增加基于字形宽高比的定向判别规则。

优化后的误差表现也更清晰：

- 省份简称位置识别率提升到 `100%`
- 第 3 到第 7 位字符识别率提升到 `100%`
- 原本最集中的 `1/L`、`7/2`、`F/E` 混淆已消除

对应测试命令如下：

```bash
python \
  ~/AI-task1/CLIP-pretrained-model/codes/plate_recognition.py \
  --device cpu \
  --shape_weight 0.2 \
  --evaluate_by_filename
```

## 7. 说明

1. `result/` 目录下保留了多轮实验的历史 JSON 文件，方便对比不同 Prompt 和策略。
2. 当前代码中已经移除了 `Takahe` 方案，现行文本策略仅保留 `raw`、`manual_compressed`、`deepseek_compressed`、`ollama_compressed`。
3. 若修改了 DeepSeek 或 Ollama 的 Prompt，建议同步删除对应缓存文件后再重新测试。
