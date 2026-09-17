import copy
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}

ET.register_namespace("w", W_NS)

BASE_DIR = Path(__file__).resolve().parent


def w_tag(tag):
    return f"{{{W_NS}}}{tag}"


def make_paragraph(text):
    p = ET.Element(w_tag("p"))
    p_pr = ET.SubElement(p, w_tag("pPr"))
    ET.SubElement(p_pr, w_tag("widowControl"), {w_tag("val"): "0"})
    r = ET.SubElement(p, w_tag("r"))
    t = ET.SubElement(r, w_tag("t"))
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    return p


def set_cell_text(cell, paragraphs):
    for child in list(cell):
        if child.tag == w_tag("p"):
            cell.remove(child)
    for text in paragraphs:
        cell.append(make_paragraph(text))


def main():
    template_path = BASE_DIR / "实验报告模板_2026.docx"
    output_path = BASE_DIR / "实验报告_2026_填写版.docx"

    replacements = {
        (0, 1): ["（请填写）"],
        (0, 3): ["（请填写）"],
        (0, 5): ["（请填写）"],
        (1, 1): ["（如无可写“无”）"],
        (1, 3): ["（请填写）"],
        (1, 5): ["（请填写）"],
        (2, 1): ["实验一：基于CLIP系列的文本图像匹配及其应用"],
        (3, 1): [
            "本次实验围绕 CLIP 系列模型的文本-图像匹配任务展开，目标是掌握多模态特征表示、长文本压缩、图像检索与结果评估的基本流程。",
            "实验要求能够从 caption.json 中读取文本描述，对过长且包含冗余信息的描述进行关键词压缩，再使用 CLIP 模型分别提取文本特征和图像特征，通过余弦相似度完成图像库检索，并将最终结果按照指定 JSON 格式输出。",
            "本次实验重点掌握的知识点包括：CLIP 的基本原理与使用方式、文本与图像特征的归一化和相似度计算、长文本截断问题与压缩策略、批量图像特征缓存、实验结果评估方法，以及如何在不训练或微调模型的前提下完成指定数据集上的图文匹配任务。",
        ],
        (4, 1): [
            "本次实验在 Linux 环境下完成，Python 运行环境为 myenv-3.9。",
            "主要依赖包括 torch、torchvision、PIL、ftfy、regex 和实验提供的 clip 代码；预训练模型参数文件为 ViT-B-32.pt。",
            "实验数据主要包括 dataset1/caption.json、dataset1/caption_with_keywords_and_image.json、dataset1/images/，开发与调试通过命令行脚本完成，并使用 evaluate_result.py 对结果进行评估。",
        ],
        (5, 1): [
            "本次实验完成了实验指导中要求的主任务，并补充实现了附加部分中的车牌识别基线方案。",
            "主任务部分实现了从 caption.json 自动读取描述文本、对长文本执行规则化关键词压缩、使用 CLIP 编码文本与图像特征、计算相似度并输出符合要求的结果 JSON 文件，随后调用评估脚本得到匹配准确率。",
            "在实现过程中修复了 clip 词表文件路径不匹配和 evaluate_result.py 变量名错误两个关键问题。修复后，主脚本 text_image_retrieval.py 能够稳定完成全部 60 条文本描述的图像检索任务，达到实验基本要求。",
            "附加部分中，实现了基于固定字符区域切分和 CLIP 模板匹配的车牌识别脚本 plate_recognition.py，能够读取车牌图片并输出识别结果。",
        ],
        (6, 1): [
            "算法设计思想：由于 CLIP 对文本长度存在限制，直接输入长文本容易导致后半段关键信息被截断，因此先对原始描述进行视觉关键词压缩，再进行图像检索。",
            "实验步骤如下：",
            "1. 从 caption.json 中逐条读取英文描述文本；",
            "2. 对每条文本执行规则化压缩，保留主体对象、动作、场景和关键属性；",
            "3. 加载 ViT-B-32.pt 预训练权重和实验提供的 clip 模块；",
            "4. 对 images 图像库中的全部图片提取特征并归一化，同时缓存图像特征；",
            "5. 对压缩后的文本提取文本特征并归一化；",
            "6. 计算文本特征与全部图像特征之间的余弦相似度；",
            "7. 选择相似度最大的图片作为匹配结果，并保存为指定 JSON 文件；",
            "8. 调用 evaluate_result.py 评估准确率；",
            "9. 在扩展任务中，对车牌图像按固定区域切分字符，并与模板进行 CLIP 特征匹配，输出识别车牌号。",
            "伪代码可概括为：读取文本与图片 -> 文本压缩 -> 图像特征提取 -> 文本特征提取 -> 相似度计算 -> 输出最优图像。",
        ],
        (7, 1): [
            "调试过程中首先遇到 clip 无法正常导入的问题。原因是 simple_tokenizer.py 默认查找 bpe_simple_vocab_16e6.txt.gz，而实验目录实际提供的是 .txt 文件且层级不同，因此增加了多候选路径与两种文件格式兼容逻辑。",
            "第二个问题是 evaluate_result.py 中循环变量与判断变量不一致，导致评估统计错误。修复该问题后，评估结果恢复正常。",
            "第三个问题是长文本直接参与匹配时效果不理想。对原始长句、压缩关键词、多变体平均特征、多变体相似度最大值等策略进行测试后，发现压缩关键词策略效果最好，因此最终固定为 compressed。",
            "最终实验结果如下：文本图像匹配结果文件为 CLIP-pretrained-model/codes/result/caption_result.json，dataset1 上的匹配准确率为 70.0%；车牌识别扩展对全量 1010 张车牌图片测试后，整牌精确匹配率为 77.72%。",
            "结果分析：文本图像匹配效果提升的主要原因在于长文本压缩后更接近视觉内容，减少了无关修辞的干扰；当前仍存在部分抽象描述和相似场景图片的混淆。车牌识别中主要误差来自易混淆字符，例如 0/O、D/0、B/8、T/7 等。",
        ],
        (8, 1): [
            "本次实验中使用 AI 工具辅助完成了目录摸排、PDF 与 docx 模板结构提取、代码问题定位、实验流程整理和报告文字撰写。",
            "在代码实现阶段，AI 工具主要用于分析现有脚本、定位 clip 词表路径错误和评估脚本变量错误，并辅助比较不同的长文本处理策略，最终确定采用“规则压缩 + CLIP 检索”的实现方式。",
            "在实验收尾阶段，AI 还帮助统一整理了测试流程、实验结果和报告内容。总体来看，AI 在本次实验中主要起到提高分析与整理效率的作用，关键算法思路和结果判断仍然依据实验要求和实际运行结果完成。",
        ],
        (9, 1): [
            "若为单人完成，可填写：队员1负责全部实验工作，包括实验要求分析、代码补全、主任务脚本实现、评估脚本修复、测试脚本编写、车牌识别扩展实现、实验结果分析与实验报告撰写；队员2无。",
            "若为双人小组，可按实际情况修改为：队员1负责主任务实现，包括文本压缩策略设计、text_image_retrieval.py 编写、结果评估与调试；队员2负责扩展任务与文档整理，包括 plate_recognition.py 实现、一键测试脚本编写、实验结果汇总与实验报告整理。",
        ],
        (10, 1): [
            "代码 README 与文件说明如下：",
            "1. CLIP-pretrained-model/codes/clip/simple_tokenizer.py：修复词表加载路径与格式兼容问题；",
            "2. CLIP-pretrained-model/codes/clip/clip.py：调整设备选择逻辑，避免导入阶段触发 CUDA 探测；",
            "3. CLIP-pretrained-model/codes/evaluate_result.py：评估文本图像匹配结果并输出 accuracy.txt；",
            "4. CLIP-pretrained-model/codes/text_image_retrieval.py：文本图像匹配主程序；",
            "5. CLIP-pretrained-model/codes/plate_recognition.py：车牌识别扩展程序；",
            "6. run_task1_tests.sh：一键测试脚本；",
            "7. 实验总结.md：实验过程、测试流程和实验结果总结。",
            "运行方式：~/AI-task1/run_task1_tests.sh",
            "若需显式指定 Python 环境，可执行：PYTHON_BIN=python ~/AI-task1/run_task1_tests.sh",
        ],
    }

    with zipfile.ZipFile(template_path, "r") as zin:
        xml_bytes = zin.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        rows = root.findall(".//w:tbl/w:tr", NS)

        for (row_idx, cell_idx), paragraphs in replacements.items():
            cells = rows[row_idx].findall("w:tc", NS)
            set_cell_text(cells[cell_idx], paragraphs)

        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        with zipfile.ZipFile(output_path, "w") as zout:
            for item in zin.infolist():
                data = new_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)

    print(f"Saved filled report to {output_path}")


if __name__ == "__main__":
    main()
