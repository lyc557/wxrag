import os

from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

# args
pdf_file_name = "docs/600519_20240403_W0YD.pdf"  # 替换为实际PDF路径
name_without_suff = pdf_file_name.split(".")[0]

# prepare env
local_image_dir, local_md_dir = "output/images", "output"
image_dir = str(os.path.basename(local_image_dir))

os.makedirs(local_image_dir, exist_ok=True)

image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
    local_md_dir
)

# read bytes
reader1 = FileBasedDataReader("")
pdf_bytes = reader1.read(pdf_file_name)  # 读取PDF文件内容

# proc
## 创建数据集实例
ds = PymuDocDataset(pdf_bytes)

## 推理判断
if ds.classify() == SupportedPdfParseMethod.OCR:
    infer_result = ds.apply(doc_analyze, ocr=True)
    ## 处理管道（OCR模式）
    pipe_result = infer_result.pipe_ocr_mode(image_writer)
else:
    infer_result = ds.apply(doc_analyze, ocr=False)
    ## 处理管道（文本模式）
    pipe_result = infer_result.pipe_txt_mode(image_writer)

### 绘制模型结果到每页
infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

### 获取模型推理结果
model_inference_result = infer_result.get_infer_res()

### 绘制页面布局结果
pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

### 绘制文本块结果
pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

### 生成Markdown内容
md_content = pipe_result.get_markdown(image_dir)

### 保存Markdown文件
pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

### 获取内容列表
content_list_content = pipe_result.get_content_list(image_dir)

### 保存内容列表
pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

### 获取中间JSON数据
middle_json_content = pipe_result.get_middle_json()

### 保存中间JSON
pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')