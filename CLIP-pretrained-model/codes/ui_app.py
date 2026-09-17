import json
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from plate_recognition import PlateRecognizer
from text_image_retrieval import TextImageRetriever


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]
IMAGE_DIR = ROOT_DIR / "dataset1" / "images"
PLATE_TEMPLATE_DIR = str(ROOT_DIR / "车牌识别" / "Character_templates")
WEIGHT_PATH = str(BASE_DIR / "ViT-B-32.pt")
CACHE_PATH = str(BASE_DIR / "result" / "image_features_ui.pt")
LLM_CACHE_PATH = str(BASE_DIR / "result" / "deepseek_compression_cache_ui.json")
OLLAMA_CACHE_PATH = str(BASE_DIR / "result" / "ollama_compression_cache_ui.json")
PLATE_UPLOAD_DIR = BASE_DIR / "result" / "ui_plate_uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

STRATEGY_LABELS = {
    "manual_compressed": "人工规则缩句",
    "deepseek_compressed": "DeepSeek 缩句",
    "ollama_compressed": "Ollama 缩句",
    "raw": "原始文本",
}

CAPTION_PATH = ROOT_DIR / "dataset1" / "caption.json"


def load_example_captions(limit=4):
    try:
        data = json.loads(CAPTION_PATH.read_text(encoding="utf-8"))
        captions = [item.get("caption", "").strip() for item in data if isinstance(item, dict)]
        return [c for c in captions if c][:limit]
    except Exception:
        return []


app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

retriever = TextImageRetriever(
    image_dir=str(IMAGE_DIR),
    weight_path=WEIGHT_PATH,
    cache_path=CACHE_PATH,
    device="cpu",
    batch_size=32,
    llm_cache_path=LLM_CACHE_PATH,
    ollama_cache_path=OLLAMA_CACHE_PATH,
)
plate_recognizer = PlateRecognizer(
    template_dir=PLATE_TEMPLATE_DIR,
    weight_path=WEIGHT_PATH,
    device="cpu",
)


def is_allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    retriever.refresh_image_index()
    PLATE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    result = None
    plate_result = None
    search_error = ""
    search_message = ""
    upload_message = ""
    upload_error = ""
    plate_message = ""
    plate_error = ""
    query = ""
    strategy = "manual_compressed"
    active_module = request.values.get("module", "text")

    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "search":
            active_module = "text"
            query = request.form.get("query", "").strip()
            strategy = request.form.get("strategy", "manual_compressed")

            if not query:
                search_error = "请输入检索文本。"
            else:
                result = retriever.search(query, strategy=strategy, top_k=6)
                search_message = "已完成图像检索。"

        elif action == "upload":
            active_module = "text"
            uploaded_file = request.files.get("image_file")
            if not uploaded_file or not uploaded_file.filename:
                upload_error = "请先选择要加入图库的图片。"
            elif not is_allowed_file(uploaded_file.filename):
                upload_error = "仅支持 jpg、jpeg、png、bmp、webp 格式。"
            else:
                suffix = Path(uploaded_file.filename).suffix.lower()
                base_name = secure_filename(Path(uploaded_file.filename).stem) or "uploaded_image"
                saved_name = f"ui_{datetime.now():%Y%m%d_%H%M%S_%f}_{base_name}{suffix}"
                save_path = IMAGE_DIR / saved_name
                uploaded_file.save(save_path)
                retriever.refresh_image_index(force=True)
                upload_message = f"图片已加入图库：{saved_name}"

        elif action == "plate_recognize":
            active_module = "plate"
            uploaded_file = request.files.get("plate_file")
            if not uploaded_file or not uploaded_file.filename:
                plate_error = "请先选择待识别的车牌图片。"
            elif not is_allowed_file(uploaded_file.filename):
                plate_error = "仅支持 jpg、jpeg、png、bmp、webp 格式。"
            else:
                suffix = Path(uploaded_file.filename).suffix.lower()
                base_name = secure_filename(Path(uploaded_file.filename).stem) or "uploaded_plate"
                saved_name = f"plate_{datetime.now():%Y%m%d_%H%M%S_%f}_{base_name}{suffix}"
                save_path = PLATE_UPLOAD_DIR / saved_name
                uploaded_file.save(save_path)
                plate_result = plate_recognizer.recognize(str(save_path))
                plate_result["image_path"] = str(save_path)
                plate_message = "已完成车牌识别。"

    return render_template(
        "index.html",
        active_module=active_module,
        result=result,
        plate_result=plate_result,
        query=query,
        strategy=strategy,
        strategy_label=STRATEGY_LABELS.get(strategy, strategy),
        search_error=search_error,
        search_message=search_message,
        upload_error=upload_error,
        upload_message=upload_message,
        plate_message=plate_message,
        plate_error=plate_error,
        image_count=len(retriever.image_paths),
        examples=load_example_captions(),
        strategies=[
            ("manual_compressed", "人工规则缩句"),
            ("deepseek_compressed", "DeepSeek 缩句"),
            ("ollama_compressed", "Ollama 缩句"),
            ("raw", "原始文本"),
        ],
    )


@app.route("/library/<path:filename>")
def serve_library_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/plate_uploads/<path:filename>")
def serve_plate_upload(filename):
    return send_from_directory(PLATE_UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
