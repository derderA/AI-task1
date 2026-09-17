import argparse
import json
import os
import re
from pathlib import Path

import requests
import torch
from PIL import Image

import clip


CODES_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODES_DIR.parents[1]
RESULT_DIR = CODES_DIR / "result"
DATASET_DIR = ROOT_DIR / "dataset1"


STOPWORDS = {
    "a", "an", "and", "are", "around", "as", "at", "be", "been", "being", "behind",
    "below", "beside", "between", "but", "by", "for", "from", "has", "have", "her",
    "hers", "his", "in", "into", "is", "it", "its", "itself", "of", "off", "on",
    "onto", "or", "our", "ours", "over", "she", "so", "than", "that", "the", "their",
    "theirs", "them", "themselves", "there", "these", "they", "this", "those", "through",
    "to", "toward", "under", "up", "very", "was", "were", "while", "with", "within",
}

FILLER_WORDS = {
    "adventure", "air", "amidst", "atmosphere", "backdrop", "beauty", "beacon",
    "beyond", "bond", "bustling", "calm", "captivating", "celebration", "chaos",
    "charm", "comfort", "companion", "connection", "creative", "day", "dedication",
    "delight", "determination", "dream", "echoing", "elegance", "emotion", "energy",
    "enthusiasm", "evident", "expression", "feast", "freedom", "fulfillment", "future",
    "glow", "grace", "happiness", "heart", "journey", "joy", "landscape", "life",
    "magic", "memory", "moment", "nature", "passion", "peace", "playful", "presence",
    "purpose", "resilience", "scene", "serene", "shared", "simple", "solace", "spirit",
    "sunlight", "surroundings", "symbol", "tableau", "testament", "thrill", "timeless",
    "tranquil", "vibrant", "warmth", "wilderness", "wonder", "world",
}

ABSTRACT_WORDS = {
    "adrenaline", "artistry", "bygone", "cadence", "captivating", "camaraderie", "chaos",
    "comfort", "contemplative", "creation", "crowd", "culture", "daring", "delights",
    "dialogue", "ebb", "era", "expertise", "faithful", "familiar", "focused", "fulfillment",
    "gentle", "grace", "grasp", "historic", "hum", "humoring", "humble", "humility",
    "intrigue", "journey", "lulling", "machinery", "mesmerizing", "moment", "motion",
    "nature", "nostalgic", "onlookers", "palpable", "past", "peaceful", "poise", "promise",
    "purpose", "reassuring", "regally", "respite", "rhythm", "rustle", "satisfaction",
    "serene", "shared", "significance", "silent", "simplicity", "slumber", "solace",
    "spectacle", "sway", "symphony", "teeming", "timeless", "tradition", "transient",
    "uncertainty", "urban", "vantage", "vital", "weary", "witness",
}

MEANINGFUL_ING_WORDS = {
    "boarding", "cooking", "dancing", "driving", "flying", "grilling", "holding", "jumping",
    "looking", "performing", "reviewing", "riding", "sitting", "skating", "skiing", "smiling",
    "standing", "trailing", "walking", "wearing", "working", "wrapping",
}

DEEPSEEK_COMPRESSION_PROMPT_EN = (
    "You rewrite image captions into one short natural visual phrase for CLIP-based image retrieval.\n"
    "Rules:\n"
    "- Use English only.\n"
    "- Use 8 to 16 words whenever possible, and never exceed 20 words.\n"
    "- Keep only visible subject, key attribute, visible action, and important scene detail.\n"
    "- Remove emotion, intention, symbolism, background story, and other abstract meaning.\n"
    "- Do not output comma-separated keyword lists.\n"
    "- Return only one short phrase or sentence.\n"
    "Examples:\n"
    "Input: A woman wearing glasses smiles with satisfaction as she reviews the images on the back of her camera.\n"
    "Output: woman with glasses reviewing photos on camera\n"
    "Input: A red truck is parked beside a building on a snowy street, creating a quiet winter scene.\n"
    "Output: red truck parked beside building on snowy street\n"
    "Input: A dog runs joyfully through the snow, enjoying the freedom of the open field.\n"
    "Output: dog running across snowy field\n"
)

OLLAMA_COMPRESSION_PROMPT_EN = (
    "Rewrite the input as one short natural visual phrase in English for CLIP image retrieval.\n"
    "Rules:\n"
    "- Use 8 to 14 words whenever possible, and never exceed 20 words.\n"
    "- Keep the visible subject, key attribute, visible action, and useful scene detail.\n"
    "- Remove emotion, intention, symbolism, and abstract meaning.\n"
    "- Do not output comma-separated keywords.\n"
    "- Do not explain your answer.\n"
    "- Return only one short phrase or sentence.\n"
    "Examples:\n"
    "Input: A man proudly stands beside his bicycle in a busy street, ready for the day's adventure.\n"
    "Output: man standing beside bicycle on busy street\n"
    "Input: A woman wearing glasses smiles with satisfaction as she reviews the images on the back of her camera.\n"
    "Output: woman with glasses reviewing photos on camera\n"
    "Input: A dog runs joyfully through the snow, enjoying the freedom of the open field.\n"
    "Output: dog running across snowy field\n"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Use CLIP to retrieve the best-matching image for each caption.")
    parser.add_argument("--caption_path", type=str, default=str(DATASET_DIR / "caption.json"))
    parser.add_argument("--image_dir", type=str, default=str(DATASET_DIR / "images"))
    parser.add_argument("--weight_path", type=str, default=str(CODES_DIR / "ViT-B-32.pt"))
    parser.add_argument("--output_path", type=str, default=str(RESULT_DIR / "caption_result.json"))
    parser.add_argument("--cache_path", type=str, default=str(RESULT_DIR / "image_features.pt"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument(
        "--strategy",
        type=str,
        default="manual_compressed",
        choices=["manual_compressed", "deepseek_compressed", "ollama_compressed", "raw"],
    )
    parser.add_argument("--deepseek_api_key", type=str, default="")
    parser.add_argument("--deepseek_model", type=str, default="deepseek-chat")
    parser.add_argument("--deepseek_base_url", type=str, default="https://api.deepseek.com/v1/chat/completions")
    parser.add_argument("--ollama_model", type=str, default="qwen2.5:1.5b")
    parser.add_argument("--ollama_base_url", type=str, default="http://localhost:11434/api/generate")
    parser.add_argument(
        "--llm_cache_path",
        type=str,
        default=str(RESULT_DIR / "deepseek_compression_cache.json"),
    )
    parser.add_argument(
        "--ollama_cache_path",
        type=str,
        default=str(RESULT_DIR / "ollama_compression_cache.json"),
    )
    parser.add_argument("--compression_output_path", type=str, default="")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def truncate_for_clip(text, max_words=20):
    words = normalize_text(text).split()
    return " ".join(words[:max_words])


def split_clauses(text):
    return [part.strip() for part in re.split(r"[,:;.!?()]+", text) if part.strip()]


def clean_clause_for_manual_phrase(clause):
    clause = normalize_text(clause)
    clause = re.sub(r"^Clad in\b", "Wearing", clause, flags=re.IGNORECASE)
    clause = re.sub(r"^From the vantage point of\b", "From", clause, flags=re.IGNORECASE)
    clause = re.sub(r"\bhorse-drawn\b", "horse drawn", clause, flags=re.IGNORECASE)
    clause = re.sub(r"\bclip-clop\b", "clip clop", clause, flags=re.IGNORECASE)
    clause = re.sub(r"\s+", " ", clause).strip(" ,")
    return clause


def score_manual_clause(clause):
    words = re.findall(r"[A-Za-z0-9']+", clause.lower())
    if not words:
        return -1
    content_score = 0
    for word in words:
        if word in FILLER_WORDS or word in ABSTRACT_WORDS:
            content_score -= 1
        elif word not in STOPWORDS:
            content_score += 2
        if word in {"man", "woman", "boy", "girl", "person", "people", "dog", "horse", "car", "boat", "bicycle", "camera"}:
            content_score += 3
        if word in {"walking", "smiling", "standing", "holding", "riding", "sitting", "driving", "performing", "reviewing", "wrapping", "grilling"}:
            content_score += 3
    return content_score


def build_manual_compressed_text(text):
    clauses = split_clauses(text)
    ranked_clauses = []
    for index, clause in enumerate(clauses):
        cleaned = clean_clause_for_manual_phrase(clause)
        if not cleaned:
            continue
        ranked_clauses.append((index, score_manual_clause(cleaned), cleaned))

    if not ranked_clauses:
        return truncate_for_clip(normalize_text(text))

    selected = sorted(ranked_clauses, key=lambda item: (-item[1], item[0]))[:4]
    selected = sorted(selected, key=lambda item: item[0])

    merged_parts = []
    current_words = 0
    for _, _, clause in selected:
        clause_words = clause.split()
        if not clause_words:
            continue
        if current_words >= 20:
            break
        remaining = 20 - current_words
        if len(clause_words) <= remaining:
            merged_parts.append(clause)
            current_words += len(clause_words)
        elif not merged_parts:
            merged_parts.append(" ".join(clause_words[:remaining]))
            current_words = 20
        elif remaining >= 4:
            merged_parts.append(" ".join(clause_words[:remaining]))
            current_words = 20
            break

    merged = ", ".join(merged_parts)
    return normalize_text(merged)


def extract_keywords(text, limit=18):
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    keywords = []
    seen = set()
    for word in words:
        if len(word) <= 2 or word in STOPWORDS or word in FILLER_WORDS:
            continue
        if word.endswith("ly") or word.endswith("ing") and len(word) > 6:
            # Keep visually meaningful verbs, skip many adverbial decorations.
            if word in {"skiing", "riding", "holding", "standing", "walking", "sitting", "jumping"}:
                pass
            else:
                continue
        if word not in seen:
            seen.add(word)
            keywords.append(word)
        if len(keywords) >= limit:
            break
    return ", ".join(keywords)


def build_text_variants(text):
    text = normalize_text(text)
    variants = [text]

    compressed = build_manual_compressed_text(text)
    if compressed:
        variants.append(compressed)

    deduped = []
    seen = set()
    for variant in variants:
        variant = normalize_text(variant)
        if not variant or variant in seen:
            continue
        seen.add(variant)
        deduped.append(variant)
    return deduped[:8]


def get_deepseek_api_key(cli_value):
    if cli_value:
        return cli_value
    env_value = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if env_value:
        return env_value
    try:
        from deepseek_config import DEEPSEEK_API_KEY  # type: ignore

        return str(DEEPSEEK_API_KEY).strip()
    except Exception:
        return ""


def load_cache(path):
    cache_path = Path(path)
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def save_cache(path, data):
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_cache_key(text, model, prompt):
    return f"{model}|||{prompt}|||{normalize_text(text)}"


def compress_with_deepseek(text, api_key, model, base_url, cache):
    text = normalize_text(text)
    cache_key = build_cache_key(text, model, DEEPSEEK_COMPRESSION_PROMPT_EN)
    if cache_key in cache:
        return cache[cache_key]

    response = requests.post(
        base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": DEEPSEEK_COMPRESSION_PROMPT_EN},
                {"role": "user", "content": text},
            ],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    compressed = data["choices"][0]["message"]["content"].strip()
    compressed = truncate_for_clip(compressed)
    cache[cache_key] = compressed
    return compressed


def compress_with_ollama(text, model, base_url, cache):
    text = normalize_text(text)
    cache_key = build_cache_key(text, model, OLLAMA_COMPRESSION_PROMPT_EN)
    if cache_key in cache:
        return cache[cache_key]

    response = requests.post(
        base_url,
        json={
            "model": model,
            "system": OLLAMA_COMPRESSION_PROMPT_EN,
            "prompt": text,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    compressed = data["response"].strip()
    compressed = truncate_for_clip(compressed)
    cache[cache_key] = compressed
    return compressed


def encode_texts(model, device, texts):
    features = []
    with torch.no_grad():
        for text in texts:
            tokens = clip.tokenize([text], truncate=True).to(device)
            text_feature = model.encode_text(tokens)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            features.append(text_feature)
    stacked = torch.cat(features, dim=0).mean(dim=0, keepdim=True)
    return stacked / stacked.norm(dim=-1, keepdim=True)


def encode_single_text(model, device, text):
    with torch.no_grad():
        tokens = clip.tokenize([text], truncate=True).to(device)
        text_feature = model.encode_text(tokens)
        return text_feature / text_feature.norm(dim=-1, keepdim=True)


def encode_images(model, preprocess, device, image_paths, batch_size):
    features = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            images = [preprocess(Image.open(path)).unsqueeze(0) for path in batch_paths]
            image_tensor = torch.cat(images, dim=0).to(device)
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
    return torch.cat(features, dim=0)


def load_or_build_image_features(model, preprocess, device, image_paths, cache_path, batch_size):
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    image_names = [path.name for path in image_paths]

    if cache_file.exists():
        cache = torch.load(cache_file, map_location="cpu")
        if cache.get("image_names") == image_names:
            return cache["features"]

    features = encode_images(model, preprocess, device, image_paths, batch_size)
    torch.save({"image_names": image_names, "features": features}, cache_file)
    return features


class TextImageRetriever:
    def __init__(
        self,
        image_dir,
        weight_path,
        cache_path,
        device="cpu",
        batch_size=32,
        deepseek_api_key="",
        deepseek_model="deepseek-chat",
        deepseek_base_url="https://api.deepseek.com/v1/chat/completions",
        llm_cache_path=str(RESULT_DIR / "deepseek_compression_cache.json"),
        ollama_model="qwen2.5:1.5b",
        ollama_base_url="http://localhost:11434/api/generate",
        ollama_cache_path=str(RESULT_DIR / "ollama_compression_cache.json"),
    ):
        self.image_dir = Path(image_dir)
        self.weight_path = weight_path
        self.cache_path = cache_path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.deepseek_api_key = get_deepseek_api_key(deepseek_api_key)
        self.deepseek_model = deepseek_model
        self.deepseek_base_url = deepseek_base_url
        self.llm_cache_path = llm_cache_path
        self.llm_cache = load_cache(llm_cache_path)
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.ollama_cache_path = ollama_cache_path
        self.ollama_cache = load_cache(ollama_cache_path)
        self.deepseek_fallback_warned = False

        self.model, self.preprocess = clip.load(self.weight_path, device=self.device)
        self.model.eval()
        self.image_paths = []
        self.image_features = None
        self.refresh_image_index(force=True)

    def refresh_image_index(self, force=False):
        current_image_paths = sorted([path for path in self.image_dir.iterdir() if path.is_file()])
        if not current_image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        if (
            force
            or not self.image_paths
            or [path.name for path in current_image_paths] != [path.name for path in self.image_paths]
        ):
            features = load_or_build_image_features(
                model=self.model,
                preprocess=self.preprocess,
                device=self.device,
                image_paths=current_image_paths,
                cache_path=self.cache_path,
                batch_size=self.batch_size,
            )
            self.image_paths = current_image_paths
            self.image_features = features.to(self.device)

    def resolve_query_text(self, caption, strategy):
        caption = normalize_text(caption)
        variants = build_text_variants(caption)
        used_text = variants[0]

        if strategy == "raw":
            used_text = variants[0]
        elif strategy == "deepseek_compressed":
            if not self.deepseek_api_key:
                if not self.deepseek_fallback_warned:
                    print("DEEPSEEK_API_KEY not found, falling back to manual_compressed strategy.")
                    self.deepseek_fallback_warned = True
                used_text = variants[1] if len(variants) > 1 else variants[0]
            else:
                used_text = compress_with_deepseek(
                    text=caption,
                    api_key=self.deepseek_api_key,
                    model=self.deepseek_model,
                    base_url=self.deepseek_base_url,
                    cache=self.llm_cache,
                )
        elif strategy == "ollama_compressed":
            used_text = compress_with_ollama(
                text=caption,
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                cache=self.ollama_cache,
            )
        else:
            used_text = variants[1] if len(variants) > 1 else variants[0]

        return caption, used_text, variants

    def search(self, caption, strategy="manual_compressed", top_k=1):
        self.refresh_image_index()
        caption, used_text, variants = self.resolve_query_text(caption, strategy)

        text_feature = encode_single_text(self.model, self.device, used_text)
        similarities = text_feature @ self.image_features.T

        top_k = max(1, min(int(top_k), similarities.size(1)))
        top_scores, top_indices = similarities.squeeze(0).topk(top_k)

        matches = []
        for index, score in zip(top_indices.tolist(), top_scores.tolist()):
            path = self.image_paths[index]
            matches.append(
                {
                    "image": path.name,
                    "image_path": str(path),
                    "score": float(score),
                }
            )

        best = matches[0]

        if strategy == "deepseek_compressed" and self.deepseek_api_key:
            save_cache(self.llm_cache_path, self.llm_cache)
        if strategy == "ollama_compressed":
            save_cache(self.ollama_cache_path, self.ollama_cache)

        return {
            "caption": caption,
            "compressed_text": used_text,
            "image": best["image"],
            "image_path": best["image_path"],
            "score": best["score"],
            "image_count": len(self.image_paths),
            "strategy": strategy,
            "matches": matches,
        }


def main():
    args = parse_args()
    captions = load_json(args.caption_path)
    retriever = TextImageRetriever(
        image_dir=args.image_dir,
        weight_path=args.weight_path,
        cache_path=args.cache_path,
        device=args.device,
        batch_size=args.batch_size,
        deepseek_api_key=args.deepseek_api_key,
        deepseek_model=args.deepseek_model,
        deepseek_base_url=args.deepseek_base_url,
        llm_cache_path=args.llm_cache_path,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        ollama_cache_path=args.ollama_cache_path,
    )

    results = []
    compression_records = []
    for item in captions:
        search_result = retriever.search(item["caption"], strategy=args.strategy)
        results.append({
            "caption": search_result["caption"],
            "image": search_result["image"],
        })
        compression_records.append({
            "caption": search_result["caption"],
            "compressed_text": search_result["compressed_text"],
            "strategy": args.strategy,
        })

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    if args.compression_output_path:
        compression_output_path = Path(args.compression_output_path)
        compression_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(compression_output_path, "w", encoding="utf-8") as file:
            json.dump(compression_records, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
