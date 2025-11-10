import json

train_data_file = "/home/guoteng/code/DeepSeek-OCR-Research/train_data/train_data_chat.jsonl"

def create_single_data(text: str, image_path: str):
    data = {
            "messages": [
                {"role": "user", "content": "<image>\nExtract the text in the image."},
                {"role": "assistant", "content": text}
            ],
            "images": [image_path]
        }
    with open(train_data_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    create_single_data("豫章故郡，洪都新府。星分翼轸，地接衡庐。襟三江而带五湖...", "/home/guoteng/code/DeepSeek-OCR-Research/train_data/images/tengwangge.png")