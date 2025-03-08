import json
import nltk
from tqdm import tqdm
from pacsum.summarizer import PacSumBert

# Tải mô hình PacSum dựa trên BERT
model = PacSumBert(model_type="bert", bert_model_path="bert-base-uncased")

# Đọc dữ liệu từ file JSON
input_file = "/kaggle/input/inputtext/input_for_BART.json"
with open(input_file, "r", encoding="utf-8") as f:
    docs = json.load(f)
    
    # Dictionary để lưu kết quả tóm tắt
pacsum_output = {}

# Áp dụng PacSum cho từng tài liệu
for ID in tqdm(docs):
    input_text = docs[ID]
    
    # Sử dụng nltk để chia đoạn văn thành câu
    sentences = nltk.sent_tokenize(input_text)
    
    # Chọn ra các câu quan trọng nhất
    summary_sentences = model.summarize(sentences, num_sentences=10)
    
    # Ghép lại thành đoạn văn tóm tắt
    summary = " ".join(summary_sentences)
    pacsum_output[ID] = summary

# Lưu kết quả vào file JSON
output_file = "/kaggle/working/pacsum_summary.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(pacsum_output, f, indent=4)

print("✅ Tóm tắt hoàn tất! Kết quả được lưu trong", output_file)
