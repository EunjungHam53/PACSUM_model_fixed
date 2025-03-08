import json
import h5py
import nltk
from nltk.tokenize import sent_tokenize

# Tải mô hình tách câu của nltk (chỉ cần chạy một lần)
nltk.download("punkt")

# Đường dẫn file
json_file = "/kaggle/input/inputtext/input_for_BART.json"
hdf5_file = "/kaggle/working/input.h5df"

# Đọc file JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Xử lý dữ liệu: tách câu và tạo abstract
processed_data = []
for key, text in data.items():
    sentences = sent_tokenize(text)  # Tách câu

    if len(sentences) > 1:
        abstract_size = max(1, len(sentences) // 2)  # Lấy 50% số câu, ít nhất là 1 câu
    else:
        abstract_size = 1  # Nếu chỉ có 1 câu thì giữ nguyên

    abstract = sentences[:abstract_size]  # Lấy nửa đầu làm abstract

    # Lưu dữ liệu vào danh sách
    processed_data.append({
        "article": sentences,
        "abstract": abstract
    })

# Lưu vào HDF5
with h5py.File(hdf5_file, "w") as hdf:
    dataset = hdf.create_dataset("dataset", data=[json.dumps(item).encode("utf-8") for item in processed_data])

print(f"✅ Chuyển đổi thành công! File HDF5 đã lưu tại: {hdf5_file}")