# Tái lập VoiceSnake desktop

## Chạy đúng model đã phát hành

Dùng source, dependency runtime cố định, gồm dependency gián tiếp trong `constraints-runtime.txt` và model trong cùng bản release. `scripts/bootstrap.py` tạo môi trường riêng rồi cài runtime; `run_game.py` chọn môi trường đó. Không chuyển `.venv` giữa hệ điều hành hay CPU khác nhau. Python khuyến nghị 3.11, các phiên bản hỗ trợ 3.9–3.12, 64 bit. PyTorch runtime dùng 2.8.0; bootstrap chọn nguồn wheel theo nền tảng.

Chạy `python -m voicesnake --doctor` bằng Python của môi trường để kiểm tra dependency, model/nhãn và checksum theo manifest trong `voicesnake/assets/`. Giữ nguyên cả model, labels và manifest khi chuyển máy. Checksum giống nhau xác nhận cùng bytes của model; không đảm bảo cùng độ trễ hoặc cùng dữ liệu microphone.

Runtime không cần dataset hay huấn luyện. Thiết lập mặc định và tham số dòng lệnh/settings quyết định detector; khi so sánh hai máy hãy giữ threshold, energy, confirmations, interval và tốc độ giống nhau. Model nhận 16 kHz mono; microphone được ưu tiên mở 16 kHz, nếu không hỗ trợ thì thu sample rate mặc định và resample về 16 kHz; frontend STFT đã đóng gói trong TorchScript.

Inference có thể chênh lệch nhỏ theo backend số học/CPU. Microphone, gain, lọc nhiễu hệ điều hành, tiếng nền và vị trí nói có thể thay đổi kết quả nhiều hơn. Chưa xác minh microphone thực tế và chưa xác nhận chạy trên mọi hệ điều hành; diagnostic thành công không thay thế việc chơi thử trên máy đích.

## Train lại từ đầu

Mở notebook ở thư mục gốc dự án, dùng môi trường riêng có dependency trong `requirements-train.txt`:

```bash
python -m pip install -r requirements-train.txt
```

Chạy tất cả cell theo thứ tự. Notebook tự tải mini Speech Commands nếu dữ liệu chưa có; cần Internet và dung lượng cho archive/dataset. Training không cần thiết để chơi game. Ghi lại Python, phiên bản thư viện, thiết bị, số thread, output notebook và checksum dataset khi báo cáo một lượt train mới.

Cấu hình lượt gốc:

- Seed 42 cho Python, NumPy và PyTorch; seed CUDA nếu có.
- Nhãn theo đúng thứ tự `up, right, left, down`, không sắp xếp lại.
- Speaker ID là phần tên trước `_nohash_`; SHA-1 của ID chia speaker vào train/validation/test theo tỷ lệ xấp xỉ 70/15/15. Speaker không xuất hiện đồng thời ở các tập. Tỷ lệ số clip thực tế được ghi trong notebook/split manifest.
- GRU khởi tạo ngẫu nhiên, batch 64, tối đa 35 epoch, learning rate 0.001, early stopping patience 8. Chọn checkpoint theo validation loss; test không dùng chọn checkpoint.
- Lượt gốc chọn epoch 18, kết thúc epoch 26. Test 637 clip: accuracy 0.940345, macro-F1 0.940698. Đây là kết quả đã lưu, không phải cam kết lần train mới ra cùng số.

Seed cố định hỗ trợ so sánh, nhưng notebook không bảo đảm tái lập bit-for-bit giữa phiên bản PyTorch, CPU, GPU và kernel khác nhau. Train lại có thể tạo checkpoint/hash và chỉ số khác. Để tái lập chính xác model inference hiện tại, dùng artifact đã phát hành và kiểm tra checksum thay vì chỉ chạy lại training.

Notebook xuất `training_outputs/voicesnake_gru4/`: TorchScript, `labels.json`, `best_state.pt`, lịch sử, split manifest, metrics và biểu đồ. `best_state.pt` là trọng số cho kiến trúc notebook; game sử dụng file TorchScript `voicesnake_gru4.pt`. Train lại không tự thay model đóng gói. Khi chủ động phát hành model mới, cập nhật đồng bộ TorchScript, labels, manifest/checksum và kết quả đánh giá; không sửa checksum chỉ để bỏ qua lỗi file cũ.

## Phạm vi chỉ số

Benchmark batch 1 CPU trong notebook gồm frontend, khoảng p50 1,85 ms. Nó không gồm capture microphone, cửa sổ audio, confirmation, cooldown hay chờ game tick.

Trong 690 clip từ ngoài bốn hướng, 82,17% vượt confidence 0.7 và RMS 0.004 trong phép đo toàn clip. Đây không phải tỷ lệ kích hoạt sai của detector streaming, cũng không phải false activations/phút. Bốn lớp softmax không có lớp unknown: confidence cao vẫn có thể là từ khác. Muốn đo điều khiển thực tế cần thu âm các hướng, tiếng nền/từ khác và ghi số lệnh sai trên tổng phút audio ngoài lệnh.

## Nguồn dữ liệu

Notebook dùng **TensorFlow mini Speech Commands**, bản trích nhỏ của Speech Commands Dataset. File `data/mini_speech_commands/README.md` trong dữ liệu gốc ghi rõ đây là excerpt cho tutorial và dẫn tới dataset gốc để xem tài liệu/giấy phép; file đó không tự ghi đầy đủ điều khoản license. Giữ attribution này và kiểm tra giấy phép archive gốc trước khi phân phối lại audio. Bản desktop không cần phân phối audio training.

Model GRU này được train mới bằng notebook dự án. Các kết quả và notebook training là bằng chứng cho lượt huấn luyện gốc, không xác nhận model hoạt động giống nhau trên mọi microphone hoặc môi trường.
