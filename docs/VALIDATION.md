# Kiểm tra bản desktop — 2026-10-23

Môi trường đã chạy: macOS arm64, Python 3.9.6 64 bit, PyTorch 2.8.0 CPU. Khuyến nghị Python 3.11 khi tạo môi trường mới; chưa kiểm chứng trực tiếp Python 3.11/Windows/Linux trong lượt này.

Đã thực hiện:

1. Cài package vào virtualenv mới không kế thừa system-site-packages; dependencies runtime tải/cài thành công.
2. Tạo ZIP source/model bằng allowlist; xác minh SHA-256 của mọi thành viên theo `SHA256SUMS.json`. ZIP không chứa dataset, recordings, môi trường `.venv` hay cache.
3. Giải nén sang đường dẫn khác có dấu cách, build/cài wheel từ source đã giải nén.
4. Chạy từ một working directory ngoài source. Xác nhận Python import package từ site-packages và load model từ assets trong package đã cài, không phụ thuộc checkout gốc.
5. `--doctor`: dependency và model checksum hợp lệ, CPU inference chạy được, output folder có dấu cách ghi được. Microphone báo WARN vì môi trường không cung cấp thiết bị input.
6. Chạy game loop Pygame với SDL dummy, keyboard-only, 10 frames; đây là kiểm tra headless, chưa phải tương tác cửa sổ/microphone thật.
7. 12 tests đạt trong môi trường sạch: hướng/rắn/va chạm, confirmation và khóa lặp, buffer, gate silence, model asset/inference, output path override và resampling 44,1/48 kHz. Resampling kiểm tra liên tục qua block, số samples, biên độ và suy giảm alias.
8. Notebook vẫn đúng schema và cú pháp Python; giữ output training đã có. Không train lại model trong lượt chỉnh portability.

Giới hạn: chưa thử microphone thật, trình điều khiển audio, UI tương tác trên máy đích hoặc Windows/Linux. Bộ cài cần Internet lần đầu và thư viện hệ thống phù hợp; sau cài có thể inference offline. Đây là bản source/model Python, không phải executable độc lập cho mọi hệ điều hành.
