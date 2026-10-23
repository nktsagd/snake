# VoiceSnake desktop

Game Snake điều khiển bằng bốn từ tiếng Anh **up, right, left, down**, hoặc bàn phím. Model GRU được train từ đầu và đóng gói sẵn cùng chương trình. Chơi trên CPU, xử lý microphone tại máy; không cần train lại hoặc tải dataset để chơi.

## Cài và chạy trên máy mới

Giải nén dự án vào thư mục bạn có quyền ghi, mở Terminal/PowerShell trong thư mục đó. Cần **Python 3.9–3.12, bản 64 bit**; khuyến nghị Python 3.11. Bộ wheel macOS hiện tại dành cho Apple Silicon; Intel macOS không được hỗ trợ trong bản này. Trên Windows, chọn “Add Python to PATH” khi cài Python.

**Windows:**

```powershell
py -3.11 scripts/bootstrap.py
py -3.11 run_game.py
```

**macOS / Linux:**

```bash
python3 scripts/bootstrap.py
python3 run_game.py
```

Các dependency runtime trực tiếp và gián tiếp được cố định trong `requirements.txt` / `constraints-runtime.txt`. Script đầu tạo `.venv` và cài dependency vào đó; cần Internet lần đầu. Script sau dùng Python trong `.venv`. Khi cài xong có thể chơi offline. Không copy `.venv` từ máy khác và không dùng `sudo pip`.

Trên Ubuntu/Debian, cài thư viện audio hệ thống nếu thiếu:

```bash
sudo apt install libportaudio2 libsndfile1
```

Cho phép ứng dụng Terminal/Python truy cập microphone trong cài đặt quyền riêng tư của hệ điều hành. macOS: **System Settings → Privacy & Security → Microphone**. Windows: **Settings → Privacy & security → Microphone**. Cần phiên desktop có màn hình và thiết bị audio để chơi bằng giọng nói.

## Kiểm tra môi trường

Dùng Python trong môi trường vừa tạo:

```bash
# macOS / Linux
.venv/bin/python -m voicesnake --doctor
.venv/bin/python -m voicesnake --list-devices
.venv/bin/python -m voicesnake --keyboard-only
```

```powershell
# Windows
.venv\Scripts\python.exe -m voicesnake --doctor
.venv\Scripts\python.exe -m voicesnake --list-devices
.venv\Scripts\python.exe -m voicesnake --keyboard-only
```

Sau khi kích hoạt `.venv`, cũng có thể chạy `voicesnake` hoặc `python -m voicesnake`. Chọn microphone cụ thể bằng `--device 0`; dùng chỉ số in bởi `--list-devices`.

## Cách chơi

| Điều khiển | Chức năng |
|---|---|
| Nói up / right / left / down | Chuyển sang hướng tuyệt đối |
| Mũi tên / WASD | Chuyển hướng bằng bàn phím |
| Space | Pause/resume |
| R | Chơi lại |
| V | Bật/tắt voice |
| SETTINGS / Tab | Mở bảng tham số |
| + / − | Chỉnh tốc độ |
| Esc | Đóng settings; nếu đang đóng thì thoát |

Rắn xuyên tường sang mép đối diện, va vào thân thì thua. Game chặn quay ngược 180°: đang đi right thì cần up/down trước khi left.

Trong SETTINGS, **Threshold, Energy, Confirmations** được cập nhật cho vòng xử lý audio kế tiếp; **Speed** cập nhật lịch di chuyển. Không cần restart. Thay đổi chỉ có hiệu lực trong phiên hiện tại. Giảm threshold/confirmations để dễ nhận lệnh hơn; giảm energy nếu giọng nhỏ. Độ nhạy cao hơn có thể tăng nhận nhầm.

`PREDICTION` và phần trăm là dự đoán thô, chưa phải lệnh đã được chấp nhận. `DETECTOR` giải thích bộ lọc; `LAST COMMAND` cho biết QUEUED / APPLIED / ALREADY / BLOCKED. Confidence cao không đảm bảo rắn sẽ rẽ nếu hướng bị chặn hoặc detector đang chờ xác nhận.

Ví dụ đặt tham số khi khởi động:

```bash
python -m voicesnake --threshold 0.55 --energy 0.004 --confirmations 2 --speed 5
```

## Model và dữ liệu ghi ra

Model mặc định nằm trong `voicesnake/assets/`, đi cùng nhãn và manifest checksum. Đây là GRU bốn lớp riêng của dự án, 80.452 tham số. Input model: mono 16 kHz, cửa sổ một giây; nếu microphone chỉ hỗ trợ 44,1/48 kHz thì chương trình resample liên tục về 16 kHz; frontend STFT nằm trong model TorchScript. Không cần checkpoint ngoài để chơi.

Dữ liệu có thể ghi nằm ngoài thư mục cài: mặc định `~/.voicesnake` trên macOS/Linux hoặc `%LOCALAPPDATA%\VoiceSnake` trên Windows. Đặt biến môi trường `VOICESNAKE_HOME` để chọn thư mục khác. `--metrics` cho phép chỉ định đường dẫn log phiên. Không copy microphone recording cá nhân khi chia sẻ dự án.

Kết quả lượt train đã lưu: **94,03% accuracy / macro-F1 0,9407 trên 637 clip test**, chia theo người nói. Inference CPU trong notebook khoảng **1,85 ms p50**, không phải độ trễ microphone đến khi rắn rẽ. Model chỉ có bốn lớp vẫn có thể gán từ khác vào hướng với confidence cao; chưa có kết quả microphone thực tế trên từng máy đích.

## Train lại và chia sẻ

Notebook `VoiceSnake_Train_GRU_4_Commands.ipynb` chứa code training và output lượt chạy gốc. Xem [hướng dẫn tái lập](docs/REPRODUCIBILITY.md) để phân biệt chạy lại đúng model đã đóng gói với train lại model mới.

Bản chia sẻ `release/VoiceSnake-desktop.zip` chứa source và model cần thiết, không chứa môi trường `.venv`, dataset hay cache. Người nhận giải nén rồi làm theo mục cài đặt. Đây là dự án Python cần cài dependency, chưa phải ứng dụng thực thi độc lập. Windows/Linux chưa được xác nhận chạy thực tế trong phiên phát triển này.

Dữ liệu training lấy từ TensorFlow **mini Speech Commands**, một phần của Speech Commands Dataset. Thông tin nguồn và phạm vi xác minh giấy phép được ghi trong tài liệu tái lập.
