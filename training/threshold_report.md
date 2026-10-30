# So sánh ngưỡng confidence

Đánh giá file `public/voicesnake.onnx` bằng ONNX Runtime CPU. Lệnh chỉ được chấp nhận nếu argmax là một trong bốn hướng và confidence đạt ngưỡng. Nhóm từ lạ đã thấy gồm các từ ngoài lệnh có trong train; nhóm chưa thấy gồm `cat`, `dog` ở validation và `bed`, `bird` ở test. Background lấy từ một file riêng cho mỗi split. Đây là clip rời, không áp dụng confirmation hoặc RMS.

| Ngưỡng | Validation: lệnh đúng / 2.449 | Validation: từ lạ đã thấy / 4.320 | Validation: từ lạ chưa thấy / 320 | Test: lệnh đúng / 2.503 | Test: từ lạ đã thấy / 4.320 | Test: từ lạ chưa thấy / 320 |
|---:|---:|---:|---:|---:|---:|---:|
| 0,70 | 86,0% | 4,21% | 21,56% | 83,7% | 4,98% | 8,44% |
| 0,80 | 82,6% | 2,78% | 13,75% | 80,0% | 3,17% | 4,38% |
| 0,85 | 80,0% | 2,18% | 11,88% | 77,7% | 2,34% | 3,75% |
| **0,90** | **75,5%** | **1,55%** | **7,19%** | **73,4%** | **1,44%** | **1,88%** |
| 0,919165 | 73,1% | 1,23% | 5,00% | 70,9% | 1,18% | 0,94% |
| 0,95 | 65,9% | 0,81% | 2,19% | 65,2% | 0,63% | 0,31% |

Các cột từ lạ là tỷ lệ **bị nhận nhầm thành lệnh**: thấp hơn là tốt hơn. Background: 0/61 clip bị nhận thành lệnh ở mọi ngưỡng trong bảng, trên cả hai split; không thể suy rộng kết luận từ hai nguồn noise này.

Ngưỡng **0,90 từng được đề xuất**: trên validation tăng tỷ lệ nhận đúng lệnh 2,4 điểm phần trăm so với 0,919165, đổi lại tỷ lệ nhận nhầm từ chưa thấy tăng từ 5,0% lên 7,19%. Game hiện dùng mặc định **0,70 và một lần xác nhận** theo lựa chọn của người dùng; bảng ở trên là phép đo clip, còn mô phỏng bên dưới dùng hai lần xác nhận. Ngưỡng 0,85 nhận thêm lệnh nhưng tỷ lệ nhận nhầm nhóm chưa thấy trên validation tăng lên 11,88%. Test chỉ mang tính tham khảo vì tập này đã được dùng khi so sánh và chọn model; không phải một phép xác nhận hoàn toàn độc lập.

## Phát lại cửa sổ trượt

Mỗi split: 100 clip lệnh (25 mỗi hướng), 50 từ lạ đã thấy, 50 từ lạ chưa thấy và 50 cửa sổ background. Mỗi clip được chèn giữa khoảng im lặng, dùng cửa sổ 1 giây, bước 100 ms, RMS 0,004, confirmation 2, cooldown 350 ms. Các thông số này mô phỏng logic quyết định của game; không gồm AudioWorklet, độ trễ worker, microphone hoặc xử lý âm thanh trình duyệt. Mẫu nhỏ, được chọn từ cùng các split ở trên, không độc lập với phép đo clip.

| Ngưỡng | Validation: lệnh đúng / 100 | Validation: từ lạ chưa thấy bị nhận / 50 | Test: lệnh đúng / 100 | Test: từ lạ chưa thấy bị nhận / 50 |
|---:|---:|---:|---:|---:|
| 0,80 | 79 | 10 | 82 | 4 |
| 0,85 | 76 | 7 | 80 | 3 |
| **0,90** | **73** | **3** | **75** | **2** |
| 0,919165 | 68 | 3 | 72 | 1 |
| 0,95 | 61 | 1 | 66 | 1 |

Trong mô phỏng validation, ở ngưỡng 0,70 có 39/50 đoạn background bị phát lệnh, trong khi ở 0,80 trở lên là 0/50. Các đoạn này đều lấy từ cùng một file và có ranh giới im lặng nhân tạo, nên đây là dấu hiệu cần thử thêm, **không phải tỷ lệ kích hoạt sai trên audio liên tục thực tế**.

Chạy lại bằng `.venv/bin/python training/evaluate_thresholds.py` sau khi tạo manifest theo README. Kết quả số đầy đủ lưu trong [threshold_sweep.json](threshold_sweep.json). Cần thu và kiểm thử micro thật, hội thoại và âm thanh game, đo false activations/giờ và lệnh bị bỏ sót trước khi chốt ngưỡng production.
