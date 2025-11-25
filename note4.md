**Giải thuật (Algorithm Flow):**

1.  **Bước 1: Nhị phân hóa (Thresholding)**

      * Chuyển ảnh đầu vào thành ảnh đen trắng (Binary Image): Vật thể màu trắng (1), nền màu đen (0).

2.  **Bước 2: Lấp đầy vùng (Region Filling) - *Bước quan trọng nhất***

      * [cite\_start]Đây là kỹ thuật trong slide "Region Filling" [cite: 1682-1688].
      * *Nguyên lý:* Ta cần tìm tất cả các "lỗ thủng" (vùng đen nằm lọt thỏm trong vùng trắng) và tô trắng chúng.
      * *Cách làm:*
        1.  Tô màu nền (FloodFill) từ viền ảnh vào trong. Tất cả những vùng đen nào *không bị tô* chính là các lỗ hổng (do bị viền trắng bao quanh ngăn lại).
        2.  Đảo ngược vùng nền vừa tô $\rightarrow$ Thu được các lỗ hổng.
        3.  Cộng lỗ hổng vào ảnh gốc $\rightarrow$ Thu được vật thể đặc.

3.  **Bước 3: Gán nhãn thành phần liên thông (Connected Components)**

      * [cite\_start]Dựa trên slide "Extraction of Connected Components" [cite: 1704-1708] [cite\_start]và bảng số liệu ở Slide 20[cite: 1207].
      * Máy tính quét toàn bộ ảnh. Các điểm trắng dính liền nhau sẽ được gán cùng một mã số (Label 1, Label 2, ..., Label n).
      * **Số lượng ($n$):** Chính là số nhãn lớn nhất tìm được.
      * **Diện tích (Area):** Đếm tổng số pixel của từng nhãn.

-----

* **Quy trình (Pipeline) chuẩn:**
    1.  **Opening (Mở):** Co $\rightarrow$ Giãn.
        * *Tác dụng:* Xóa các chấm trắng nhỏ (nhiễu nền), làm trơn đường viền lồi.
        * *Lưu ý:* Kích thước kernel quyết định kích thước hạt nhiễu bị xóa.
    2.  **Closing (Đóng):** Giãn $\rightarrow$ Co.
        * *Tác dụng:* Lấp đầy các lỗ đen nhỏ (trên đường vân), nối liền các vết đứt gãy.
        * *Lưu ý:* Thực hiện SAU Opening để đạt hiệu quả tốt nhất (như sơ đồ trong hình).

**Về code:** Chúng ta cần viết một hàm `fingerprint_cleaning` (hoặc tên tương tự) thực hiện chuỗi `Closing(Opening(img))`.
