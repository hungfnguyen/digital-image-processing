**1. Miền tần số là gì? (Khác gì miền không gian?)**
* [cite_start]**Miền không gian (Spatial Domain):** Là cái bạn nhìn thấy bình thường, xử lý trực tiếp trên giá trị của từng điểm ảnh (pixel)[cite: 28].
* **Miền tần số (Frequency Domain):** Là cách nhìn bức ảnh dưới dạng "sự thay đổi". [cite_start]Thay vì tọa độ (x, y), nó dùng tần số để mô tả mức độ thay đổi độ sáng nhanh hay chậm[cite: 18, 28].
    * [cite_start]**Tần số thấp (Low Frequency):** Vùng ảnh mượt, màu sắc thay đổi từ từ (ví dụ: bầu trời, bức tường trơn)[cite: 313].
    * [cite_start]**Tần số cao (High Frequency):** Vùng ảnh thay đổi độ sáng đột ngột (ví dụ: cạnh bàn, sợi tóc, nhiễu)[cite: 313].

**2. Quy trình xử lý (Thần chú 3 bước)**
Thay vì chỉnh sửa trực tiếp pixel, ta đi đường vòng:
1.  [cite_start]**Biến đổi:** Dùng **Fourier Transform (DFT/FFT)** để chuyển ảnh từ miền không gian sang miền tần số [cite: 170-171].
2.  [cite_start]**Lọc:** Nhân bức ảnh (lúc này là các tần số) với một cái **Bộ lọc (Filter)** để giữ lại hoặc loại bỏ các tần số mong muốn[cite: 174, 177].
3.  [cite_start]**Nghịch đảo:** Dùng **Inverse Fourier Transform (IDFT)** để chuyển ngược về miền không gian để xem kết quả[cite: 176, 184].

**3. Hai loại lọc cơ bản nhất**
* **Lọc thông thấp (Lowpass Filter):**
    * [cite_start]**Cơ chế:** Giữ lại tần số thấp, chặn tần số cao[cite: 43].
    * [cite_start]**Kết quả:** Ảnh bị **mờ đi (làm trơn)**, dùng để khử nhiễu hoặc làm mịn da[cite: 299, 314].
    * [cite_start]Ví dụ: Ideal, Butterworth, Gaussian Lowpass [cite: 300-302].
* **Lọc thông cao (Highpass Filter):**
    * [cite_start]**Cơ chế:** Giữ lại tần số cao, chặn tần số thấp[cite: 48].
    * [cite_start]**Kết quả:** Ảnh trở nên **sắc nét hơn**, các đường biên/cạnh nổi rõ lên (dùng để làm nét ảnh)[cite: 488, 512].

> "Xử lý miền tần số là chuyển ảnh sang dạng sóng. Muốn ảnh mờ đi thì lọc bỏ sóng cao (Lowpass), muốn ảnh nét lên thì lọc bỏ sóng thấp (Highpass), sau đó chuyển ngược lại."