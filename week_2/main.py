import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

# Import các module xử lý
import transformations as trans
import filters as ftr
import histogram as hist
# Đảm bảo bạn đã có file frequency.py
import frequency as freq
# Đảm bảo bạn đã tạo file morphology.py
import morphology as morph 

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Xử lý ảnh số - Full Features (Week 2, 3, 4)")
        # Tăng chiều rộng cửa sổ để chứa đủ 3 cột điều khiển
        self.root.geometry("1800x950") 
        self.root.configure(bg="#e0e0e0")

        self.original_image = None
        self.processed_image = None

        # --- VÙNG HIỂN THỊ ẢNH ---
        # Dời ảnh sang trái để nhường chỗ cho panel điều khiển bên phải
        self.left_label = tk.Label(self.root, bg="#ccc", text="Ảnh gốc")
        self.left_label.place(x=20, y=30, width=480, height=480)

        self.right_label = tk.Label(self.root, bg="#ccc", text="Ảnh sau xử lý")
        self.right_label.place(x=510, y=30, width=480, height=480)

        # --- MENU COMBOBOX ---
        tk.Label(self.root, text="Chọn phép biến đổi", bg="#e0e0e0", font=("Arial", 11, "bold")).place(x=1000, y=10)
        self.method_var = tk.StringVar(value="Negative image")
        self.method_box = ttk.Combobox(
            self.root,
            textvariable=self.method_var,
            values=[
                "Negative image", "Biến đổi Log", "Biến đổi Piecewise-Linear", "Biến đổi Gamma",
                "Làm trơn ảnh (lọc trung bình)", "Làm trơn ảnh (lọc Gauss)", "Làm trơn ảnh (lọc trung vị)",
                "Cân bằng sáng dùng Histogram", "Phát hiện biên (Sobel)", "Phát hiện biên (Laplacian)",
                "--- MIỀN TẦN SỐ (HW3) ---",
                "Lọc Ideal Lowpass", "Lọc Ideal Highpass",
                "Lọc Butterworth Lowpass", "Lọc Butterworth Highpass",
                "Lọc Gaussian Lowpass", "Lọc Gaussian Highpass",
                "--- HÌNH THÁI HỌC (HW4) ---",
                "Erosion (Co)", "Dilation (Giãn)",
                "Opening (Mở - Xóa nhiễu)", "Closing (Đóng - Lấp lỗ)",
                "Boundary Extraction (Trích biên)", "Morphological Gradient"
            ],
            state="readonly", width=35
        )
        self.method_box.place(x=1000, y=35)
        self.method_box.bind("<<ComboboxSelected>>", lambda e: self.update_image())

        # --- KHUNG ĐIỀU KHIỂN CHÍNH (SCROLLABLE) ---
        self.right_container = tk.Frame(self.root, bg="#f8f8f8", relief="ridge", bd=3)
        # Chiều rộng 780 để chứa đủ 3 cột con
        self.right_container.place(x=1000, y=70, width=780, height=850) 

        self.canvas = tk.Canvas(self.right_container, bg="#f8f8f8", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.right_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f8f8f8")

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Tạo giao diện điều khiển (3 cột)
        self.create_sliders(self.scrollable_frame)

        # --- BUTTONS ---
        tk.Button(self.root, text="Chọn ảnh", command=self.open_image, bg="#ddd", font=("Arial", 10, "bold")).place(x=20, y=520, width=100, height=40)
        tk.Button(self.root, text="Cập nhật", command=self.update_image, bg="#87CEEB", font=("Arial", 10, "bold")).place(x=140, y=520, width=100, height=40)
        tk.Button(self.root, text="Lưu ảnh", command=self.save_image, bg="#90EE90", font=("Arial", 10, "bold")).place(x=260, y=520, width=100, height=40)

    def create_sliders(self, parent):
        self.params = {}
        
        # === CHIA 3 CỘT (FRAMES) ===
        col1 = tk.Frame(parent, bg=parent["bg"], width=250)
        col1.pack(side="left", fill="both", expand=True, padx=5, anchor="n")
        
        col2 = tk.Frame(parent, bg=parent["bg"], width=250)
        col2.pack(side="left", fill="both", expand=True, padx=5, anchor="n")
        
        col3 = tk.Frame(parent, bg=parent["bg"], width=250)
        col3.pack(side="left", fill="both", expand=True, padx=5, anchor="n")

        def add_slider(frame_parent, text, from_, to, init, step=1):
            f = tk.Frame(frame_parent, bg=frame_parent["bg"])
            f.pack(pady=2, fill="x")
            tk.Label(f, text=text, bg=frame_parent["bg"], anchor="w").pack(padx=5)
            var = tk.DoubleVar(value=init)
            tk.Scale(f, from_=from_, to=to, orient="horizontal", resolution=step, variable=var, bg=frame_parent["bg"], length=200).pack(padx=5)
            self.params[text] = var

        # --- CỘT 1: TUẦN 2 (SPATIAL) ---
        tk.Label(col1, text="--- TUẦN 2 (CƠ BẢN) ---", bg="#f8f8f8", fg="blue", font=("Arial", 10, "bold")).pack(pady=5)
        
        s1 = tk.LabelFrame(col1, text="Biến đổi mức xám", bg="#C2E0F2")
        s1.pack(fill="x", padx=2)
        add_slider(s1, "Hệ số C (Log)", 1, 100, 40)
        add_slider(s1, "Gamma", 0.1, 5.0, 1.0, step=0.1)

        s2 = tk.LabelFrame(col1, text="Piecewise Linear", bg="#E6B8F2")
        s2.pack(fill="x", padx=2)
        add_slider(s2, "Hệ số Cao (r2)", 0, 255, 140)
        add_slider(s2, "Hệ số Thấp (r1)", 0, 255, 70)

        s3 = tk.LabelFrame(col1, text="Lọc Không Gian", bg="#F7CAAC")
        s3.pack(fill="x", padx=2, pady=5)
        add_slider(s3, "Kích thước lọc (Spatial)", 1, 31, 3, step=2)
        add_slider(s3, "Hệ số Sigma (Gauss)", 0, 10, 1, step=0.5)

        # --- CỘT 2: TUẦN 3 (FREQUENCY) ---
        tk.Label(col2, text="--- TUẦN 3 (TẦN SỐ) ---", bg="#f8f8f8", fg="red", font=("Arial", 10, "bold")).pack(pady=5)
        
        s4 = tk.LabelFrame(col2, text="Tham số Tần Số", bg="#FFFACD")
        s4.pack(fill="x", padx=2)
        add_slider(s4, "Tần số cắt (D0)", 1, 200, 30)
        add_slider(s4, "Bậc bộ lọc (n)", 1, 10, 2)

        hw1 = tk.LabelFrame(col2, text="HW3-1: LP + HP", bg="#E0F7FA")
        hw1.pack(fill="x", padx=2, pady=5)
        tk.Button(hw1, text="Chạy HW3-1", command=self.run_hw3_1).pack(fill="x", padx=5, pady=2)
        self.lbl_time1 = tk.Label(hw1, text="Time: -- ms", bg="#E0F7FA", font=("Arial", 8))
        self.lbl_time1.pack()

        hw2 = tk.LabelFrame(col2, text="HW3-2: HP n lần", bg="#E0F7FA")
        hw2.pack(fill="x", padx=2, pady=5)
        self.passes_var = tk.IntVar(value=1)
        tk.Radiobutton(hw2, text="1 lần", variable=self.passes_var, value=1, bg="#E0F7FA").pack(anchor="w")
        tk.Radiobutton(hw2, text="10 lần", variable=self.passes_var, value=10, bg="#E0F7FA").pack(anchor="w")
        tk.Radiobutton(hw2, text="100 lần", variable=self.passes_var, value=100, bg="#E0F7FA").pack(anchor="w")
        tk.Button(hw2, text="Chạy HW3-2", command=self.run_hw3_2).pack(fill="x", padx=5)
        
        tk.Button(col2, text="So sánh Spatial vs Freq", command=self.run_benchmark, bg="#FFCC80").pack(fill="x", pady=10)
        self.lbl_bench = tk.Label(col2, text="", bg="#f8f8f8", font=("Arial", 8))
        self.lbl_bench.pack()

        # --- CỘT 3: TUẦN 4 (MORPHOLOGY) ---
        tk.Label(col3, text="--- TUẦN 4 (HÌNH THÁI) ---", bg="#f8f8f8", fg="green", font=("Arial", 10, "bold")).pack(pady=5)
        
        s5 = tk.LabelFrame(col3, text="Tham số Hình Thái", bg="#D5E8D4")
        s5.pack(fill="x", padx=2)
        add_slider(s5, "Kích thước Morph (Kernel)", 1, 50, 3, step=2)
        
        # Bài tập tuần 4
        hw4 = tk.LabelFrame(col3, text="HW4: Đếm & Diện Tích", bg="#E1D5E7")
        hw4.pack(fill="x", padx=2, pady=10)
        tk.Button(hw4, text="1. Lấp lỗ (Region Filling)", command=self.run_fill_holes, bg="white").pack(fill="x", padx=5, pady=2)
        tk.Button(hw4, text="2. Đếm & Đo Diện Tích", command=self.run_count_objects, bg="white").pack(fill="x", padx=5, pady=2)
        
        self.lbl_count = tk.Label(hw4, text="Số lượng: --\nDiện tích: --", bg="#E1D5E7", justify="left", font=("Arial", 9))
        self.lbl_count.pack(pady=5)
        
        # Bài tập HW6 (Lọc vân tay)
        hw6_frame = tk.LabelFrame(col3, text="HW6: Lọc Vân Tay", bg="#E1D5E7")
        hw6_frame.pack(fill="x", padx=2, pady=10)
        tk.Button(hw6_frame, text="Làm sạch (Open -> Close)", command=self.run_fingerprint_clean, bg="white").pack(fill="x", padx=5, pady=2)
        
        tk.Label(col3, text="* Morph hoạt động tốt nhất\ntrên ảnh nhị phân (đen trắng).", bg="#f8f8f8", fg="#555", font=("Arial", 8, "italic")).pack(pady=5)

    # --- XỬ LÝ ẢNH ---
    def open_image(self):
        fp = filedialog.askopenfilename()
        if fp:
            self.original_image = cv2.imread(fp)
            self.display_image(self.original_image, self.left_label)

    def display_image(self, img, label):
        if img is None: return
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        p = Image.fromarray(img)
        p.thumbnail((480, 480))
        imgtk = ImageTk.PhotoImage(p)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def update_image(self):
        if self.original_image is None: return
        method = self.method_var.get()
        img = self.original_image.copy()

        # Lấy params
        c = self.params["Hệ số C (Log)"].get()
        gamma = self.params["Gamma"].get()
        k_sp = int(self.params["Kích thước lọc (Spatial)"].get())
        if k_sp % 2 == 0: k_sp += 1
        sigma = self.params["Hệ số Sigma (Gauss)"].get()
        
        d0 = int(self.params["Tần số cắt (D0)"].get())
        n_ord = int(self.params["Bậc bộ lọc (n)"].get())
        
        k_morph = int(self.params["Kích thước Morph (Kernel)"].get())
        if k_morph % 2 == 0: k_morph += 1

        # ROUTER
        if method == "Negative image": img = trans.negative(img)
        elif method == "Biến đổi Log": img = trans.log_transform(img, c)
        elif method == "Biến đổi Gamma": img = trans.gamma_transform(img, gamma)
        elif method == "Làm trơn ảnh (lọc trung bình)": img = ftr.mean_filter(img, k_sp)
        elif method == "Làm trơn ảnh (lọc Gauss)": img = ftr.gaussian_filter(img, k_sp, sigma)
        elif method == "Phát hiện biên (Sobel)": img = ftr.sobel_filter(img)
        
        # Frequency
        elif "Lọc Ideal" in method: 
            t = 'ideal_lp' if 'Lowpass' in method else 'ideal_hp'
            img = freq.apply_filter(img, d0, t)
        elif "Lọc Butterworth" in method:
            t = 'butter_lp' if 'Lowpass' in method else 'butter_hp'
            img = freq.apply_filter(img, d0, t, n_ord)
        elif "Lọc Gaussian" in method:
            t = 'gauss_lp' if 'Lowpass' in method else 'gauss_hp'
            img = freq.apply_filter(img, d0, t)

        # Morphology (Tuần 4)
        elif method == "Erosion (Co)": img = morph.erode(img, k_morph)
        elif method == "Dilation (Giãn)": img = morph.dilate(img, k_morph)
        elif method == "Opening (Mở - Xóa nhiễu)": img = morph.opening(img, k_morph)
        elif method == "Closing (Đóng - Lấp lỗ)": img = morph.closing(img, k_morph)
        elif method == "Boundary Extraction (Trích biên)": img = morph.boundary_extraction(img, k_morph)
        elif method == "Morphological Gradient": img = morph.gradient(img, k_morph)

        self.processed_image = img
        self.display_image(img, self.right_label)

    # --- CÁC HÀM HW3 (TẦN SỐ) ---
    def run_hw3_1(self):
        if self.original_image is None: return
        img, t = freq.apply_filter_sequence(self.original_image, 25)
        self.display_image(img, self.right_label)
        self.lbl_time1.config(text=f"Time: {t:.2f} ms")

    def run_hw3_2(self):
        if self.original_image is None: return
        img, t = freq.apply_multi_pass(self.original_image, 30, self.passes_var.get())
        self.display_image(img, self.right_label)

    def run_benchmark(self):
        if self.original_image is None: return
        ts, tf = freq.benchmark_spatial_vs_freq(self.original_image)
        self.lbl_bench.config(text=f"Spatial: {ts:.1f}ms | Freq: {tf:.1f}ms")

    # --- CÁC HÀM HW4 (HÌNH THÁI HỌC) ---
    def run_fill_holes(self):
        """
        Chỉ hiển thị kết quả lấp lỗ sang phải, KHÔNG cập nhật đè ảnh gốc bên trái.
        """
        if self.original_image is None: return
        img_filled = morph.fill_holes(self.original_image)
        self.processed_image = img_filled
        self.display_image(img_filled, self.right_label)

    def run_count_objects(self):
        """
        Đếm số lượng (Đã tách dính + Lấp lỗ bên trong hàm analyze_objects_with_filling).
        Hiển thị danh sách dọc (Newline).
        """
        if self.original_image is None: return
        img_res, count, areas = morph.analyze_objects_with_filling(self.original_image)
        self.processed_image = img_res
        self.display_image(img_res, self.right_label)
        
        # Format danh sách diện tích theo hàng dọc
        max_lines = 15
        area_list = areas[:max_lines]
        area_str = "\n".join(area_list)
        
        if len(areas) > max_lines:
            area_str += f"\n... và {len(areas) - max_lines} đối tượng khác"
        
        self.lbl_count.config(text=f"TỔNG SỐ LƯỢNG: {count}\n-------------------\n{area_str}")

    def run_fingerprint_clean(self):
        if self.original_image is None: return
        k = int(self.params["Kích thước Morph (Kernel)"].get())
        if k % 2 == 0: k += 1
        img_clean = morph.clean_fingerprint(self.original_image, kernel_size=k)
        self.processed_image = img_clean
        self.display_image(img_clean, self.right_label)

    def save_image(self):
        if self.processed_image is not None:
            fp = filedialog.asksaveasfilename(defaultextension=".png")
            if fp: cv2.imwrite(fp, self.processed_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()