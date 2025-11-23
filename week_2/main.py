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
# Đảm bảo bạn đã tạo file frequency.py cùng thư mục
import frequency as freq  

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng xử lý ảnh - Digital Image Processing (Week 2 & 3)")
        # 1. Tăng chiều rộng cửa sổ để chứa thanh điều khiển to hơn
        self.root.geometry("1780x850") 
        self.root.configure(bg="#e0e0e0")

        # --- Biến lưu ảnh ---
        self.original_image = None
        self.processed_image = None

        # --- Khu vực hiển thị ảnh (Trái / Phải) ---
        self.left_label = tk.Label(self.root, bg="#ccc", text="Ảnh gốc")
        self.left_label.place(x=30, y=30, width=500, height=500)

        self.right_label = tk.Label(self.root, bg="#ccc", text="Ảnh sau xử lý")
        self.right_label.place(x=560, y=30, width=500, height=500)

        # --- Menu chọn phép biến đổi ---
        tk.Label(self.root, text="Chọn loại biến đổi", bg="#e0e0e0", font=("Arial", 11, "bold")).place(x=1080, y=20)
        self.method_var = tk.StringVar(value="Negative image")
        self.method_box = ttk.Combobox(
            self.root,
            textvariable=self.method_var,
            values=[
                "Negative image",
                "Biến đổi Log",
                "Biến đổi Piecewise-Linear",
                "Biến đổi Gamma",
                "Làm trơn ảnh (lọc trung bình)",
                "Làm trơn ảnh (lọc Gauss)",
                "Làm trơn ảnh (lọc trung vị)",
                "Cân bằng sáng dùng Histogram",
                "Phát hiện biên (Sobel)",      
                "Phát hiện biên (Laplacian)",
                "--- MIỀN TẦN SỐ (HW3) ---",
                "Lọc Ideal Lowpass (Làm mờ)",
                "Lọc Ideal Highpass (Làm nét)",
                "Lọc Butterworth Lowpass",
                "Lọc Butterworth Highpass",
                "Lọc Gaussian Lowpass",
                "Lọc Gaussian Highpass"
            ],
            state="readonly",
            width=35,
        )
        self.method_box.place(x=1080, y=45)
        self.method_box.bind("<<ComboboxSelected>>", lambda e: self.update_image())

        # --- Khung công cụ bên phải (CÓ THANH CUỘN) ---
        # 2. Mở rộng chiều rộng khung chứa lên 650 để chia 2 cột
        self.right_container = tk.Frame(self.root, bg="#f8f8f8", relief="ridge", bd=3)
        self.right_container.place(x=1080, y=90, width=680, height=720) # Tăng width lên 680

        self.canvas = tk.Canvas(self.right_container, bg="#f8f8f8", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.right_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f8f8f8")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Tạo các thanh trượt và nút bấm
        self.create_sliders(self.scrollable_frame)

        # --- Các nút điều khiển chính (Phía dưới ảnh) ---
        tk.Button(self.root, text="Chọn ảnh", command=self.open_image, font=("Arial", 10, "bold"), bg="#ddd").place(x=30, y=550, width=120, height=40)
        tk.Button(self.root, text="Cập nhật", command=self.update_image, font=("Arial", 10, "bold"), bg="#87CEEB").place(x=180, y=550, width=120, height=40)
        tk.Button(self.root, text="Lưu ra file", command=self.save_image, font=("Arial", 10, "bold"), bg="#90EE90").place(x=330, y=550, width=120, height=40)

    # ---------- Tạo nhóm slider & Buttons ----------
    def create_sliders(self, parent):
        self.params = {}
        
        # 3. TẠO 2 CỘT (FRAMES) ĐỂ CHIA BỐ CỤC
        col1 = tk.Frame(parent, bg=parent["bg"]) # Cột trái (Tuần 2)
        col1.pack(side="left", fill="both", expand=True, padx=5, anchor="n")
        
        col2 = tk.Frame(parent, bg=parent["bg"]) # Cột phải (Tuần 3)
        col2.pack(side="left", fill="both", expand=True, padx=5, anchor="n")

        def add_slider(frame_parent, text, from_, to, init, step=1):
            frame = tk.Frame(frame_parent, bg=frame_parent["bg"])
            frame.pack(pady=2, fill="x")
            tk.Label(frame, text=text, bg=frame_parent["bg"], anchor="w", font=("Arial", 9)).pack(padx=5)
            var = tk.DoubleVar(value=init)
            tk.Scale(frame, from_=from_, to=to, orient="horizontal", resolution=step, length=280, variable=var,
                     bg=frame_parent["bg"], troughcolor="#cfcfcf").pack(padx=5)
            self.params[text] = var

        # === CỘT 1: CÁC BÀI TẬP TUẦN 2 ===
        #tk.Label(col1, text="--- TUẦN 2 (SPATIAL) ---", bg="#f8f8f8", fg="blue", font=("Arial", 10, "bold")).pack(pady=5)

        s1 = tk.LabelFrame(col1, text="Biến đổi mức xám", bg="#C2E0F2", font=("Arial", 9, "bold"))
        s1.pack(fill="x", padx=5, pady=5)
        add_slider(s1, "Hệ số C (Log)", 1, 100, 40)
        add_slider(s1, "Gamma", 0.1, 5.0, 1.0, step=0.1)

        s2 = tk.LabelFrame(col1, text="Piecewise Linear", bg="#E6B8F2", font=("Arial", 9, "bold"))
        s2.pack(fill="x", padx=5, pady=5)
        add_slider(s2, "Hệ số Cao (r2)", 0, 255, 140)
        add_slider(s2, "Hệ số Thấp (r1)", 0, 255, 70)

        s3 = tk.LabelFrame(col1, text="Lọc Không Gian (Spatial)", bg="#F7CAAC", font=("Arial", 9, "bold"))
        s3.pack(fill="x", padx=5, pady=5)
        add_slider(s3, "Kích thước lọc (Mean/Med/Gauss)", 1, 31, 3, step=2)
        add_slider(s3, "Hệ số Sigma (Gauss)", 0, 10, 1, step=0.5)

        # === CỘT 2: CÁC BÀI TẬP TUẦN 3 ===
        #tk.Label(col2, text="--- TUẦN 3 (FREQUENCY) ---", bg="#f8f8f8", fg="red", font=("Arial", 10, "bold")).pack(pady=5)
        
        s4 = tk.LabelFrame(col2, text="Tham số lọc Tần Số", bg="#FFFACD", font=("Arial", 9, "bold"))
        s4.pack(fill="x", padx=5, pady=5)
        add_slider(s4, "Tần số cắt (D0)", 1, 200, 30)
        add_slider(s4, "Bậc bộ lọc (n - Butterworth)", 1, 10, 2)

        # HW3-1: Lowpass -> Highpass
        hw1 = tk.LabelFrame(col2, text="Gaussian LP + HP", bg="#E0F7FA")
        hw1.pack(fill="x", padx=5, pady=5)
        tk.Button(hw1, text="Chạy HW3-1 (D0=25)", command=self.run_hw3_1, bg="white").pack(fill="x", padx=5, pady=2)
        self.lbl_time1 = tk.Label(hw1, text="Time: -- ms", bg="#E0F7FA", font=("Arial", 8))
        self.lbl_time1.pack()

        # HW3-2: Highpass n lần
        hw2 = tk.LabelFrame(col2, text="Highpass n lần", bg="#E0F7FA")
        hw2.pack(fill="x", padx=5, pady=5)
        self.passes_var = tk.IntVar(value=1)
        tk.Radiobutton(hw2, text="1 lần", variable=self.passes_var, value=1, bg="#E0F7FA").pack(anchor="w", padx=5)
        tk.Radiobutton(hw2, text="10 lần", variable=self.passes_var, value=10, bg="#E0F7FA").pack(anchor="w", padx=5)
        tk.Radiobutton(hw2, text="100 lần", variable=self.passes_var, value=100, bg="#E0F7FA").pack(anchor="w", padx=5)
        tk.Button(hw2, text="Chạy HW3-2 (D0=30)", command=self.run_hw3_2, bg="white").pack(fill="x", padx=5, pady=2)
        self.lbl_time2 = tk.Label(hw2, text="Time: -- ms", bg="#E0F7FA", font=("Arial", 8))
        self.lbl_time2.pack()

        # Benchmark
        tk.Button(col2, text="So sánh Spatial vs Freq", command=self.run_benchmark, bg="#FFCC80", font=("Arial", 9, "bold")).pack(fill="x", padx=5, pady=10)
        self.lbl_bench = tk.Label(col2, text="", bg="#f8f8f8", justify="left", fg="blue")
        self.lbl_bench.pack()

        # Khoảng trống dưới cùng
        tk.Frame(parent, height=50, bg="#f8f8f8").pack()

    # ---------- Hàm xử lý ảnh ----------
    def open_image(self):
        filepath = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if filepath:
            self.original_image = cv2.imread(filepath)
            if self.original_image is None: return
            self.display_image(self.original_image, self.left_label)
            self.processed_image = None
            self.right_label.configure(image='')

    def display_image(self, img, label_widget):
        if img is None: return
        # Convert BGR -> RGB
        if len(img.shape) == 2: # Ảnh xám
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((500, 500)) 
        imgtk = ImageTk.PhotoImage(img_pil)
        label_widget.imgtk = imgtk 
        label_widget.configure(image=imgtk)

    def update_image(self):
        if self.original_image is None: return

        method = self.method_var.get()
        img = self.original_image.copy()

        # Lấy tham số chung
        c = self.params["Hệ số C (Log)"].get()
        gamma = self.params["Gamma"].get()
        high = int(self.params["Hệ số Cao (r2)"].get())
        low = int(self.params["Hệ số Thấp (r1)"].get())
        if low >= high: low = high - 1 if high > 0 else 0

        ksize = int(self.params["Kích thước lọc (Mean/Med/Gauss)"].get())
        if ksize % 2 == 0: ksize += 1
        sigma = self.params["Hệ số Sigma (Gauss)"].get()
        
        # Tham số tần số
        d0 = int(self.params["Tần số cắt (D0)"].get())
        n_order = int(self.params["Bậc bộ lọc (n - Butterworth)"].get())

        # --- Router xử lý ---
        if method == "Negative image": img = trans.negative(img)
        elif method == "Biến đổi Log": img = trans.log_transform(img, c)
        elif method == "Biến đổi Piecewise-Linear": img = trans.piecewise_linear(img, r1=low, r2=high)
        elif method == "Biến đổi Gamma": img = trans.gamma_transform(img, gamma)
        
        elif method == "Làm trơn ảnh (lọc trung bình)": img = ftr.mean_filter(img, ksize)
        elif method == "Làm trơn ảnh (lọc Gauss)": img = ftr.gaussian_filter(img, ksize, sigma)
        elif method == "Làm trơn ảnh (lọc trung vị)": img = ftr.median_filter(img, ksize)
        elif method == "Cân bằng sáng dùng Histogram": img = hist.equalize_histogram(img)
        elif method == "Phát hiện biên (Sobel)": img = ftr.sobel_filter(img)
        elif method == "Phát hiện biên (Laplacian)": img = ftr.laplacian_filter(img)

        # --- XỬ LÝ MIỀN TẦN SỐ (GỌI FREQUENCY.PY) ---
        elif method == "Lọc Ideal Lowpass (Làm mờ)": 
            img = freq.apply_filter(img, d0, 'ideal_lp')
        elif method == "Lọc Ideal Highpass (Làm nét)": 
            img = freq.apply_filter(img, d0, 'ideal_hp')
        elif method == "Lọc Butterworth Lowpass": 
            img = freq.apply_filter(img, d0, 'butter_lp', n_order)
        elif method == "Lọc Butterworth Highpass": 
            img = freq.apply_filter(img, d0, 'butter_hp', n_order)
        elif method == "Lọc Gaussian Lowpass": 
            img = freq.apply_filter(img, d0, 'gauss_lp')
        elif method == "Lọc Gaussian Highpass": 
            img = freq.apply_filter(img, d0, 'gauss_hp')

        self.processed_image = img
        self.display_image(self.processed_image, self.right_label)

    # --- CÁC HÀM CHẠY BÀI TẬP HW3 ---
    def run_hw3_1(self):
        if self.original_image is None: return
        # Bài 1: Lowpass -> Highpass (D0=25)
        img_out, t_ms = freq.apply_filter_sequence(self.original_image, d0=25)
        self.processed_image = img_out
        self.display_image(img_out, self.right_label)
        self.lbl_time1.config(text=f"Time: {t_ms:.2f} ms")

    def run_hw3_2(self):
        if self.original_image is None: return
        # Bài 2: Highpass n lần (D0=30)
        n = self.passes_var.get()
        img_out, t_ms = freq.apply_multi_pass(self.original_image, d0=30, passes=n)
        self.processed_image = img_out
        self.display_image(img_out, self.right_label)
        self.lbl_time2.config(text=f"Time: {t_ms:.2f} ms")

    def run_benchmark(self):
        if self.original_image is None: return
        # So sánh tốc độ
        ts, tf = freq.benchmark_spatial_vs_freq(self.original_image, ksize=15)
        msg = f"Spatial (Conv): {ts:.2f} ms\nFrequency (FFT): {tf:.2f} ms"
        if tf < ts: msg += "\n=> FFT nhanh hơn!"
        else: msg += "\n=> Spatial nhanh hơn!"
        self.lbl_bench.config(text=msg)

    def save_image(self):
        if self.processed_image is None: return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if filepath:
            cv2.imwrite(filepath, self.processed_image)
            print(f"Đã lưu: {filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()