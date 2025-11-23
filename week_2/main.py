import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

# Import các module xử lý (đảm bảo bạn đã có các file này cùng thư mục)
import transformations as trans
import filters as ftr
import histogram as hist

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng xử lý ảnh - Digital Image Processing")
        self.root.geometry("1450x780") # Tăng chiều cao cửa sổ lên chút cho thoải mái
        self.root.configure(bg="#e0e0e0")

        # --- Biến lưu ảnh ---
        self.original_image = None
        self.processed_image = None

        # --- Khu vực hiển thị ảnh (Trái / Phải) ---
        self.left_label = tk.Label(self.root, bg="#ccc", text="Ảnh gốc")
        self.left_label.place(x=50, y=50, width=480, height=480)

        self.right_label = tk.Label(self.root, bg="#ccc", text="Ảnh sau xử lý")
        self.right_label.place(x=580, y=50, width=480, height=480)

        # --- Menu chọn phép biến đổi ---
        tk.Label(self.root, text="Chọn loại biến đổi", bg="#e0e0e0", font=("Arial", 11, "bold")).place(x=1100, y=20)
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
                "Phát hiện biên (Sobel)",      # High-pass
                "Phát hiện biên (Laplacian)"   # High-pass
            ],
            state="readonly",
            width=35,
        )
        self.method_box.place(x=1100, y=45)
        # Tự động cập nhật khi chọn menu khác
        self.method_box.bind("<<ComboboxSelected>>", lambda e: self.update_image())

        # --- Khung công cụ bên phải (CÓ THANH CUỘN) ---
        # 1. Tạo Frame chứa chính (Container) - Tăng width và height
        self.right_container = tk.Frame(self.root, bg="#f8f8f8", relief="ridge", bd=3)
        self.right_container.place(x=1100, y=90, width=330, height=630)

        # 2. Tạo Canvas và Scrollbar
        self.canvas = tk.Canvas(self.right_container, bg="#f8f8f8", highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.right_container, orient="vertical", command=self.canvas.yview)
        
        # 3. Frame nội dung bên trong Canvas
        self.scrollable_frame = tk.Frame(self.canvas, bg="#f8f8f8")

        # 4. Cấu hình sự kiện cuộn
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Tạo window bên trong canvas
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Link scrollbar với canvas
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # 5. Pack lên giao diện
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Tiêu đề trong frame cuộn
        tk.Label(self.scrollable_frame, text="Công cụ biến đổi (Tham số)", bg="#f8f8f8", font=("Arial", 11, "bold")).pack(pady=10)

        # --- Tạo các thanh trượt vào trong scrollable_frame ---
        self.create_sliders(self.scrollable_frame)

        # --- QUAN TRỌNG: Cập nhật lại vùng cuộn sau khi thêm widget ---
        self.scrollable_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        # --- Các nút điều khiển (Phía dưới ảnh gốc) ---
        # Dời xuống y=580 để không bị sát quá
        tk.Button(self.root, text="Chọn ảnh", command=self.open_image, font=("Arial", 10, "bold"), bg="#ddd").place(x=50, y=580, width=120, height=40)
        tk.Button(self.root, text="Cập nhật", command=self.update_image, font=("Arial", 10, "bold"), bg="#87CEEB").place(x=200, y=580, width=120, height=40)
        tk.Button(self.root, text="Lưu ra file", command=self.save_image, font=("Arial", 10, "bold"), bg="#90EE90").place(x=350, y=580, width=120, height=40)

    # ---------- Tạo nhóm slider ----------
    def create_sliders(self, parent):
        self.params = {}
        
        # Helper function để vẽ slider gọn gàng
        def add_slider(frame_parent, text, from_, to, init, step=0.1):
            frame = tk.Frame(frame_parent, bg=frame_parent["bg"])
            frame.pack(pady=2, fill="x")
            tk.Label(frame, text=text, bg=frame_parent["bg"], anchor="w", font=("Arial", 9)).pack(padx=5)
            var = tk.DoubleVar(value=init)
            tk.Scale(frame, from_=from_, to=to, orient="horizontal", resolution=step, length=260, variable=var,
                     bg=frame_parent["bg"], troughcolor="#cfcfcf", width=12).pack(padx=5)
            self.params[text] = var

        # Log
        section1 = tk.LabelFrame(parent, text="Biến đổi Log", bg="#C2E0F2", font=("Arial", 9, "bold"))
        section1.pack(fill="x", padx=5, pady=5)
        add_slider(section1, "Hệ số C (Log)", 1, 100, 40, step=1)

        # Piecewise
        section2 = tk.LabelFrame(parent, text="Biến đổi Piecewise-Linear", bg="#E6B8F2", font=("Arial", 9, "bold"))
        section2.pack(fill="x", padx=5, pady=5)
        add_slider(section2, "Hệ số Cao (r2)", 0, 255, 140, step=1)
        add_slider(section2, "Hệ số Thấp (r1)", 0, 255, 70, step=1)

        # Gamma
        section3 = tk.LabelFrame(parent, text="Biến đổi Gamma", bg="#F7CAAC", font=("Arial", 9, "bold"))
        section3.pack(fill="x", padx=5, pady=5)
        add_slider(section3, "Gamma", 0.1, 5.0, 1.0)

        # Mean
        section4 = tk.LabelFrame(parent, text="Làm trơn ảnh (lọc trung bình)", bg="#F2C2D4", font=("Arial", 9, "bold"))
        section4.pack(fill="x", padx=5, pady=5)
        add_slider(section4, "Kích thước lọc (Mean)", 1, 15, 3, step=2)

        # Gauss
        section5 = tk.LabelFrame(parent, text="Làm trơn ảnh (lọc Gauss)", bg="#FAD4A7", font=("Arial", 9, "bold"))
        section5.pack(fill="x", padx=5, pady=5)
        add_slider(section5, "Kích thước lọc (Gauss)", 1, 15, 3, step=2)
        add_slider(section5, "Hệ số Sigma", 0, 10, 1)

        # Median
        section6 = tk.LabelFrame(parent, text="Làm trơn ảnh (lọc trung vị)", bg="#F7A6D0", font=("Arial", 9, "bold"))
        section6.pack(fill="x", padx=5, pady=5)
        add_slider(section6, "Kích thước lọc (Median)", 1, 15, 3, step=2)
        
        # Histogram (Chỉ hiện text thông báo)
        section7 = tk.LabelFrame(parent, text="Cân bằng sáng dùng Histogram", bg="#B7E3B0", font=("Arial", 9, "bold"))
        section7.pack(fill="x", padx=5, pady=5)
        tk.Label(section7, text="* Tự động cân bằng (Không tham số)", bg="#B7E3B0", fg="#333", anchor="w").pack(padx=5, pady=5)

        # Khoảng trống dưới cùng để scroll được hết
        tk.Frame(parent, height=20, bg="#f8f8f8").pack()

    # ---------- Hàm xử lý ảnh ----------
    def open_image(self):
        filepath = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if filepath:
            # Đọc ảnh
            self.original_image = cv2.imread(filepath)
            if self.original_image is None:
                print("Lỗi: Không đọc được ảnh!")
                return
            
            # Reset hiển thị
            self.display_image(self.original_image, self.left_label)
            self.processed_image = None
            self.right_label.configure(image='')
            self.right_label.image = None

    def display_image(self, img, label_widget):
        if img is None: return
        # Convert BGR (OpenCV) to RGB (PIL)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Resize giữ tỷ lệ (thumbnail) để vừa khung 480x480
        img_pil.thumbnail((480, 480)) 
        imgtk = ImageTk.PhotoImage(img_pil)
        
        # Giữ tham chiếu để không bị garbage collection thu hồi
        label_widget.imgtk = imgtk 
        label_widget.configure(image=imgtk)

    def update_image(self):
        if self.original_image is None:
            return

        method = self.method_var.get()
        img = self.original_image.copy()

        # Lấy giá trị slider
        c = self.params["Hệ số C (Log)"].get()
        gamma = self.params["Gamma"].get()
        high = int(self.params["Hệ số Cao (r2)"].get())
        low = int(self.params["Hệ số Thấp (r1)"].get())
        
        # Logic bảo vệ: low luôn < high
        if low >= high:
            low = high - 1 if high > 0 else 0

        # Kích thước kernel phải là số lẻ
        mean_size = int(self.params["Kích thước lọc (Mean)"].get())
        if mean_size % 2 == 0: mean_size += 1
        
        gauss_size = int(self.params["Kích thước lọc (Gauss)"].get())
        if gauss_size % 2 == 0: gauss_size += 1
            
        sigma = self.params["Hệ số Sigma"].get()
        
        median_size = int(self.params["Kích thước lọc (Median)"].get())
        if median_size % 2 == 0: median_size += 1

        # --- Router xử lý ---
        if method == "Negative image":
            img = trans.negative(img)
            
        elif method == "Biến đổi Log":
            img = trans.log_transform(img, c)
            
        elif method == "Biến đổi Piecewise-Linear":
            img = trans.piecewise_linear(img, r1=low, r2=high)
            
        elif method == "Biến đổi Gamma":
            img = trans.gamma_transform(img, gamma)
            
        elif method == "Làm trơn ảnh (lọc trung bình)":
            img = ftr.mean_filter(img, mean_size)
            
        elif method == "Làm trơn ảnh (lọc Gauss)":
            img = ftr.gaussian_filter(img, gauss_size, sigma)
            
        elif method == "Làm trơn ảnh (lọc trung vị)":
            img = ftr.median_filter(img, median_size)
            
        elif method == "Cân bằng sáng dùng Histogram":
            img = hist.equalize_histogram(img)
            
        elif method == "Phát hiện biên (Sobel)":
            img = ftr.sobel_filter(img)
            
        elif method == "Phát hiện biên (Laplacian)":
            img = ftr.laplacian_filter(img)

        self.processed_image = img
        self.display_image(self.processed_image, self.right_label)

    def save_image(self):
        if self.processed_image is None:
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if filepath:
            cv2.imwrite(filepath, self.processed_image)
            print(f"Ảnh đã được lưu tại: {filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()