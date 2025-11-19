✓ Clip limit: 2.0
✓ Tile grid: 8×8
✓ Áp dụng trên LAB color space (L channel)
```

**Giải thích**:
- **Clip limit = 2.0**: Hệ số giới hạn contrast, càng cao contrast càng mạnh
- **Tile grid = 8×8**: Chia ảnh thành lưới 8×8 blocks để xử lý cục bộ
- **LAB L channel**: Chỉ xử lý kênh độ sáng, giữ nguyên màu sắc

---

### **1.2. Công thức tính Clip Point (β)**

**Công thức 1:**
```
β = (M/N) × (1 + (α/100) × S_max)
```

**Định nghĩa các thành phần:**

| **Ký hiệu** | **Tên gọi** | **Ý nghĩa** | **Giá trị ví dụ** |
|-------------|-------------|-------------|-------------------|
| **β** | Clip Point | Ngưỡng cắt histogram | Tính được |
| **M** | Block pixels | Tổng số pixel trong 1 block | 64 (cho 8×8) |
| **N** | Gray levels | Số mức xám | 256 (ảnh 8-bit) |
| **α** | Clip factor | Hệ số điều chỉnh | 2.0 × 100 = 200 |
| **S_max** | Max slope | Độ dốc tối đa của CDF | 4 (thường dùng) |

---

### **1.3. Ví dụ tính toán cụ thể**

**Bài toán**: Tính β cho block 8×8 với clip limit = 2.0
```
Cho:
- Block size: 8×8 → M = 64 pixels
- Gray levels: N = 256
- Clip limit: 2.0 → α = 2.0 × 100 = 200
- S_max = 4

Tính:
β = (M/N) × (1 + (α/100) × S_max)
  = (64/256) × (1 + (200/100) × 4)
  = 0.25 × (1 + 2 × 4)
  = 0.25 × (1 + 8)
  = 0.25 × 9
  = 2.25
```

**Ý nghĩa β = 2.25**:
- Histogram không được vượt quá 2.25 × (M/N) pixels
- Tức là: 2.25 × 0.25 × 64 = **36 pixels**
- Nếu histogram[i] > 36 → cắt bớt xuống còn 36

---

### **1.4. Minh họa Histogram Clipping**
```
TRƯỚC KHI CẮT:
Histogram:
Mức xám: [50    51    52    53    54]
Pixels:  [10    25    50    30    15]
                       ↑
                  Vượt quá β=36!

SAU KHI CẮT:
Histogram:
Mức xám: [50    51    52    53    54]
Pixels:  [10    25    36    30    15]
                       ↑
                 Cắt ở β=36

Phần bị cắt: 50 - 36 = 14 pixels
```

---

### **1.5. Công thức Transform - Output Grayscale**

**Công thức 2:**
```
gr_out = (gr_max - gr_min) × CDF(gr_in) + gr_min
```

**Định nghĩa các thành phần:**

| **Ký hiệu** | **Tên gọi** | **Ý nghĩa** |
|-------------|-------------|-------------|
| **gr_out** | Output gray | Giá trị xám sau transform |
| **gr_in** | Input gray | Giá trị xám đầu vào |
| **gr_max** | Max gray | Giá trị xám tối đa = 255 |
| **gr_min** | Min gray | Giá trị xám tối thiểu = 0 |
| **CDF(gr_in)** | Cumulative Distribution | Hàm phân phối tích lũy [0,1] |

---

### **1.6. Ví dụ Transform**

**Bài toán**: Transform giá trị xám 100
```
Giả sử histogram đã được clipping và normalize:

Bước 1: Tính CDF
Histogram (sau clip): [5, 10, 36, 25, 20, 15, ...]
CDF:                  [5, 15, 51, 76, 96, 111, ...]

Tổng pixels = 130
CDF_normalized(100) = 76/130 = 0.585

Bước 2: Transform
gr_out = (255 - 0) × 0.585 + 0
       = 255 × 0.585
       = 149.175
       ≈ 149

Kết quả: Giá trị 100 → 149
```

---

### **1.7. Quy trình CLAHE hoàn chỉnh**
```
INPUT: Ảnh fundus RGB
         ↓
┌────────────────────────┐
│ 1. Chuyển sang LAB     │
│    Lấy L channel       │
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 2. Chia thành 8×8 grid │
│    (tile grid)         │
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 3. Với mỗi block:      │
│    • Tính histogram    │
│    • Tính β            │
│    • Clip histogram    │
│    • Phân phối lại     │
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 4. Tính CDF normalize  │
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 5. Transform pixels    │
│    gr_out = f(CDF)     │
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 6. Bilinear interp     │
│    (làm mịn ranh giới) │
└────────────────────────┘
         ↓
OUTPUT: L channel enhanced
```

---

## **PHẦN 2: ADAPTIVE GABOR FILTER VỚI CHAOTIC MAP**

---

### **2.1. Chebyshev Chaotic Map**

**Công thức:**
```
x_{t+1} = cos(n · arccos(x_t))
```

**Định nghĩa:**

| **Ký hiệu** | **Ý nghĩa** | **Giá trị** |
|-------------|-------------|-------------|
| **x_t** | Giá trị chaotic tại thời điểm t | [-1, 1] |
| **x_{t+1}** | Giá trị tiếp theo | [-1, 1] |
| **n** | Bậc của Chebyshev polynomial | Thường n=2 hoặc 3 |

**Đặc điểm:**
- ✓ Tạo chuỗi số **giả ngẫu nhiên** (pseudo-random)
- ✓ Tính chất **chaotic** (hỗn loạn): nhạy cảm với điều kiện ban đầu
- ✓ Phân bố đều trong [-1, 1]

---

### **2.2. Ví dụ Chebyshev Map**
```
Cho: n = 2, x_0 = 0.5

Bước 1: t=0
x_1 = cos(2 · arccos(0.5))
    = cos(2 · 60°)
    = cos(120°)
    = -0.5

Bước 2: t=1
x_2 = cos(2 · arccos(-0.5))
    = cos(2 · 120°)
    = cos(240°)
    = -0.5

Bước 3: t=2 (với n=3 để thay đổi)
x_3 = cos(3 · arccos(-0.5))
    = cos(3 · 120°)
    = cos(360°)
    = 1.0

Chuỗi: 0.5 → -0.5 → -0.5 → 1.0 → ...
       ↑ Dao động "hỗn loạn"
```

---

### **2.3. Adaptive Gabor Kernel**

**Công thức chính:**
```
g(x,y; θ,λ,σ,γ) = exp(-(x'² + γ²y'²)/(2σ²)) · cos(2πx'/λ + ψ)
```

**Với chaotic perturbation:**
```
θ' = θ + x_t · 0.1  (góc quay được nhiễu loạn)
```

---

### **2.4. Định nghĩa các thành phần Gabor**

| **Ký hiệu** | **Tên gọi** | **Ý nghĩa** | **Giá trị paper** |
|-------------|-------------|-------------|-------------------|
| **x, y** | Tọa độ | Vị trí pixel trong kernel | - |
| **x', y'** | Tọa độ xoay | Sau khi xoay góc θ | Tính từ x,y |
| **θ** | Orientation | Góc hướng của filter | π·i/8 (8 hướng) |
| **λ** | Wavelength | Bước sóng của sinusoid | 10.0 |
| **σ** | Std deviation | Độ rộng Gaussian envelope | 3.0 |
| **γ** | Aspect ratio | Tỷ lệ không gian (ellipticity) | 0.5 |
| **ψ** | Phase offset | Độ lệch pha | 0 hoặc π/2 |

---

### **2.5. Công thức tọa độ xoay**
```
x' = x·cos(θ) + y·sin(θ)
y' = -x·sin(θ) + y·cos(θ)
```

**Ý nghĩa**: Xoay hệ tọa độ để filter có thể detect edge theo nhiều hướng khác nhau.

**Ví dụ với θ = 45° (π/4)**:
```
Cho điểm (x,y) = (1, 0)

x' = 1·cos(45°) + 0·sin(45°)
   = 1·0.707 + 0
   = 0.707

y' = -1·sin(45°) + 0·cos(45°)
   = -0.707 + 0
   = -0.707

Điểm mới: (0.707, -0.707)
```

---

### **2.6. Tham số thực tế**

**Từ slide:**
```
Kernel size: 31×31 pixels
σ = 3.0  → Gaussian envelope rộng vừa phải
λ = 10.0 → Bước sóng lớn (detect texture thô)
γ = 0.5  → Hình ellipse (không tròn)
8 orientations: θ_i = π·i/8, i=0..7
```

**Minh họa 8 hướng:**
```
      θ=0°         θ=45°        θ=90°       θ=135°
       |            ╱            —            ╲
       |          ╱                            ╲
       
    θ=180°       θ=225°       θ=270°      θ=315°
       |            ╲            —            ╱
       |              ╲                    ╱
```

---

### **2.7. Ví dụ tính Gabor value**

**Bài toán**: Tính g(0, 5) với:
- θ = 0° (hướng dọc)
- λ = 10.0
- σ = 3.0
- γ = 0.5
- ψ = 0
```
Bước 1: Tính x', y'
x' = 0·cos(0) + 5·sin(0) = 0
y' = -0·sin(0) + 5·cos(0) = 5

Bước 2: Tính Gaussian part
exp(-(x'² + γ²y'²)/(2σ²))
= exp(-(0 + 0.25×25)/(2×9))
= exp(-6.25/18)
= exp(-0.347)
= 0.707

Bước 3: Tính Cosine part
cos(2πx'/λ + ψ)
= cos(2π×0/10 + 0)
= cos(0)
= 1.0

Bước 4: Kết hợp
g(0,5) = 0.707 × 1.0 = 0.707
```

---

### **2.8. Chaotic Perturbation**

**Công thức:**
```
θ' = θ + x_t · 0.1
```

**Ý nghĩa**: 
- Thêm nhiễu loạn nhỏ vào góc quay
- Giúp filter "khám phá" các hướng lân cận
- Tăng khả năng detect edge ở góc không chuẩn

**Ví dụ:**
```
Cho: θ = 45°, x_t = 0.3 (từ Chebyshev map)

θ' = 45° + 0.3 × 0.1 × (180°/π)
   = 45° + 0.03 × 57.3°
   = 45° + 1.72°
   = 46.72°

→ Góc bị "lệch" nhẹ 1.72° mỗi lần
→ Giúp detect edge không hoàn toàn thẳng
```

---

### **2.9. Quy trình Adaptive Gabor**
```
INPUT: Ảnh grayscale sau CLAHE
         ↓
┌────────────────────────────┐
│ 1. Khởi tạo Chebyshev map  │
│    x_0 = 0.5 (random seed) │
└────────────────────────────┘
         ↓
┌────────────────────────────┐
│ 2. For mỗi orientation:    │
│    θ_i = π·i/8, i=0..7     │
└────────────────────────────┘
         ↓
┌────────────────────────────┐
│ 3. Chaotic perturbation:   │
│    θ' = θ + x_t · 0.1      │
│    Cập nhật x_t            │
└────────────────────────────┘
         ↓
┌────────────────────────────┐
│ 4. Tạo Gabor kernel 31×31: │
│    g(x,y; θ',λ,σ,γ)        │
└────────────────────────────┘
         ↓
┌────────────────────────────┐
│ 5. Convolution với ảnh     │
│    filtered = img ⊗ kernel │
└────────────────────────────┘
         ↓
┌────────────────────────────┐
│ 6. Tổng hợp 8 hướng:       │
│    output = Σ|filtered_i|  │
└────────────────────────────┘
         ↓
OUTPUT: Ảnh sau Gabor filtering
```

---

## **PHẦN 3: TẠI SAO CẦN CHAOTIC MAP?**

---

### **3.1. So sánh Fixed vs Adaptive**

**Fixed Gabor (θ cố định):**
```
θ = [0°, 45°, 90°, 135°, ...]

      |     ╱     —     ╲
      |   ╱             ╲
      
Chỉ detect 8 hướng chính xác
→ Bỏ sót edge ở góc lệch (ví dụ 47°)
```

**Adaptive Gabor (θ có nhiễu loạn):**
```
θ' = [1.2°, 46.8°, 91.5°, 133.7°, ...]
         ↑      ↑      ↑       ↑
    Dao động xung quanh giá trị chuẩn

      | |   ╱╱    ══     ╲╲
      |||  ╱╱             ╲╲
      
Detect được nhiều hướng hơn
→ Bắt được edge lệch ✅
```

---

### **3.2. Lợi ích của Chaotic Map**
```
✓ Tính giả ngẫu nhiên (pseudo-random)
  → Tránh pattern lặp lại
  
✓ Phân bố đều (uniform distribution)
  → Khám phá đều khắp không gian góc
  
✓ Nhạy cảm điều kiện ban đầu
  → Mỗi ảnh có seed khác nhau
  
✓ Tính toán nhanh
  → Chỉ cần cos và arccos
```

---

### **3.3. Kết quả thực nghiệm**

**Từ paper:**
```
Accuracy improvement (với Adaptive Gabor):
- Dataset 1: 85.54% → 98.01% (+12.47%)
- Dataset 2: 91.10% → 97.99% (+6.89%)
- Dataset 3: 87.10% → 99.10% (+12.00%)

Segmentation IoU:
- Without Gabor: 0.2915
- With Adaptive Gabor + CLAHE: 0.8272
  (+184% improvement!)