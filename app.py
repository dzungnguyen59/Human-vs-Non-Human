import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

# 1. Cấu hình tiêu đề trang (hiển thị trên tab trình duyệt)
st.set_page_config(page_title="Human Detection Web", layout="centered")

# 2. Load model (Sử dụng cache để tối ưu tốc độ)
@st.cache_resource
def load_my_model():
    # Đảm bảo file .keras nằm cùng thư mục với file app.py này
    model = tf.keras.models.load_model("my_resnet50_model.keras",compile=False)
    return model

model = load_my_model()
class_names = ['Human', 'Non-Human'] # Thay đổi thứ tự nếu cần

# 3. Giao diện chính
st.title("Nhận diện Human vs Non-Human")
st.write("Được tạo bởi: uvuvwevwevwe onyetenyevwe ugwemubwem ossas.")

# 4. Khu vực upload ảnh
test_image = st.file_uploader("Chọn tài liệu đê", type=["jpg", "png", "jpeg"])

if test_image is not None:
    # Hiển thị ảnh đã upload
    img = Image.open(test_image)
    
    # Tạo 2 cột để giao diện cân đối: cột trái hiện ảnh, cột phải hiện kết quả
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption='Ảnh đã tải lên', use_container_width=True)
    
    with col2:
        # Nhấn nút để bắt đầu dự đoán
        if st.button("Bắt đầu nhận diện"):
            with st.spinner('Đang phân tích...'):
                # Tiền xử lý ảnh
                img_resized = img.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_ready = preprocess_input(img_array)
                
                # Dự đoán
                predictions = model.predict(img_ready)
                result_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                # Hiển thị kết quả bằng Alert box
                label = class_names[result_idx]
                if label == 'Human':
                    st.success(f"KẾT QUẢ: **{label}**")
                else:
                    st.error(f"KẾT QUẢ: **{label}**")
                
                st.write(f"Độ tin cậy: **{confidence:.2f}%**")
                
                # Hiển thị biểu đồ cột nhỏ cho xác suất
                st.progress(int(confidence))