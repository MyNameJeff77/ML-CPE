# LAB07 - CNN Image Classification: Cat vs Dog

## 1. Cat vs Dog

**LAB07: Image Classification using Convolutional Neural Network (CNN)**

โครงงานนี้เป็นการสร้างโมเดล Convolutional Neural Network (CNN) สำหรับจำแนกรูปภาพออกเป็น 2 ประเภท ได้แก่

- Cat
- Dog

โดยใช้ภาษา Python และ TensorFlow/Keras ในการสร้างและฝึกสอนโมเดล

---

## 2. วัตถุประสงค์

1. ศึกษาหลักการทำงานของ Convolutional Neural Network (CNN)
2. เรียนรู้การเตรียมข้อมูลรูปภาพสำหรับ Machine Learning
3. ฝึกการแบ่ง Dataset ออกเป็น Training, Validation และ Testing
4. สร้างและฝึกสอนโมเดล CNN สำหรับจำแนกรูปภาพ
5. ประเมินประสิทธิภาพของโมเดลด้วย Accuracy, Classification Report และ Confusion Matrix
6. ศึกษาผลของการปรับ Hyperparameter เช่น Learning Rate และ Batch Size

---

## 3. Dataset

Dataset ที่ใช้ใน LAB07 คือ **Cat and Dog Dataset** จาก Kaggle

Dataset นี้ประกอบด้วยรูปภาพแมวและสุนัข และถูกจัดทำขึ้นสำหรับการนำไปใช้ในการฝึกโมเดล Deep Learning โดยมีโครงสร้างข้อมูลที่แบ่งเป็นหมวดหมู่ของแมวและสุนัข

แหล่งข้อมูล:

https://www.kaggle.com/datasets/tongpython/cat-and-dog

Dataset ระบุ License เป็น **CC0: Public Domain**

---

## 4. จำนวนข้อมูลที่ใช้

เพื่อให้สามารถฝึกโมเดลได้รวดเร็วและเหมาะสมกับเครื่องที่ใช้ในการทดลอง จึงเลือกใช้ข้อมูลจำนวนประมาณ 6,000 รูป

- Cat = 3,000 รูป
- Dog = 3,000 รูป
- รวมทั้งหมด = 6,000 รูป

กำหนดในโปรแกรมด้วย

```python
MAX_PER_CLASS = 3000

5. การแบ่ง Dataset

ข้อมูลทั้งหมดถูกแบ่งออกเป็น 3 ส่วน โดยใช้ Stratified Split เพื่อรักษาสัดส่วนของแต่ละ Class ให้ใกล้เคียงกัน

Dataset	จำนวนโดยประมาณ	สัดส่วน
Training	4,800	80%
Validation	600	10%
Testing	600	10%
Total	6,000	100%

การแบ่งข้อมูลแบบนี้ช่วยให้สามารถใช้ Training Set สำหรับฝึกโมเดล, Validation Set สำหรับติดตามผลระหว่างการฝึก และ Test Set สำหรับประเมินโมเดลกับข้อมูลที่โมเดลไม่เคยใช้ในการฝึก

6. โครงสร้างโปรเจกต์

LAB07/
│
├── PetImages/
│   ├── Cat/
│   └── Dog/
│
└── m-project/
    ├── main.py
    ├── cnn_model.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── split_data.py
    ├── evaluate.py
    │
    └── outputs/
        ├── cnn_model.keras
        ├── best_model.keras
        ├── history.json
        ├── classes.json
        ├── labels.npy
        ├── features.npy
        ├── X_train.npy
        ├── X_val.npy
        ├── X_test.npy
        ├── y_train.npy
        ├── y_val.npy
        ├── y_test.npy
        ├── confusion_matrix.png
        └── training_history.png

7. เทคโนโลยีที่ใช้
    Python
    TensorFlow
    Keras
    OpenCV
    NumPy
    Scikit-learn
    Matplotlib

8. การเตรียมข้อมูลภาพ

    รูปภาพที่นำเข้าสู่ระบบจะถูกปรับขนาดให้มีขนาดเท่ากันก่อนนำไปใช้กับ CNN

    กำหนดขนาดรูปภาพเป็น

        IMG_SIZE = 100

    ดังนั้นรูปภาพแต่ละภาพจะถูกปรับเป็นขนาด

        100 x 100 pixels

    นอกจากนี้ยังมีการแปลงค่าพิกเซลให้อยู่ในช่วง 0-1 ด้วยการ Rescaling

        layers.Rescaling(1.0 / 255)

 9. Data Augmentation

เพื่อช่วยเพิ่มความหลากหลายของข้อมูลและลดโอกาสการเกิด Overfitting โมเดลมีการทำ Data Augmentation ได้แก่

Random Horizontal Flip
Random Rotation
Random Zoom       

ตัวอย่างเช่น

layers.RandomFlip("horizontal")
layers.RandomRotation(0.05)
layers.RandomZoom(0.10)

การทำ Data Augmentation ช่วยให้โมเดลเรียนรู้ลักษณะของภาพที่หลากหลายมากขึ้น

10. โครงสร้าง CNN Model

โมเดลที่ใช้ประกอบด้วย Convolutional Layer หลายระดับ โดยมีโครงสร้างหลักดังนี้

Input Image
     │
     ▼
Rescaling
     │
     ▼
Data Augmentation
     │
     ▼
Conv2D 32
     │
Batch Normalization
     │
Max Pooling
     │
     ▼
Conv2D 64
     │
Batch Normalization
     │
Max Pooling
     │
     ▼
Conv2D 128
     │
Batch Normalization
     │
Max Pooling
     │
     ▼
Global Average Pooling
     │
     ▼
Dropout
     │
     ▼
Dense 64
     │
     ▼
Dropout
     │
     ▼
Sigmoid
     │
     ▼
Cat / Dog

11. Model Configuration

โมเดลใช้ Optimizer แบบ Adam

keras.optimizers.Adam(
    learning_rate=1e-3
)

สำหรับปัญหา Binary Classification ใช้ Loss Function เป็น

binary_crossentropy

และใช้

accuracy

เป็น Metric สำหรับวัดประสิทธิภาพของโมเดล

12. Training Configuration

กำหนดค่าการฝึกโมเดลดังนี้

EPOCHS = 50
BATCH_SIZE = 32

โมเดลจะสามารถฝึกได้สูงสุด 50 Epoch และใช้ข้อมูลครั้งละ 32 รูปต่อ Batch

นอกจากนี้ยังใช้ Callback เพื่อช่วยควบคุมการฝึกโมเดล ได้แก่

ReduceLROnPlateau
EarlyStopping
ModelCheckpoint

โดย ModelCheckpoint จะบันทึกโมเดลที่มีผล Validation ดีที่สุดไว้ในไฟล์

best_model.keras

13. ผลการทดลอง

จากการทดลองปรับค่า Learning Rate พบว่าการใช้

learning_rate = 1e-3

ให้ผลการฝึกที่ดีกว่าการใช้

learning_rate = 3e-4

ในการทดลองล่าสุด ค่า Training Accuracy สามารถเพิ่มขึ้นไปประมาณ 89% และ Validation Accuracy อยู่ประมาณ 84-85% โดยค่าจริงอาจแตกต่างกันเล็กน้อยในแต่ละรอบการฝึก

ผลการทดลองแสดงให้เห็นว่าโมเดลสามารถเรียนรู้ลักษณะของภาพแมวและสุนัขได้ และมีช่องว่างระหว่าง Training Accuracy กับ Validation Accuracy ในระดับที่สามารถนำไปวิเคราะห์ต่อได้

หมายเหตุ: ค่าผลลัพธ์อาจเปลี่ยนแปลงได้ตาม Dataset, Random Seed และสภาพแวดล้อมในการ Training


สรุป

LAB07 นี้เป็นการประยุกต์ใช้ Convolutional Neural Network (CNN) สำหรับจำแนกรูปภาพ Cat และ Dog โดยเริ่มตั้งแต่การเตรียมข้อมูล การ Preprocessing การแบ่ง Dataset การสร้าง CNN Model การ Training และการประเมินผล

จากการทดลองพบว่าการเพิ่มจำนวนข้อมูลจาก Dataset ขนาดเล็กเป็นประมาณ 6,000 รูป ช่วยให้โมเดลสามารถเรียนรู้ได้ดีขึ้น และการปรับ Learning Rate เป็น 1e-3 ให้ผลการทดลองที่ดีกว่าค่าที่ทดลองก่อนหน้า

ผลลัพธ์ที่ได้สามารถนำไปใช้เป็นพื้นฐานสำหรับการศึกษา Image Classification และการพัฒนาโมเดล Deep Learning ต่อไป