## Regression & Classification

## ชุดข้อมูล Kaggle https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

---

# วัตถุประสงค์

- ศึกษาหลักการของ Regression และ Classification
- ฝึกการเตรียมข้อมูลก่อนสร้างโมเดล
- ศึกษาการใช้ Principal Component Analysis (PCA)
- สร้างโมเดล Linear Regression และ Logistic Regression
- ประเมินและเปรียบเทียบประสิทธิภาพของโมเดล

---

# ขั้นตอนการทำงาน

## 1. Import Library

นำเข้าไลบรารีที่จำเป็นสำหรับการวิเคราะห์ข้อมูล สร้างโมเดล Machine Learning และประเมินผลลัพธ์

ไลบรารีที่ใช้ ได้แก่

- pandas
- numpy
- matplotlib
- scikit-learn

---

## 2. Load Dataset

โหลดชุดข้อมูล Red Wine Quality จากไฟล์ CSV

จากนั้นตรวจสอบว่าข้อมูลถูกโหลดเข้ามาเรียบร้อยด้วยคำสั่ง

- head()
- shape
- info()

---

## 3. Data Exploration (EDA)

สำรวจข้อมูลเบื้องต้นของ Dataset

ประกอบด้วย

- จำนวนข้อมูล
- ชนิดข้อมูล
- ค่าสถิติพื้นฐาน
- Missing Value

จากนั้นสร้าง Histogram เพื่อดูการกระจายตัวของข้อมูลแต่ละ Feature

---

# LAB 1 : Regression

## 4. Simple Linear Regression

เลือกตัวแปร

```
alcohol
```

เป็น Feature

และใช้

```
quality
```

เป็น Target

จากนั้นแบ่งข้อมูลเป็น Training Set และ Testing Set ด้วย train_test_split()

สร้างโมเดลด้วย

```
LinearRegression()
```

เพื่อทำนายคุณภาพของไวน์

---

## 5. Prediction

ใช้โมเดลที่สร้างขึ้นทำนายค่าคุณภาพของไวน์จากข้อมูลชุด Test

---

## 6. Evaluation

ประเมินผลด้วย

- MAE
- RMSE
- R² Score

พร้อมแสดง Scatter Plot เพื่อเปรียบเทียบค่าจริงและค่าที่โมเดลทำนาย

---

## 7. Multiple Linear Regression

เลือกทุก Feature

ยกเว้น

```
quality
```

เพื่อสร้างโมเดล Multiple Linear Regression

จากนั้นฝึกโมเดล ทำนายผล และประเมินประสิทธิภาพเช่นเดียวกับ Simple Linear Regression

---

## 8. เปรียบเทียบ Regression

เปรียบเทียบค่า

- MAE
- RMSE
- R² Score

ของทั้งสองโมเดล

เพื่อดูว่าโมเดลใดให้ผลการทำนายแม่นยำกว่า

---

# LAB 2 : Classification

## 9. Preparing Classification Data

สร้างคอลัมน์ใหม่

```
quality_label
```

กำหนดค่า

- Good Wine = 1
- Bad Wine = 0

โดยใช้เงื่อนไข

```
Quality >= 6
```

---

## 10. Feature Scaling

ใช้

```
StandardScaler()
```

ปรับขนาดข้อมูล

เพื่อให้แต่ละ Feature อยู่ในช่วงเดียวกัน

ช่วยให้โมเดลเรียนรู้ได้ดีขึ้น

---

## 11. Principal Component Analysis (PCA)

ใช้ PCA ลดจำนวน Feature

ให้เหลือ 2 มิติ

เพื่อใช้แสดงผลข้อมูลและ Decision Boundary

---

## 12. Logistic Regression

สร้างโมเดล

```
LogisticRegression()
```

แล้วฝึกโมเดลด้วยข้อมูล Training

จากนั้นใช้โมเดลทำนายข้อมูล Testing

---

## 13. Model Evaluation

ประเมินผลด้วย

- Accuracy
- Precision
- Recall
- F1-score

พร้อมสร้าง

- Confusion Matrix
- Classification Report

เพื่อวิเคราะห์ผลการจำแนกข้อมูล

---

## 14. ROC Curve

สร้าง ROC Curve

และคำนวณค่า AUC

เพื่อประเมินประสิทธิภาพของโมเดล

---

## 15. Decision Boundary

แสดง Decision Boundary

เพื่อให้เห็นการแบ่งพื้นที่ของข้อมูลแต่ละ Class

หลังจากลดมิติข้อมูลด้วย PCA

---

# LAB 3 : Model Comparison

## 16. Simple vs Multiple Linear Regression

เปรียบเทียบ

- MAE
- RMSE
- R² Score

ของทั้งสองโมเดล

เพื่อวิเคราะห์ว่าโมเดลใดเหมาะสมกับชุดข้อมูลมากกว่า

---

## 17. Training vs Testing Performance

เปรียบเทียบผลลัพธ์ของข้อมูล

- Training
- Testing

เพื่อวิเคราะห์ว่าโมเดลเกิด

- Overfitting
- Underfitting

หรือไม่

---

## 18. Regression vs Classification

เปรียบเทียบลักษณะของโมเดลทั้งสองประเภท

Regression

ใช้ตัวชี้วัด

- MAE
- RMSE
- R² Score

Classification

ใช้ตัวชี้วัด

- Accuracy
- Precision
- Recall
- F1-score

---

## 19. Model Performance Metrics

สรุปผลการประเมินของทุกโมเดล

เปรียบเทียบประสิทธิภาพของ

- Simple Linear Regression
- Multiple Linear Regression
- Logistic Regression

---

# สรุปผล

 Multiple Linear Regression ให้ผลการทำนายคุณภาพไวน์ดีกว่า Simple Linear Regression เนื่องจากใช้ข้อมูลจากหลาย Feature พร้อมกัน

สำหรับงาน Classification โมเดล Logistic Regression สามารถจำแนกไวน์เป็น Good Wine และ Bad Wine ได้อย่างมีประสิทธิภาพ โดยใช้ Accuracy, Precision, Recall และ F1-score ในการประเมินผล

นอกจากนี้ การทำ Feature Scaling และ Principal Component Analysis (PCA) ยังช่วยให้โมเดลสามารถเรียนรู้ข้อมูลได้ดีขึ้น และช่วยให้การแสดงผลข้อมูลในรูปแบบ 2 มิติทำได้ง่ายขึ้น
