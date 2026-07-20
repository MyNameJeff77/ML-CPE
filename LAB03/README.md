ชุดข้อมูลที่ใช้ Kaggle https://www.kaggle.com/datasets/yasserh/titanic-dataset?utm_source=chatgpt.com

# LAB 1 : Regression (Titanic Dataset)

## Objective
ศึกษาการสร้างโมเดล Regression โดยใช้ชุดข้อมูล Titanic เพื่อทำนายอายุ (Age) ของผู้โดยสารจากข้อมูลที่เกี่ยวข้อง

---

## ขั้นตอนการทำงาน

### 1. Import Libraries
นำเข้าไลบรารีที่จำเป็น เช่น Pandas สำหรับจัดการข้อมูล และ Scikit-learn สำหรับสร้างโมเดล Machine Learning

### 2. Load Dataset
โหลดชุดข้อมูล Titanic จากไฟล์ `dataset.csv` และตรวจสอบข้อมูลเบื้องต้น

### 3. Data Preprocessing
เตรียมข้อมูลก่อนสร้างโมเดล โดยเติมค่าที่หายไปในคอลัมน์ Age ด้วยค่ามัธยฐาน (Median) และแปลงข้อมูลเพศ (Sex) จากข้อความเป็นตัวเลข

### 4. Simple Linear Regression
กำหนดให้ตัวแปรอิสระ (X) คือ `Fare` และตัวแปรเป้าหมาย (y) คือ `Age` จากนั้นแบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ สร้างโมเดล Linear Regression และใช้โมเดลทำนายค่าอายุของผู้โดยสาร

### 5. Multiple Linear Regression
กำหนดตัวแปรอิสระหลายตัว ได้แก่ `Pclass`, `Fare` และ `SibSp` เพื่อใช้ร่วมกันในการทำนาย `Age` จากนั้นแบ่งข้อมูล สร้างโมเดล และทดสอบผลการทำนาย

---

## สรุป

- Simple Linear Regression ใช้ตัวแปรเพียง 1 ตัวในการทำนาย
- Multiple Linear Regression ใช้หลายตัวแปรร่วมกัน จึงสามารถนำข้อมูลมาประกอบการทำนายได้มากขึ้น
- การทดลองนี้ช่วยให้เข้าใจขั้นตอนการสร้างโมเดล Regression ตั้งแต่การเตรียมข้อมูล การฝึกโมเดล และการทำนายผล