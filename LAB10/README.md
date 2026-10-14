# Aquarium Object Detection with YOLOv8

โปรเจกต์นี้เป็นงานตรวจจับวัตถุในภาพตู้ปลา (Object Detection) โดยใช้ **YOLOv8** และชุดข้อมูล Aquarium จาก Roboflow Universe ฝึกโมเดลเพื่อระบุชนิดของสัตว์น้ำในภาพ พร้อมประเมินผลและทดสอบการตรวจจับบนภาพจริง

## 1. แหล่งอ้างอิง Dataset

- **Dataset:** [Aquarium Dataset – Roboflow Universe (Version 2)](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2)
- **ผู้เผยแพร่:** Roboflow 100
- **ประเภทงาน:** Object Detection
- **License:** CC BY 4.0
- **จำนวนภาพตามหน้า Dataset:** 638 ภาพ แบ่งเป็น Train 448 ภาพ, Validation 127 ภาพ และ Test 63 ภาพ

ชุดข้อมูลมี 7 คลาส ได้แก่

1. `fish`
2. `jellyfish`
3. `penguin`
4. `puffin`
5. `shark`
6. `starfish`
7. `stingray`

> หากนำชุดข้อมูลหรือภาพจากชุดข้อมูลไปเผยแพร่ในรายงานหรือผลงาน ควรระบุแหล่งที่มาและปฏิบัติตามเงื่อนไขของ CC BY 4.0

## 2. โครงสร้างโปรเจกต์

โครงสร้างหลักที่ใช้ในโปรเจกต์:

```text
LAB10/
├── dataset/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── train.py
├── evaluate.py
├── evaluate_best30.py
├── image_test.py
├── compare_epochs.py
├── runs/
│   ├── aquarium/
│   │   └── weights/
│   │       ├── best.pt
│   │       └── last.pt
│   ├── epoch_comparison/
│   ├── aquarium_30epochs_test/
│   └── image_test/
└── README.md
```

โฟลเดอร์ `runs/` จะถูกสร้างหรือเติมไฟล์เมื่อรันสคริปต์ต่าง ๆ ชื่อโฟลเดอร์ผลลัพธ์อาจแตกต่างได้ตามค่าที่ตั้งไว้ในสคริปต์

## 3. เตรียมสภาพแวดล้อม

เปิด Terminal ที่โฟลเดอร์ `LAB10` และเปิดใช้งาน Python virtual environment ที่ติดตั้งไลบรารีไว้แล้ว จากนั้นตรวจสอบไลบรารี:

```powershell
python -c "import torch, ultralytics; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Ultralytics:', ultralytics.__version__)"
```

โปรเจกต์นี้เคยตรวจพบ GPU `NVIDIA GeForce RTX 4070 Laptop GPU` และ PyTorch ที่รองรับ CUDA หาก `CUDA available` เป็น `True` จะสามารถใช้ GPU ในการฝึกและประเมินโมเดลได้

หากยังไม่ได้ติดตั้งไลบรารีที่จำเป็น ให้ติดตั้งใน environment ที่ใช้กับโปรเจกต์:

```powershell
python -m pip install ultralytics
```

## 4. ตรวจสอบไฟล์กำหนด Dataset

ไฟล์ `dataset/data.yaml` ระบุพาธของชุดข้อมูล จำนวนคลาส และชื่อคลาส ตัวอย่างการตั้งค่าที่ใช้:

```yaml
train: train/images
val: valid/images
test: test/images

nc: 7
names: ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']
```

สคริปต์ฝึกและประเมินผลอ้างอิงไฟล์นี้ โดยพาธภายใน YAML ต้องสอดคล้องกับโครงสร้างโฟลเดอร์จริง

## 5. ฝึกโมเดล

รันคำสั่งจากโฟลเดอร์ `LAB10`:

```powershell
python train.py
```

ขั้นตอนหลักของการฝึก:

1. โหลดโมเดลเริ่มต้น YOLOv8n (`yolov8n.pt`)
2. อ่านภาพและ Label จากชุด Train
3. ฝึกโมเดลเพื่อเรียนรู้การระบุตำแหน่งและชนิดของวัตถุ
4. ใช้ชุด Validation ตรวจสอบผลระหว่างการฝึก
5. บันทึกน้ำหนักโมเดลและผลการฝึกไว้ใน `runs/`

ไฟล์น้ำหนักที่สำคัญ:

- `best.pt` — น้ำหนักจาก Epoch ที่ให้ผลดีที่สุดตามเกณฑ์ Validation ของการฝึกครั้งนั้น
- `last.pt` — น้ำหนักจาก Epoch สุดท้าย

ตรวจสอบพารามิเตอร์ใน `train.py` เพื่อดูค่า `epochs`, `imgsz`, `batch`, `device`, `project` และ `name` ที่ใช้จริง เนื่องจากค่าที่ตั้งในสคริปต์เป็นตัวกำหนดการฝึก

## 6. เปรียบเทียบจำนวน Epoch

ใช้สคริปต์เปรียบเทียบโมเดลที่ฝึกด้วยจำนวน Epoch ต่างกัน:

```powershell
python compare_epochs.py
```

สคริปต์นี้ฝึกโมเดลแยกกันสำหรับ 10, 20 และ 30 Epoch แล้วเปรียบเทียบผลบนชุด Validation โดยใช้ตัวชี้วัด Precision, Recall, mAP@50 และ mAP@50–95 ผลและกราฟจะอยู่ภายใต้:

```text
runs/epoch_comparison/
```

ผล Validation จากการรันเปรียบเทียบครั้งหนึ่ง:

| Epoch | Precision | Recall | mAP@50 | mAP@50–95 |
|---:|---:|---:|---:|---:|
| 10 | 0.7165 | 0.6317 | 0.6755 | 0.3923 |
| 20 | 0.7825 | 0.6755 | 0.7300 | 0.4390 |
| 30 | 0.8073 | 0.6536 | 0.7447 | 0.4567 |

จากผลชุดนี้ โมเดล 30 Epoch มี Precision, mAP@50 และ mAP@50–95 สูงที่สุด ส่วนโมเดล 20 Epoch มี Recall สูงที่สุด การเลือกโมเดลควรพิจารณาว่าต้องการลดการตรวจจับผิดหรือเพิ่มโอกาสตรวจพบวัตถุให้มากขึ้น

## 7. ประเมินผลบนชุด Test

ประเมินโมเดลด้วยชุด Test ซึ่งไม่ใช่ชุดเดียวกับที่ใช้ฝึก:

```powershell
python evaluate.py
```

หากต้องการประเมินโมเดล 30 Epoch จากการเปรียบเทียบ ให้ตรวจสอบว่า `evaluate_best30.py` ชี้ไปยังไฟล์นี้:

```text
runs/epoch_comparison/aquarium_30epochs/weights/best.pt
```

จากนั้นรัน:

```powershell
python evaluate_best30.py
```

ผลการประเมินและกราฟจะถูกบันทึกในโฟลเดอร์ `runs/` ตามค่าที่กำหนดในสคริปต์

### ผล Test จากการรันครั้งล่าสุด

| ตัวชี้วัด | ผล |
|---|---:|
| Precision | 0.7994 (79.94%) |
| Recall | 0.6953 (69.53%) |
| mAP@50 | 0.7740 (77.40%) |
| mAP@50–95 | 0.4812 (48.12%) |

การประเมินครั้งนี้ใช้ภาพ Test 63 ภาพ และมีวัตถุที่ติดป้ายกำกับ 584 instances ผลรายคลาสที่ได้:

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| fish | 0.766 | 0.657 | 0.730 | 0.413 |
| jellyfish | 0.834 | 0.805 | 0.851 | 0.549 |
| penguin | 0.807 | 0.659 | 0.750 | 0.344 |
| puffin | 0.688 | 0.457 | 0.539 | 0.268 |
| shark | 0.852 | 0.756 | 0.836 | 0.586 |
| starfish | 0.735 | 0.818 | 0.854 | 0.555 |
| stingray | 0.915 | 0.715 | 0.858 | 0.654 |

**ข้อสังเกต:** Puffin เป็นคลาสที่มีผล mAP@50 ต่ำที่สุดในผลชุดนี้ จึงเป็นคลาสที่ควรพิจารณาปรับปรุงในอนาคต อย่างไรก็ตาม ควรตรวจสอบพาธโมเดลในสคริปต์ประเมินผลก่อนสรุปว่าค่านี้เป็นผลของโมเดล 30 Epoch โดยเฉพาะ

## 8. ทดสอบกับภาพ

รันสคริปต์ทดสอบภาพ:

```powershell
python image_test.py
```

สคริปต์จะนำโมเดลที่กำหนดไว้ในไฟล์ไปตรวจจับวัตถุบนภาพจากชุด Test และบันทึกภาพผลลัพธ์พร้อมกรอบ Bounding Box และชื่อคลาสไว้ในโฟลเดอร์ผลลัพธ์ เช่น:

```text
runs/image_test/
```

ภาพที่ได้ช่วยตรวจสอบด้วยสายตาว่าโมเดลระบุคลาสและตำแหน่งของวัตถุได้เหมาะสมหรือไม่ ควรพิจารณาทั้งตัวอย่างที่ตรวจจับถูกและตัวอย่างที่ตรวจจับผิด ไม่ควรใช้ภาพตัวอย่างเพียงไม่กี่ภาพแทนการประเมินเชิงตัวเลข

## 9. ความหมายของตัวชี้วัด

- **Precision:** สัดส่วนการตรวจจับที่โมเดลทำนายว่าเป็นวัตถุแล้วถูกต้อง
- **Recall:** สัดส่วนวัตถุจริงที่โมเดลตรวจพบ
- **mAP@50:** ค่าเฉลี่ยประสิทธิภาพการตรวจจับที่ใช้เกณฑ์ IoU 0.50
- **mAP@50–95:** ค่า mAP ที่เฉลี่ยหลายเกณฑ์ IoU ตั้งแต่ 0.50 ถึง 0.95 จึงเป็นเกณฑ์ที่เข้มงวดกว่า mAP@50 โดยทั่วไป

## 10. ลำดับการทำงานโดยสรุป

1. ดาวน์โหลด Dataset Version 2 จาก Roboflow ในรูปแบบที่รองรับ YOLO และจัดเก็บไว้ใน `dataset/`
2. ตรวจสอบ `dataset/data.yaml` และโครงสร้าง `train`, `valid`, `test`
3. ฝึกโมเดลด้วย `python train.py`
4. เปรียบเทียบ 10, 20 และ 30 Epoch ด้วย `python compare_epochs.py`
5. ประเมินโมเดลบน Test ด้วย `python evaluate.py` หรือสคริปต์ประเมินโมเดลที่ต้องการ
6. ทดลองตรวจจับภาพด้วย `python image_test.py`
7. ตรวจสอบตัวเลขประเมินผลและภาพที่ทำนาย ก่อนสรุปผลโครงการ

## References

1. Roboflow 100, *Aquarium Dataset, Version 2*. [https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2](https://universe.roboflow.com/roboflow-100/aquarium-qlnqy/dataset/2)
2. Roboflow, *Roboflow 100: A Multi-Domain Object Detection Benchmark*. [https://roboflow.com/blog/roboflow-100](https://roboflow.com/blog/roboflow-100)
3. Ultralytics, *YOLO Documentation*. [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
