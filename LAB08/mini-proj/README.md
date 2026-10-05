# LAB08 - Cat vs Dog - 6 DCNN Models

ใช้ Transfer Learning กับ Dataset Cat vs Dog และเปรียบเทียบ 6 โมเดล:

1. VGG16
2. VGG19
3. InceptionV3
4. ResNet101
5. MobileNetV2
6. EfficientNetB0

## Dataset

จัดโฟลเดอร์เป็น:

PetImages/
├── Cat/
└── Dog/

แล้ววางไว้ในโฟลเดอร์ mini-proj/

## วิธีรัน

ติดตั้ง:

pip install -r requirements.txt

แบ่งข้อมูล:

python split_data.py

Train ทั้ง 6 โมเดล:

python train_models.py

สร้างกราฟเปรียบเทียบ:

python compare_models.py

ผลสำคัญ:

outputs/
├── comparison_training_accuracy.png
├── comparison_validation_accuracy.png
├── comparison_training_loss.png
├── comparison_validation_loss.png
├── comparison_test_accuracy.png
└── model_comparison.csv
