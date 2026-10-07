# Climbing Kids Safety AI

ระบบตรวจจับการปีน/พื้นที่อันตรายจากกล้องหรือไฟล์วิดีโอ พร้อมส่งข้อความแจ้งเตือนผ่าน LINE Messaging API

## ตั้งค่า LINE

1. คัดลอก `.env.example` เป็น `.env`
2. กรอก Channel Access Token และ LINE Target ID ลงใน `.env`
3. เพิ่มบอต LINE เป็นเพื่อน และตรวจสอบว่าบอตสามารถส่งข้อความ push ไปยัง Target ID นั้นได้
4. เปิดโปรแกรมใหม่ แล้วกด **ส่งข้อความทดสอบ LINE** บนหน้าเว็บ

อย่าแชร์หรือ commit ไฟล์ `.env` เพราะมีข้อมูลลับอยู่ ไฟล์นี้ถูกละเว้นโดย `.gitignore`

## เริ่มโปรแกรม

ติดตั้ง Python dependencies ที่ใช้ (`opencv-python`, `numpy`, `ultralytics`) แล้วรัน:

```powershell
py .\cctv_video_detect.py
```

โปรแกรมเปิดหน้าเว็บที่ `http://127.0.0.1:8080` โดยอัตโนมัติ
