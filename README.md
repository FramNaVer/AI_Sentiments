# 🧠 AI Sentiment Analytics

<div align="center">
  <img src="emotion-recognition.png" alt="AI Sentiment Analytics" width="600">
  
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=for-the-badge)](https://huggingface.co/transformers/)
  
  **เครื่องมือวิเคราะห์ความรู้สึกจากข้อความด้วยปัญญาประดิษฐ์ขั้นสูง**
  
  *รองรับทั้งภาษาไทยและภาษาอังกฤษ*  *เป็นระบบที่ เขียนโดย AI 80% เนื่องจากมาเป็นการเขียน model การใช้ AI ครั้งเเรกครับผม 🙏
  *** ส่วนของ ระบบ จะยังใช้ไม่ได้ เพราะ เป็นโปรเเกรมที่ มี model 400 MB ไม่สามารถอัพขึ้น git ได้ เเละ model ที่ทำการเทรนยังไม่ได้เปิดเป็น สาธารณะ 🙏🙏🙏
</div>

---

## ✨ คุณสมบัติหลัก

### 🎯 **การวิเคราะห์ที่แม่นยำ**
- 🇹🇭 **รองรับภาษาไทย** - ใช้โมเดล BERT ที่ได้รับการฝึกฝนเฉพาะ
- 🇺🇸 **รองรับภาษาอังกฤษ** - โมเดลมาตรฐานสากล
- 🎚️ **ปรับระดับความเชื่อมั่นได้** - กำหนดเกณฑ์ตามความต้องการ

### 📊 **การแสดงผลที่สวยงาม**
- 📈 **กราฟแบบ Interactive** - ใช้ Plotly สำหรับการแสดงผลแบบโต้ตอบ
- 🎨 **UI ที่ทันสมัย** - ออกแแบบด้วย CSS สไตล์โมเดิร์น
- 📱 **Responsive Design** - ใช้งานได้ทุกอุปกรณ์

### 🤖 **AI Recommendations**
- 💡 **คำแนะนำอัจฉริยะ** - วิเคราะห์ผลลัพธ์และให้คำแนะนำ
- 📋 **สรุปอัตโนมัติ** - สรุปแนวโน้มและข้อเสนอแนะ
- 🎯 **การปรับปรุง** - แนะนำวิธีปรับปรุงความพึงพอใจ

### 📁 **รองรับไฟล์หลากหลาย**
- 📊 **CSV Files** - ไฟล์ข้อมูลมาตรฐาน
- 📋 **Excel Files** - รองรับ .xlsx
- 💾 **Export ผลลัพธ์** - ดาวน์โหลดผลการวิเคราะห์

---

## 🚀 การติดตั้งและใช้งาน

### 📋 ความต้องการระบบ

```
Python 3.8+
GPU (แนะนำ, แต่ไม่บังคับ)
RAM อย่างน้อย 4GB
พื้นที่ว่าง 2GB สำหรับโมเดล
```

### ⚙️ การติดตั้ง

1. **โคลนโปรเจค**
```bash
git clone https://github.com/FramNaVer/AI_Sentiments.git
cd AI_Sentiments
```

2. **สร้าง Virtual Environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **ติดตั้ง Dependencies**
```bash
pip install -r requirements.txt
```

4. **ดาวน์โหลดโมเดล**
```bash
# โมเดลจะถูกดาวน์โหลดอัตโนมัติในครั้งแรกที่รัน
# หรือดาวน์โหลดเองจาก Hugging Face
```

### 🎮 การใช้งาน

1. **เรียกใช้แอปพลิเคชัน**
```bash
streamlit run sentiment_analyzer.py
```

2. **เปิดเว็บเบราว์เซอร์**
```
http://localhost:8501
```

3. **อัปโหลดไฟล์ข้อมูล**
   - เลือกไฟล์ CSV หรือ Excel
   - กำหนดคอลัมน์ที่มีข้อความ
   - เลือกภาษาโมเดล

4. **วิเคราะห์ผลลัพธ์**
   - ดูกราฟการกระจายความรู้สึก
   - อ่านคำแนะนำจาก AI
   - ดาวน์โหลดผลลัพธ์

---

## 📊 ตัวอย่างการใช้งาน

### 📝 **รูปแบบข้อมูลที่รองรับ**

```csv
comment,rating
"สินค้าดีมาก บริการเยี่ยม",5
"การจัดส่งช้าไป แต่สินค้าโอเค",3
"ไม่พอใจเลย คุณภาพแย่",1
```

### 📈 **ผลลัพธ์ที่ได้**

- **Positive**: 45.2% (1,356 ความคิดเห็น)
- **Neutral**: 32.1% (963 ความคิดเห็น)  
- **Negative**: 22.7% (681 ความคิดเห็น)

### 🎯 **คำแนะนำ AI**

> 💡 **ข้อเสนะแนะ**: จากการวิเคราะห์พบว่าลูกค้าพอใจในคุณภาพสินค้า แต่ควรปรับปรุงเรื่องการจัดส่งเพื่อเพิ่มความพึงพอใจ

---

## 🛠️ เทคโนโลยีที่ใช้

### 🧠 **AI & Machine Learning**
- **[Transformers](https://huggingface.co/transformers/)** - BERT Models
- **[PyTorch](https://pytorch.org/)** - Deep Learning Framework
- **[Hugging Face](https://huggingface.co/)** - Pre-trained Models

### 💻 **Web Application**
- **[Streamlit](https://streamlit.io/)** - Web App Framework
- **[Plotly](https://plotly.com/)** - Interactive Visualization
- **[Pandas](https://pandas.pydata.org/)** - Data Manipulation

### 🎨 **UI/UX**
- **Custom CSS** - Modern Design
- **Google Fonts** - Typography
- **Responsive Layout** - Mobile Friendly

---

## 📁 โครงสร้างโปรเจค

```
AI_Sentiments/
├── 📄 sentiment_analyzer.py     # Main application
├── 🎨 style.css                # Additional styling
├── 🤖 bert_sentiment/          # AI Models directory
│   ├── 📊 label_encoder.pkl    # Label encoder
│   └── 🧠 bert_sentiment/      # BERT model files
├── 🖼️ assets/                  # Images and assets
├── 📋 requirements.txt         # Python dependencies
├── 🚫 .gitignore              # Git ignore rules
└── 📖 README.md               # This file
```

---

## 🎯 การใช้งานขั้นสูง

### ⚙️ **ปรับแต่งพารามิเตอร์**

```python
# ในโค้ด sentiment_analyzer.py
max_length = 128        # ความยาวข้อความสูงสุด
confidence_threshold = 0.7  # เกณฑ์ความเชื่อมั่น
batch_size = 32         # ขนาด batch สำหรับการประมวลผล
```

### 🔧 **เพิ่มโมเดลใหม่**

1. วางไฟล์โมเดลใน `bert_sentiment/`
2. อัปเดตฟังก์ชัน `load_models()`
3. เพิ่มตัวเลือกใน UI

### 📊 **Custom Visualization**

```python
# เพิ่มกราฟใหม่
fig = px.pie(results, values='count', names='sentiment')
st.plotly_chart(fig)
```

---

## 🤝 การมีส่วนร่วม

เรายินดีรับการมีส่วนร่วมจากชุมชน! 

### 🐛 **รายงานปัญหา**
- ใช้ [GitHub Issues](https://github.com/FramNaVer/AI_Sentiments/issues)
- ระบุรายละเอียดและขั้นตอนการเกิดปัญหา

### 💡 **เสนอฟีเจอร์ใหม่**
- เปิด Feature Request ใน Issues
- อธิบายการใช้งานและประโยชน์

### 🔧 **Pull Requests**
1. Fork โปรเจค
2. สร้าง feature branch
3. Commit การเปลี่ยนแปลง
4. สร้าง Pull Request

---

## 📄 ลิขสิทธิ์

โปรเจคนี้เผยแพร่ภายใต้ [MIT License](LICENSE)

---

## 👨‍💻 ผู้พัฒนา

<div align="center">
  
**FramNaVer** 

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/FramNaVer)

*"AI for everyone, everywhere"*

</div>

---

## 🙏 ขอบคุณ

- **Hugging Face** - สำหรับ pre-trained models
- **Streamlit Community** - สำหรับ framework ที่ยอดเยี่ยม
- **Open Source Community** - สำหรับ libraries ต่างๆ

---

<div align="center">
  
### 🌟 ถ้าโปรเจคนี้มีประโยชน์ กรุณาให้ Star ⭐

**Made with ❤️ for Thai AI Community**

</div>
