import streamlit as st
import pandas as pd
import torch
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from io import BytesIO
from xlsxwriter import Workbook
import numpy as np

# CONFIG
st.set_page_config(
    page_title="AI Sentiment Analytics", 
    layout="wide", 
    page_icon="🧠", 
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #f0f2f6;
        text-align: center;
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Chart Container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    /* Upload Area */
    .upload-area {
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: #f0f3ff;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* AI Suggestion Box */
    .ai-suggestion {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        color: #333;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    /* Status Indicators */
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Data Table Styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
</style>
""", unsafe_allow_html=True)

# LOAD MODELS
@st.cache_resource
def load_models():
    model_dict = {}
    
    try:
        # Thai model
        thai_path = "bert_sentiment/bert_sentiment"
        thai_tokenizer = AutoTokenizer.from_pretrained(thai_path, use_fast=False)
        thai_model = AutoModelForSequenceClassification.from_pretrained(thai_path)
        thai_label = joblib.load(f"{thai_path}/../label_encoder.pkl")
        model_dict["thai"] = (thai_tokenizer, thai_model, thai_label)
    except:
        st.error("⚠️ ไม่สามารถโหลดโมเดลภาษาไทยได้")
    
    try:
        # English model
        eng_model_id = "distilbert-base-uncased-finetuned-sst-2-english"
        eng_tokenizer = AutoTokenizer.from_pretrained(eng_model_id)
        eng_model = AutoModelForSequenceClassification.from_pretrained(eng_model_id)
        eng_label = {0: "Negative", 1: "Positive"}
        model_dict["eng"] = (eng_tokenizer, eng_model, eng_label)
    except:
        st.error("⚠️ ไม่สามารถโหลดโมเดลภาษาอังกฤษได้")
    
    return model_dict

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 AI Sentiment Analytics</h1>
    <p>วิเคราะห์ความรู้สึกจากข้อความด้วยปัญญาประดิษฐ์ขั้นสูง</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🎛️ การตั้งค่า")
    
    # Language Selection
    st.markdown("#### 🌐 เลือกภาษาโมเดล")
    model_lang = st.radio(
        "เลือกภาษาที่ต้องการวิเคราะห์",
        ["🇹🇭 ภาษาไทย", "🇺🇸 English"],
        help="เลือกภาษาที่ตรงกับข้อมูลของคุณ"
    )
    model_key = "thai" if "ไทย" in model_lang else "eng"
    
    st.markdown("#### ⚙️ พารามิเตอร์")
    max_length = st.slider("ความยาวข้อความสูงสุด", 50, 500, 128)
    confidence_threshold = st.slider("เกณฑ์ความเชื่อมั่น", 0.5, 1.0, 0.7)
    
    st.markdown("#### 📊 ตัวเลือกการแสดงผล")
    show_invalid_data = st.checkbox("แสดงข้อมูลที่ไม่ถูกต้อง", value=True)
    show_recommendations = st.checkbox("แสดงคำแนะนำ AI", value=True)

# Load models
with st.spinner("🔄 กำลังโหลดโมเดล AI..."):
    model_dict = load_models()

if not model_dict:
    st.error("❌ ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบการตั้งค่า")
    st.stop()

# Prediction function
def predict_sentiment(text, model_dict, model_key, max_length=128):
    try:
        tokenizer, model, label_data = model_dict[model_key]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            
        pred_id = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][pred_id].item()
        
        if model_key == "thai":
            prediction = label_data.inverse_transform([pred_id])[0]
        else:
            prediction = label_data[pred_id]
            
        return prediction, confidence
        
    except Exception as e:
        return f"Error: {e}", 0.0

# Enhanced recommendation function
def generate_recommendation(comment_text, sentiment, model_key):
    if model_key != "thai":
        return "💡 ระบบให้คำแนะนำสำหรับความคิดเห็นภาษาไทยเท่านั้น"
    
    if sentiment.lower() not in ["negative", "ลบ"]:
        return "✅ ความคิดเห็นนี้เป็นบวก ควรคงไว้ซึ่งคุณภาพการบริการที่ดี"
    
    prompt = f"""
    คุณคือผู้เชี่ยวชาญด้านการวิเคราะห์ความคิดเห็นลูกค้า
    ความคิดเห็น: "{comment_text}"
    
    โปรดให้คำแนะนำสั้นๆ เพื่อปรับปรุงสถานการณ์ (ไม่เกิน 100 คำ)
    """
    
    try:
        api_key = st.secrets.get("openrouter_api_key", "")
        if not api_key:
            return "⚠️ ต้องการ API Key เพื่อสร้างคำแนะนำ"
            
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "meta-llama/llama-3-8b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                               headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        return f"❌ ไม่สามารถสร้างคำแนะนำได้: {str(e)[:50]}..."

# Data cleaning function
def clean_comments(df, column):
    original_col = df[column]
    df[column] = original_col.astype(str)
    cleaned_col = df[column].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    is_invalid = (
        original_col.isna() | 
        (cleaned_col == "") | 
        (cleaned_col.str.len() < 3) | 
        (cleaned_col.str.len() > 500)
    )
    
    df_invalid = df[is_invalid].copy()
    df_valid = df[~is_invalid].copy().reset_index(drop=True)
    df_valid[column] = cleaned_col[~is_invalid].reset_index(drop=True)
    
    return df_valid, df_invalid

# File upload section
st.markdown("### 📁 อัปโหลดข้อมูล")

uploaded_file = st.file_uploader(
    "เลือกไฟล์ข้อมูลความคิดเห็น",
    type=["csv", "xlsx"],
    help="รองรับไฟล์ CSV และ Excel (XLSX)"
)

if uploaded_file:
    try:
        # Load data
        with st.spinner("📖 กำลังอ่านข้อมูล..."):
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
                
        st.markdown('<div class="status-success">✅ อ่านไฟล์สำเร็จ</div>', unsafe_allow_html=True)
        
        # Column selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 🎯 เลือกคอลัมน์ข้อมูล")
            columns = list(df.columns)
            selected_column = st.selectbox(
                "คอลัมน์ที่มีข้อความ",
                options=columns,
                help="เลือกคอลัมน์ที่มีความคิดเห็นหรือข้อความที่ต้องการวิเคราะห์"
            )
            
            custom_column = st.text_input(
                "หรือป้อนชื่อคอลัมน์",
                value=selected_column,
                help="สามารถพิมพ์ชื่อคอลัมน์เองได้"
            )
            comment_column = custom_column.strip()
            
        with col2:
            st.markdown("#### 👀 ตัวอย่างข้อมูล")
            st.dataframe(df.head(5), use_container_width=True)
            
        # Validate column
        if comment_column not in df.columns:
            st.error(f"❌ ไม่พบคอลัมน์ '{comment_column}' ในข้อมูล")
            st.info("📋 คอลัมน์ที่มีในไฟล์: " + ", ".join(df.columns))
            st.stop()
            
        # Clean data
        with st.spinner("🧹 กำลังทำความสะอาดข้อมูล..."):
            df_clean, df_invalid = clean_comments(df, comment_column)
            
        # Show data summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">ข้อมูลทั้งหมด</div>
                <div class="metric-value" style="color: #3498db;">{:,}</div>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">ข้อมูลที่ใช้ได้</div>
                <div class="metric-value" style="color: #27ae60;">{:,}</div>
            </div>
            """.format(len(df_clean)), unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">ข้อมูลที่ตัดออก</div>
                <div class="metric-value" style="color: #e74c3c;">{:,}</div>
            </div>
            """.format(len(df_invalid)), unsafe_allow_html=True)
            
        # Show invalid data if requested
        if show_invalid_data and len(df_invalid) > 0:
            with st.expander(f"⚠️ ข้อมูลที่ไม่ผ่านการตรวจสอบ ({len(df_invalid)} รายการ)"):
                st.dataframe(df_invalid.head(10), use_container_width=True)
                
        # Analysis button
        if len(df_clean) > 0:
            analyze_button = st.button(
                "🚀 เริ่มวิเคราะห์ความรู้สึก",
                use_container_width=True,
                help=f"วิเคราะห์ {len(df_clean):,} ข้อความด้วย AI"
            )
            
            if analyze_button:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Analysis
                predictions = []
                confidences = []
                
                for i, text in enumerate(df_clean[comment_column]):
                    pred, conf = predict_sentiment(text, model_dict, model_key, max_length)
                    predictions.append(pred)
                    confidences.append(conf)
                    
                    # Update progress
                    progress = (i + 1) / len(df_clean)
                    progress_bar.progress(progress)
                    status_text.text(f"วิเคราะห์แล้ว {i+1}/{len(df_clean)} ข้อความ")
                
                df_clean["predicted_sentiment"] = predictions
                df_clean["confidence"] = confidences
                
                # Filter by confidence if needed
                high_confidence_df = df_clean[df_clean["confidence"] >= confidence_threshold]
                
                progress_bar.empty()
                status_text.empty()
                
                # Results
                st.markdown("## 📊 ผลการวิเคราะห์")
                
                # Summary metrics
                counts = df_clean["predicted_sentiment"].value_counts()
                total_analyzed = len(df_clean)
                high_conf_count = len(high_confidence_df)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">วิเคราะห์ทั้งหมด</div>
                        <div class="metric-value">{:,}</div>
                    </div>
                    """.format(total_analyzed), unsafe_allow_html=True)
                    
                with col2:
                    most_common = counts.index[0] if len(counts) > 0 else "N/A"
                    most_count = counts.iloc[0] if len(counts) > 0 else 0
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">ความรู้สึกหลัก</div>
                        <div class="metric-value" style="font-size: 1.2rem;">{}</div>
                        <div style="color: #7f8c8d; font-size: 0.8rem;">{:,} ครั้ง</div>
                    </div>
                    """.format(most_common, most_count), unsafe_allow_html=True)
                    
                with col3:
                    avg_confidence = df_clean["confidence"].mean()
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">ความเชื่อมั่นเฉลี่ย</div>
                        <div class="metric-value">{:.1%}</div>
                    </div>
                    """.format(avg_confidence), unsafe_allow_html=True)
                    
                with col4:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-label">ความเชื่อมั่นสูง</div>
                        <div class="metric-value">{:,}</div>
                        <div style="color: #7f8c8d; font-size: 0.8rem;">≥{:.0%}</div>
                    </div>
                    """.format(high_conf_count, confidence_threshold), unsafe_allow_html=True)
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("#### 🥧 การกระจายความรู้สึก")
                    
                    fig_pie = px.pie(
                        values=counts.values,
                        names=counts.index,
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_layout(
                        font=dict(size=12),
                        showlegend=True,
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.markdown("#### 📈 ความถี่ความรู้สึก")
                    
                    fig_bar = px.bar(
                        x=counts.index,
                        y=counts.values,
                        labels={"x": "ความรู้สึก", "y": "จำนวน"},
                        color=counts.values,
                        color_continuous_scale="viridis"
                    )
                    fig_bar.update_layout(
                        showlegend=False,
                        height=400,
                        xaxis_title="ความรู้สึก",
                        yaxis_title="จำนวน"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Confidence distribution
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("#### 📊 กระจายความเชื่อมั่น")
                
                fig_hist = px.histogram(
                    df_clean,
                    x="confidence",
                    nbins=20,
                    title="",
                    labels={"confidence": "ความเชื่อมั่น", "count": "จำนวน"}
                )
                fig_hist.add_vline(
                    x=confidence_threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"เกณฑ์ {confidence_threshold:.0%}"
                )
                fig_hist.update_layout(height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed results
                tab1, tab2, tab3 = st.tabs(["📋 ผลลัพธ์ทั้งหมด", "⭐ ความเชื่อมั่นสูง", "💡 คำแนะนำ AI"])
                
                with tab1:
                    st.markdown("#### 📊 ตารางผลลัพธ์")
                    display_df = df_clean[[comment_column, "predicted_sentiment", "confidence"]].copy()
                    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
                    st.dataframe(display_df, use_container_width=True)
                    
                with tab2:
                    st.markdown(f"#### ⭐ ผลลัพธ์ที่มีความเชื่อมั่น ≥ {confidence_threshold:.0%}")
                    if len(high_confidence_df) > 0:
                        display_high_conf = high_confidence_df[[comment_column, "predicted_sentiment", "confidence"]].copy()
                        display_high_conf["confidence"] = display_high_conf["confidence"].apply(lambda x: f"{x:.1%}")
                        st.dataframe(display_high_conf, use_container_width=True)
                    else:
                        st.info("ไม่มีข้อมูลที่มีความเชื่อมั่นตามเกณฑ์ที่กำหนด")
                        
                with tab3:
                    if show_recommendations and model_key == "thai":
                        st.markdown("#### 💡 คำแนะนำจาก AI สำหรับความคิดเห็นเชิงลบ")
                        
                        negative_comments = df_clean[
                            df_clean["predicted_sentiment"].str.lower().isin(["negative", "ลบ"])
                        ].head(5)
                        
                        if len(negative_comments) > 0:
                            for idx, row in negative_comments.iterrows():
                                comment = row[comment_column]
                                sentiment = row["predicted_sentiment"]
                                conf = row["confidence"]
                                
                                with st.expander(f"💬 {comment[:100]}... (เชื่อมั่น: {conf:.1%})"):
                                    suggestion = generate_recommendation(comment, sentiment, model_key)
                                    st.markdown(f'<div class="ai-suggestion">🤖 {suggestion}</div>', 
                                              unsafe_allow_html=True)
                        else:
                            st.success("🎉 ไม่พบความคิดเห็นเชิงลบ - ข้อมูลของคุณมีความรู้สึกที่ดี!")
                    else:
                        st.info("💡 คำแนะนำ AI สำหรับโมเดลภาษาไทยเท่านั้น")
                
                # Download results
                st.markdown("### 💾 ดาวน์โหลดผลลัพธ์")
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df_clean.to_excel(writer, index=False, sheet_name="Sentiment_Analysis")
                    if len(df_invalid) > 0:
                        df_invalid.to_excel(writer, index=False, sheet_name="Invalid_Data")
                        
                st.download_button(
                    "📥 ดาวน์โหลดไฟล์ Excel",
                    data=output.getvalue(),
                    file_name=f"sentiment_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
        else:
            st.warning("⚠️ ไม่มีข้อมูลที่สามารถวิเคราะห์ได้ กรุณาตรวจสอบไฟล์อีกครั้ง")
            
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")
        st.info("💡 กรุณาตรวจสอบรูปแบบไฟล์และลองใหม่อีกครั้ง")

else:
    # Landing page
    st.markdown("""
    <div style="color:#333; text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
        <h3>🎯 เริ่มต้นใช้งาน</h3>
        <p>อัปโหลดไฟล์ CSV หรือ Excel ที่มีข้อมูลความคิดเห็นของคุณ</p>
        <p>ระบบจะวิเคราะห์ความรู้สึกและให้คำแนะนำเพื่อปรับปรุงการบริการ</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
