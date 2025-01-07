# streamlit_phishing_app.py

import streamlit as st
import joblib
import re

# Model va vektorizatorni yuklash
model = joblib.load('model_phishing.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Emailni oldindan qayta ishlash funksiyasi
def preprocess_text(text):
    # Hyperlinklarni olib tashlash
    text = re.sub(r'http\S+', '', text)
    # Belgilarni olib tashlash
    text = re.sub(r'[^\w\s]', '', text)
    # Kichik harflarga o'tkazish
    text = text.lower()
    # Qo'shimcha bo'shliqlarni olib tashlash
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit interfeysini yaratish
st.title("Phishing Email Detection App")
st.write("Ushbu dastur emailni phishing yoki xavfsiz ekanligini aniqlash uchun mo'ljallangan.")

# Foydalanuvchi kiritishi uchun matn maydoni
email_input = st.text_area("Email matnini kiriting", placeholder="Bu yerga email matnini yozing...")

if st.button("Aniqlash"):
    if email_input.strip():
        # Emailni qayta ishlash
        processed_email = preprocess_text(email_input)
        # Matnni TF-IDF formatiga o'tkazish
        email_features = vectorizer.transform([processed_email])
        # Bashorat qilish
        prediction = model.predict(email_features)
        # Natijalarni ko'rsatish
        if prediction[0] == 1:
            st.error("Bu phishing email!")
        else:
            st.success("Bu xavfsiz email.")
    else:
        st.warning("Iltimos, email matnini kiriting.")

# Qo'shimcha: Eslatma
st.write("---")
st.write("Model: SGDClassifier, Saqlash formati: `.pkl`")
