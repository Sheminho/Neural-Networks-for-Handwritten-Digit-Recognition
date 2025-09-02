from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = FastAPI()

model = load_model("Shimy_model.h5")

@app.get("/")
def home():
    return{"message":"Digit Recognition API is running!, Hi from Umm khinan"}

@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    # قراءة الصورة المرفوعة
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # تحويل لصورة رمادية
    image = image.resize((28, 28))  # نفس حجم MNIST
    img_array = np.array(image) / 255.0  # تطبيع البيانات
    img_array = 1 - img_array
    img_array = img_array.reshape(1, 28, 28, 1)
    

    # التنبؤ
    prediction = model.predict(img_array)
    digit = int(np.argmax(prediction))

    return {"predicted_digit": int(digit)}