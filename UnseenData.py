from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model

model = load_model("my_model.h5")
classes = ["ไม่ใส่หน้ากาก", "ใส่หน้ากาก", "ใส่หน้ากากผิดวิธี", "Class 3", "Class 4", 
           "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"]

Test_path = "D:/Code/AI/Check the masked person/test"

for i, imgtest in enumerate(os.listdir(Test_path), 1):
    TestFilePart = os.path.join(Test_path, imgtest)
    
    
    if not os.path.isfile(TestFilePart) or not imgtest.lower().endswith((".jpg", ".png", ".jpeg")):
        print(f" ไม่สามารถทำงานได้: {imgtest}")
        continue

    try:
        ShowTestImg = Image.open(TestFilePart).convert('L')
        ShowTestImg = ShowTestImg.resize((512, 512))
        img_array = np.array(ShowTestImg) / 255.0 

        input_shape = model.input_shape

        if len(input_shape) == 2:  
            img_array = img_array.reshape((1, 512 * 512))  
        else:  
            img_array = np.expand_dims(img_array, axis=0) 
            img_array = np.expand_dims(img_array, axis=-1)  

        
        prediction = model.predict(img_array)[0]
        
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]  
        print(f"กำลังเขียนข้อมูลลงไฟล์: {imgtest}")
        
        with open(r'D:/Code/AI/Check the masked person/fileWrite/test.txt', 'a', encoding='utf-8') as file:
            print(f"ภาพที่ {i}: {imgtest} → {classes[predicted_class]} (ความมั่นใจ {confidence:.2%})\n")
            file.write(f"ภาพที่ {i}: {imgtest} → {classes[predicted_class]} (ความมั่นใจ {confidence:.2%})\n")

    except Exception as e:
        print(f" เกิดข้อผิดพลาดกับไฟล์ {imgtest}: {e}")
