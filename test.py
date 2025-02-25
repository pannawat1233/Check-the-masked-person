import os
import re
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Flatten, Dense
from tensorflow.keras.models import load_model


folder_path = "D:/Code/AI/Check the masked person/images"
xml_folder = "D:/Code/AI/Check the masked person/annotations"

x_data = []
y_data = []

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        label = obj.find("name").text
        bndbox = obj.find("bndbox")  
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})
    return objects

def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else -1

xml_files = sorted(os.listdir(xml_folder), key=extract_number)
encode = LabelEncoder()

for file in xml_files:
    file_path = os.path.join(xml_folder, file)

    if not os.path.isfile(file_path) or not file.lower().endswith('.xml'):
        continue

    try:
        image_name = file.replace(".xml", ".png")  
        image_path = os.path.join(folder_path, image_name)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue  
        img = Image.open(image_path).convert('L') 
        img = img.resize((512, 512)) 
        x_data.append(np.array(img))
        y_data.append(parse_xml(file_path)[0]["label"])  

    except Exception as e:
        print(f"Error processing {file}: {e}")



y_data = encode.fit_transform(y_data)
x_data = np.array(x_data)
x_data = x_data.astype('float32') / 255.0
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

model = Sequential()
model.add(Flatten(input_shape=(512, 512 ,1)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=14)
model.evaluate(x_test, y_test)

model.save('my_model.h5') 
model = load_model("my_model.h5")

