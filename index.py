from PIL import Image
import os

def resize_images(folder_path, target_size=(512, 512)):
    print(f"Starting resize process for folder: {folder_path}")  
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        print(f"Checking file: {file_path}")  
        
       
        if not os.path.isfile(file_path):  
            continue

        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:

                img = Image.open(file_path)
                img = img.resize(target_size)

                new_file_path = os.path.join(folder_path, f"resized_{file}")
                img.save(new_file_path)
                print(f"Resized and saved: {new_file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

resize_images(r'D:\KU66\Code\DataSet\Check the masked person\images', target_size=(512, 512))
