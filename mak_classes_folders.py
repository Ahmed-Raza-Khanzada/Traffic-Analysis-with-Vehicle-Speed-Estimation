import os
import shutil
inpt_str = ["car","bus","truck","ambulance"]
# Path to the folder containing the mixed images
input_folder = r'H:\upwork\vehicle-submission\vehicle-speed-estimation-v7\output_images_classification\training_image'  # Change this to your input folder path

# Path to the output folder where class-wise folders will be created
output_folder = r'H:\upwork\vehicle-submission\vehicle-speed-estimation-v7\output_images_classification\arrange_data'  # Change this to your output folder path

# Iterate through all image files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpeg') or filename.endswith('.jpg') or  filename.endswith('.png'):
        # Extract class information from the filename (e.g., "103car.txt" -> "car")
        class_name = filename.split('.')[0]  # Extract the class name from the filename
        for i in inpt_str:
            if i in class_name.lower():
                class_name = i
                break
            
        class_folder = os.path.join(output_folder, class_name)
        
        # Create the class-wise folder if it doesn't exist
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        
        # Get the full path of the source image
        source_path = os.path.join(input_folder, filename)
        
        # Get the full path of the destination image (inside the class folder)
        destination_path = os.path.join(class_folder, filename)
        
        # Move the image to the class-wise folder
        shutil.move(source_path, destination_path)

print("Images organized into class-wise folders.")
