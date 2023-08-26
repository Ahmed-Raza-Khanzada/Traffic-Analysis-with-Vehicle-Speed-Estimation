import cv2
import numpy as np
import math
from collections import Counter


all_locations = []
def change_tuple(element):
    t = []
    for i in element.split(","):
        t.append(int(i))
    t = tuple(t)
    return t

def get_coordinates(path="coor.txt"):
    with open(path,"r") as f:
      l = f.readlines()
      for loc in l:
        locations = []
     

        for pos,coor in enumerate(loc.split("),")):
            if pos==0:
                coor = coor[2:]
            elif pos==(len(loc.split("),"))-1):
                coor = coor[2:-3]
            else:
                coor = coor[2:]

            coor = change_tuple(coor)
            locations.append(coor)
        all_locations.append(locations)
    return all_locations  


def estimateSpeed(initial_fps,location1, location2,id1):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] *0.2 #/ carWidht
    # Calculate ppm based on bounding box width
    # if len(location2)>2:
    #     bounding_box_area = location2[3]*location2[2]
    #     if bounding_box_area > 0:
    #         chage_in_y = abs(location2[1]>location1[1])
    #         if chage_in_y:
    #             ppm =  (bounding_box_area /10000)+chage_in_y # Adjust the factor50 as needed
    #             # print(f"ID {id1}","Car area", bounding_box_area, "PPM", ppm)
    #         else:
    #             ppm =  (bounding_box_area /10000)
    #     else:

    #         ppm = 17  # Default ppm value
    # else:
    #     ppm = 20  # Default ppm value
    # print(id1,bounding_box_area,ppm)
    if location2[1]<200:
        ppm = 9
    elif location2[1]<267:
        ppm = 11
    else:
        ppm = 30
    d_meters = d_pixels / ppm
    fps = initial_fps
    speed = d_meters * fps * 3.6
    # if speed>80:
    #     speed = 60+((speed-80)*0.6)
    return speed

def fancy_bbox(img,bbox,l=13,t=2):
    x,y,x1,y1 = bbox
    color = (245, 66, 12)
    cv2.line(img,(x,y),(x+l,y),color,t)
    cv2.line(img,(x,y),(x,y+l),color,t)
    
    cv2.line(img,(x1,y),(x1-l,y),color,t)
    cv2.line(img,(x1,y),(x1,y+l),color,t)
   
    cv2.line(img,(x,y1),(x,y1-l),color,t)
    cv2.line(img,(x,y1),(x+l,y1),color,t)
   
    cv2.line(img,(x1,y1),(x1-l,y1),color,t)
    cv2.line(img,(x1,y1),(x1,y1-l),color,t)    
    return img

def get_polygon_center(polygon):
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    center_x = sum(x_coords) / len(polygon)
    center_y = sum(y_coords) / len(polygon)
    return center_x, center_y
def make_poly(img,spot):
	
    overlay = img.copy()

    alpha = 0.6
    output = img.copy()
    zones_color = (242, 64, 24)#(220,20,150)#
    for i in range(len(spot)):
        output = cv2.fillPoly(img,[np.array(spot[i],np.int32)],zones_color[::-1])
        center_x, center_y = get_polygon_center(spot[i])
        cv2.putText(output, f"Zone{i+1}", (int(center_x)-10, int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
            0, output)

    return output
def create_coor_file(coor,path="./coors/coor.txt"):
	textfile = open(path, "w")
	for element in coor:
		textfile.write(str(element) + "\n")
	textfile.close()
        


def count_time_in_video_frame(ret,cap, prev_timestamp):
 

  
    current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

    if prev_timestamp is not None:
        time_interval = current_timestamp - prev_timestamp
        print(f"Timestamp = {current_timestamp:.2f}s, Time Interval = {time_interval:.2f}s")

    return current_timestamp
def replace_zone_name(zname):
    t = "Zone "
    return t+str(int(zname[-1])+1) 
def write_copywright(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 1
    x,y = 8,13
    transparent_box_alpha = 0.8
    overall_text = "Created by: Ahmed Raza Khanzada"
    color_design = (242, 172, 82)#(200, 200, 200)
    # Calculate box dimensions
    text_size = cv2.getTextSize(overall_text, font, 0.4, font_thickness)[0]
    box_width = text_size[0]+17
    # Calculate the total height required for text within the box
    box_height = text_size[1] + 5

    # Draw the transparent box
    box_coords = [(x, y-3), (x + box_width, y-3), (x + box_width, y + box_height), (x, y + box_height)]
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(box_coords, np.int32)], color_design)
    cv2.addWeighted(overlay, transparent_box_alpha, frame, 1 - transparent_box_alpha, 0, frame)
    
    cv2.putText(frame, overall_text, (x+5,y+text_size[1]), cv2.FONT_HERSHEY_PLAIN, font_scale, (0,0,0), 2)
    return frame    

def draw_horizontal_line(image, height1, line_color = (0,0,255)  ):
    # Get image dimensions
    height, width, _ = image.shape
    
    # Draw a horizontal line on the image
   
    line_thickness = 2
    cv2.line(image, (0, height1 ), (width, height1 ), line_color, line_thickness)

    return image


def display_zone_info(video_frame, zone_data,total_objects, x=20, y=50):

    # # # # # Call the function to draw the horizontal line on the image
    # video_frame = draw_horizontal_line(video_frame, 267)
    # video_frame = draw_horizontal_line(video_frame, 200, (255,255,0))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)
    font_thickness = 1
    box_line_spacing =25
    line_spacing = 30
    transparent_box_alpha = 0.5  # Adjust transparency here
    y= y+7
    old_y = y
    color_design = (242, 172, 82)#(200, 200, 200)
    all_class_names = list(total_objects.values())
    video_frame = write_copywright(video_frame)
    if len(all_class_names)>0:
        
        counter_objects = Counter(all_class_names)
        
        fix_width_box = f"Total    Personss: 1000 "
        for pos,(k,v) in enumerate(counter_objects.items()):
            if pos==0:
                y = y+25
            else:
                y = y+20
            
            
            overall_text = f"Total {str(k).capitalize()}s: {str(v)}"

            # Calculate box dimensions
            text_size = cv2.getTextSize(overall_text, font, font_scale, font_thickness)[0]
            box_width = max(text_size[0], cv2.getTextSize(fix_width_box, font, font_scale, font_thickness)[0][0]) -5
            # Calculate the total height required for text within the box
            box_height = text_size[1] + 15 + (len(set(all_class_names)) + 1) 

            # Draw the transparent box
            box_coords = [(x, y-5), (x + box_width, y-5), (x + box_width, y + box_height), (x, y + box_height)]
            overlay = video_frame.copy()
            cv2.fillPoly(overlay, [np.array(box_coords, np.int32)], color_design)
            cv2.addWeighted(overlay, transparent_box_alpha, video_frame, 1 - transparent_box_alpha, 0, video_frame)
           
            cv2.putText(video_frame, overall_text, (x+5,y+text_size[1]), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 2)

        y += box_height+line_spacing

    x=1030
    y=old_y
    for zone_name, vehicle_list in zone_data.items():
        # Extract object class and speed information
        object_classes = [item[1] for item in vehicle_list]
        speeds = [item[4] for item in vehicle_list]

        # Calculate total object count and average speed
        total_objects = len(object_classes)
        avg_speed = sum(speeds) / total_objects if total_objects > 0 else 0

        # Prepare text to display
        text = f"{replace_zone_name(zone_name).capitalize()}"
        text2 = f"Average Speed: {round(avg_speed, 2)} km/h"
        
        # Get the size of the text
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # Calculate box dimensions
        box_width = max(text_size[0], cv2.getTextSize(text2, font, font_scale, font_thickness)[0][0]) + 10

        # Calculate the total height required for text within the box
        box_height = text_size[1] + 17 + (len(set(object_classes)) + 1) * 19

        # Draw the transparent box
        box_coords = [(x, y-5), (x + box_width, y-5), (x + box_width, y + box_height), (x, y + box_height)]
        overlay = video_frame.copy()
        cv2.fillPoly(overlay, [np.array(box_coords, np.int32)], color_design)
        cv2.addWeighted(overlay, transparent_box_alpha, video_frame, 1 - transparent_box_alpha, 0, video_frame)

        # Add text on top of the transparent box
        cv2.putText(
            video_frame,
            text,
            (x + 5, y + text_size[1]),
            font,
            font_scale + 0.1,
            font_color,
            font_thickness + 1,  # Increase font thickness for bold effect
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            video_frame,
            text2,
            (x + 5, y + text_size[1] + 20),
            font,
            font_scale,
            font_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

        unique_classes = set(object_classes)
        each_object_y = y
        for i, object_class in enumerate(unique_classes):
            object_count = object_classes.count(object_class)
            text = f"{object_class}: {object_count}"
            each_object_y += 19
            cv2.putText(
                video_frame,
                text,
                (x + 5, each_object_y + text_size[1] + 20),
                font,
                font_scale,
                font_color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )
        
        # Increment y for the next zone's information
        y += box_height + box_line_spacing

    return video_frame