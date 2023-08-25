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
def estimateSpeed(initial_fps,location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] *0.2 #/ carWidht
    ppm = 20#20
    d_meters = d_pixels / ppm
    fps = initial_fps
    speed = d_meters * fps * 3.6
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

    for i in range(len(spot)):
        output = cv2.fillPoly(img,[np.array(spot[i],np.int32)],(150,20,220))
        center_x, center_y = get_polygon_center(spot[i])
        cv2.putText(output, f"Zone {i+1}", (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
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

def display_zone_info(video_frame, zone_data,total_objects, x=20, y=50):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)
    font_thickness = 1
    line_spacing = 30
    transparent_box_alpha = 0.5  # Adjust transparency here
    old_y = y
    all_class_names = list(total_objects.values())
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
            box_width = max(text_size[0], cv2.getTextSize(fix_width_box, font, font_scale, font_thickness)[0][0]) 
            # Calculate the total height required for text within the box
            box_height = text_size[1] + 15 + (len(set(all_class_names)) + 1) 

            # Draw the transparent box
            box_coords = [(x, y-5), (x + box_width, y-5), (x + box_width, y + box_height), (x, y + box_height)]
            overlay = video_frame.copy()
            cv2.fillPoly(overlay, [np.array(box_coords, np.int32)], (200, 200, 200))
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
        cv2.fillPoly(overlay, [np.array(box_coords, np.int32)], (200, 200, 200))
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
        for i, object_class in enumerate(unique_classes):
            object_count = object_classes.count(object_class)
            text = f"{object_class}: {object_count}"
            y += 19
            cv2.putText(
                video_frame,
                text,
                (x + 5, y + text_size[1] + 20),
                font,
                font_scale,
                font_color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )
        
        # Increment y for the next zone's information
        y += box_height + line_spacing-20

    return video_frame