import cv2
import numpy as np
import math

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
    # ppm = location2[2] / carWidht
    ppm = 20
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
def make_poly(img,spot):
	
	overlay = img.copy()
	
	alpha = 0.6
	output = img.copy()
	
	for i in range(len(spot)):
		output = cv2.fillPoly(img,[np.array(spot[i],np.int32)],(150,20,220))
	cv2.addWeighted(overlay, alpha, output, 1 - alpha,
			0, output)
 
	return output
def create_coor_file(coor,path="./coors/coor.txt"):
	textfile = open(path, "w")
	for element in coor:
		textfile.write(str(element) + "\n")
	textfile.close()