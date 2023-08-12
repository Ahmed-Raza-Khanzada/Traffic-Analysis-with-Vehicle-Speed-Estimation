from tkinter import Frame
import cv2
import numpy as np
import os.path
import math
import argparse
from utilities import create_coor_file
from utilities import make_poly


coors= []
points = []
def Click(events,x,y,flags,params):
	if events==cv2.EVENT_LBUTTONDOWN:
		points.append((x,y))
	elif events==cv2.EVENT_RBUTTONDOWN :
		for pos,coor in enumerate(coors):
			result = cv2.pointPolygonTest(np.array(coors[pos],np.int32),(x,y),False)
			if result>=0:
				print(coors[pos],"Removed")
				coors.pop(pos)
if __name__=="__main__":
		
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_video',  type=str, help='Input video path zones declaration')
	parser.add_argument('--coor',  type=str, help='video zones path name of file')

	opt = parser.parse_args()
	cv2.namedWindow("Image", cv2.WINDOW_NORMAL)	
	# Using resizeWindow()
	cv2.resizeWindow("Image", 1500, 700)
	cap = cv2.VideoCapture(opt.input_video) #cap.read()
	_,img2 = cap.read()
	from_video = False
	while True:
		if from_video:
			_,img = cap.read()
		else:
			img = img2.copy()
		img = np.array(img)
		if len(coors)>0:
			img = make_poly(img,coors)
		if len(points)>0:
			for i in points:
				cv2.circle(img, i, 5, (0, 0, 255), -1)

		cv2.imshow("Image",img)
		cv2.setMouseCallback("Image",Click)

		if len(points)==4:
			coors.append(points)
			points = []
		if cv2.waitKey(120) == ord("q"):
			print("************Final Coordinates Added****************")
			print(coors)
			create_coor_file(coor=coors,path=opt.coor)
			break

