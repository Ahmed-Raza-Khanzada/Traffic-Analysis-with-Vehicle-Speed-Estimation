from detection_helpers import *
from tracking_helpers import *
import argparse

<<<<<<< HEAD


=======
>>>>>>> 20b512d9dfc3886a917bda147a3e376063bda927
from  bridge_wrapper import *
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] ="0" # comment out below line to enable tensorflow logging outputs
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

 
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--output_video',  type=str, help='Output video path')
	parser.add_argument('--input_video',  type=str,  help='Input video path')
	parser.add_argument('--coor',  type=str, default='./coors/coor.txt', help='video zones path')
	parser.add_argument('--tracker',  type=str, default='./deep_sort/model_weights/mars-small128.pb', help='video Tracker path')
	parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path(s)')
	parser.add_argument('--show', type=bool, default=False, help='Show video')

	opt = parser.parse_args()
	detector = Detector(classes = [0,1,2,3,5,7]) # it'll detect ONLY [person,horses,sports ball]. class = None means detect all classes. List info at: "data/coco.yaml"
	detector.load_model(opt.weights) # pass the path to the trained weight file
	# Initialise  class that binds detector and tracker in one class
	tracker = YOLOv7_DeepSORT(reID_model_path=opt.tracker, detector=detector)
	# output = None will not save the output video
	tracker.track_video(opt.input_video, output=opt.output_video, show_live =opt.show, skip_frames = 0, count_objects = True, verbose=1,coor_path=opt.coor)