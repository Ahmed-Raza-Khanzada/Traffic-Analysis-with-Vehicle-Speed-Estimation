'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
'''

import os
import random as rnd
import string
import time
import tensorflow as tf
from keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES'] ="0" # comment out below line to enable tensorflow logging outputs
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
from class_predict import predict_class_mymodel
from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *
from utilities import make_poly , get_coordinates,estimateSpeed,fancy_bbox,count_time_in_video_frame,display_zone_info,replace_zone_name
import os
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    

 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True


class YOLOv7_DeepSORT:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, reID_model_path:str, detector, max_cosine_distance:float=0.8, nn_budget:float=None, nms_max_overlap:float=0.92,
    coco_names_path:str ="./io_data/input/classes/coco.names",  ):#nms_max_overlap:float=0.92
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.detector = detector
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker


    def track_video(self,video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0,coor_path="/coors/coor.txt"):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
        '''
        try: # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)
        initial_fps = vid.get(cv2.CAP_PROP_FPS)
        
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/initial_fps


        minutes = int(duration/60)
        seconds = duration%60

        out = None
        if output: # get video ready to save locally if flag is set
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output, codec, fps, (width, height))
        areas = get_coordinates(path=coor_path)
        area_names = []
        for i in range(len(areas)):
            area_names.append("Area"+str(i))
        frame_num = 0

        entries = {}
        zones = {}
        zones_trace = {}
        total_objects = {}
        current_frame = 0
        count_time = 0
        prev_timestamp = None
        # Load the model
        my_class_model = load_model("classification_model/keras_Model.h5", compile=False)
        # Load the labels
        my_class_model_names = open("classification_model/labels.txt", "r").readlines()
        while True: # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            prev_timestamp = count_time_in_video_frame(return_value,vid, prev_timestamp)
            frame_num +=1

            if skip_frames and not frame_num % skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
            if verbose >= 1:start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            yolo_dets = self.detector.detect(frame.copy(), plot_bb = False)  # Get the detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_1 = frame.copy()

            if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0
            
            else:
                bboxes = yolo_dets[:,:4]
                bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
                bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

                scores = yolo_dets[:,4]
                classes = yolo_dets[:,-1]
                num_objects = bboxes.shape[0]
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            
            names = []
            for i in range(num_objects): # loop through objects and use class index to get class name
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)

            names = np.array(names)
            count = len(names)

            # if count_objects:
            #     cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            self.tracker.predict()  # Call the tracker
            self.tracker.update(detections) #  updtate using Kalman Gain

            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                x,y,x2,y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                try:
                    class_name = track.get_class()#predict_class_mymodel(frame_1[y:y2,x:x2],my_class_model_names,my_class_model)#
                except:
                    continue
                cx,cy = int((x + x2)/2),int((y+y2)/2)
                object_id = track.track_id
                
                for area_no,area in enumerate(areas):
                    result = cv2.pointPolygonTest(np.array(area,np.int32),(cx,cy),False)
                    speed = False
                    if result>=0:
                        print("In Area")	
                    # cv2.circle(img2, pt[0], 5, (0, 0, 255), -1)
                    # cv2.rectangle(img2, (pt[-1][0],pt[-1][1]), (pt[-1][2],pt[-1][3]), (204,207,76), 2)
                        print("Frame NUMBER",frame_num)
                        
                        if object_id not in entries.keys() :
                            print(frame_num,"Frame no")
                            speed = 0
                    
                           
                            entries[object_id] = ((cx,cy),class_name,area_names[area_no],speed)
                        
                        if frame_num>2:
                                
                            if object_id in entries.keys():
                                # print("Object Id Exist")
                                speed = estimateSpeed(initial_fps,entries[object_id][0],(cx,cy))

                                # if area_no==1:
                                #     print("*"*25,object_id,"Speed Detected",speed)
                                # if speed<6:
                                #     speed = 0
                               
                                # try:
                                #     random_id = ''.join(rnd.choices(string.ascii_uppercase + string.digits, k=7))
                                #     while random_id  in os.listdir('H:/upwork/vehicle-submission/vehicle-speed-estimation-v7/output_images_classification/arrange_data'):
                                #         random_id = ''.join(rnd.choices(string.ascii_uppercase + string.digits, k=7))
                                #     cv2.imwrite( f'H:/upwork/vehicle-submission/vehicle-speed-estimation-v7/output_images_classification/arrange_data/{random_id}.jpg',frame_1[y:y2,x:x2])
                                # except:
                                #     pass
                               
                                old_values  = entries[object_id][1:]
                                print(object_id,class_name,"area name official",area_names[area_no],"old value expected area",old_values[1],speed,"$"*25)
                                entries[object_id] = ((cx,cy),old_values[0],area_names[area_no],speed)
                                
                        break
                color = (245, 105, 12)
                
                if object_id in entries.keys():
                    speed2 = entries[object_id][-1]
                    speed2 = round(speed2,2)
                    speed2 = str(speed2).capitalize()+"KM/H"
                    
                    still_area = False
                    for area_no,area in enumerate(areas):
                        result = cv2.pointPolygonTest(np.array(area,np.int32),(cx,cy),False)
                       
                        if result>=0:
                            still_area = True
                            break
                    if not still_area:
                        new_id  = entries[object_id][2]
                       
                        old_clss = class_name
                        if object_id not in total_objects.keys():
                                total_objects[object_id] = class_name
                   
                        if count_time>60:
                            count_time1 = round(count_time/60,2)
                            count_time1 = str(count_time1)+" mins"
                        else:
                            count_time1 = str(count_time)+" secs"
                
                        
                        print(object_id,class_name,new_id,speed,"#"*25)

                        if zones_trace.get(new_id) is not None:
                            if object_id not in zones_trace[new_id]:
                                    zones_trace[new_id].append(object_id)
                                    zones[new_id].append([object_id,old_clss,count_time1,entries[object_id][2],entries[object_id][-1]])
                        else:
                            zones_trace[new_id] = [object_id]
                            zones[new_id] = [[object_id,old_clss,count_time1,entries[object_id][2],entries[object_id][-1]]]


                        
                else:
                    speed2 = "Not in zone"
               
                # print("Speed2",speed2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-15)), (int(bbox[0])+(len(class_name)+len(str(object_id))+len(speed2))*13, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name.capitalize()+", "+str(int(object_id))+", "+speed2, (int(bbox[0]), int(bbox[1]-3)), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
                frame = fancy_bbox(frame,(x,y,x2,y2))
            current_frame +=1
            if current_frame==int(initial_fps):
                current_frame =0
                count_time += 1
        
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            # if verbose >= 1:
            # print(zones,"********")
            frame = display_zone_info(frame, zones,total_objects,10,30)#10,140
            cr_fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
            if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(cr_fps,2)}")
            else: print(f"Processed frame no: {frame_num} || Current FPS: {round(cr_fps,2)} || Time {count_time} || Objects tracked: {count}")
            # cv2.putText(frame, "FPS: "+str(round(cr_fps,2)), (600,30), cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,0), 2)
           
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            result = make_poly(result,areas)
            if output: out.write(result) # save output video

            if show_live:
                cv2.imshow("Output Video", cv2.resize(result,(1350,750) )  )
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    print(zones)
                    break
        
        cv2.destroyAllWindows()
