# import the necessary packages
from mytracker import AI_Tracker
from collections import OrderedDict 
from imutils.video import FPS
from math import floor
import numpy as np
import argparse
import imutils
import time
import cv2


# construct the argument parse and parse the arguments
def main(select_objectID=[4],is_tracking=None,is_localization=None,is_speed=None,is_select_frames=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt",
                    default='caffe_models/MobileNetSSD_deploy.prototxt.txt',
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model",
                    default='caffe_models/MobileNetSSD_deploy.caffemodel',
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    select_CLASSES=["car", "person"]
    #select_CLASSES=['person']
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture('/home/gray/gray_projects/deep learning for gray/gray_cnn_database/红外视频/720P-60/20480001.MOV')
    # init the centroid tracker
    ct=AI_Tracker()
    time.sleep(2.0)
    fps = FPS().start()
    frame_counter=0
    centroids_mask = OrderedDict()
    select_objectID=select_objectID
    # intit the p1 of cv2.line
    p1 = None
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame=frame[1]
        frame_counter += 1
        if frame is None:
            break
        frame = imutils.resize(frame, width=400)
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        rects=[]
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if CLASSES[idx] in select_CLASSES and confidence > args["confidence"]:

                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                rects.append(box.astype('int'))
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2)
        objects= ct.update_centroid(rects)
        ct.update_speed()
        ct.update_localization()
        objects_centroids_buffer=ct.objects_centroids_buffer
        # objects_bbox_buffer=ct.objects_bbox_buffer
        # standardize the camera parameters,the block should decomments when first launch program
        # if len(objects_bbox_buffer)!=0:
        #     if select_objectID[0] in objects.keys() and len(objects_bbox_buffer[ct.selectObjectID])>=10:
        #         ct.camera_standardize()
        # if len(objects)>0:
        #     print('the select objects({}) appear at {} frame'.format(select_CLASSES[0],frame_counter))
        j=floor(frame_counter/2)
        # update the trace of objects every 2 frames
        if frame_counter%2==0:
            if is_tracking is not None:
                if j==1:
                    for (objectID,centroids) in objects_centroids_buffer.items():
                        if objectID in select_objectID:
                            p1=centroids[0]
                            centroids_mask[objectID]=[p1]
                            centroids_mask[objectID].append(centroids[-1])
                            for k in range(len(centroids_mask[objectID])-1):
                                cv2.line(frame, tuple(centroids_mask[objectID][k]), tuple(centroids_mask[objectID][k+1]),(0, 0, 255), 3)
                else:
                    for (objectID,centroids) in objects_centroids_buffer.items():
                        if objectID in select_objectID and objectID in objects.keys():
                            if objectID in centroids_mask.keys():
                                centroids_mask[objectID].append(centroids[-1])
                                for k in range(len(centroids_mask[objectID])-1):
                                    cv2.line(frame, tuple(centroids_mask[objectID][k]), tuple(centroids_mask[objectID][k+1]),(0, 0, 255), 3)
                            else:
                                p1 = centroids[0]
                                centroids_mask[objectID] = [p1]
                                centroids_mask[objectID].append(centroids[-1])
                                for k in range(1):
                                    cv2.line(frame, tuple(centroids_mask[objectID][k]), tuple(centroids_mask[objectID][k+1]),(0, 0, 255), 3)

        for (objectID,centroid) in objects.items():
            text='ID:{}'.format(objectID)
            cv2.putText(frame,text,(centroid[0]-10,centroid[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            cv2.circle(frame,(centroid[0],centroid[1]),4,(0,255,0),-1)
        for (objectID,centroid) in objects.items():
            if is_select_frames is not None:
                if objectID in select_objectID and objectID in objects.keys():
                    text='the select object(ID:{}) appears in the {} frame'.format(objectID,frame_counter)
                    cv2.putText(frame,text,(1,10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        # update the speed of selected objects
        for (objectID,velocity) in ct.objects_velocity.items():
            if is_speed is not None:
                if objectID in select_objectID and objectID in objects.keys():
                    text='velocity:{}km/h'.format(int(velocity*3.6))
                    print('velocity:{}km/h'.format(int(velocity * 3.6)))
                    cv2.putText(frame,text,(objects[objectID][0]-20,objects[objectID][1]+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        # update the position of selected objects
        for (objectID,position) in ct.objects_position.items():
            if is_localization is not None:
                if objectID in select_objectID and objectID in objects.keys():
                    text='SE-{},DIS-{}m'.format(int(position[0]), int(position[1]))
                    cv2.putText(frame,text,(objects[objectID][0]-20,objects[objectID][1]+40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print(len(ct.objects_bbox_buffer[select_objectID[0]]))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(is_speed=True,is_localization=True)
