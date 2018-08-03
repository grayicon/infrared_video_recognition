from scipy.spatial import distance as dis
from scipy.optimize import fsolve
import numpy as np
from math import pi,sqrt,atan
from collections import OrderedDict


class AI_Tracker():
    def __init__(self, selectObjectID=2, camera_velocity=0,maxDisappeared=10):
        self.objects_centroid = OrderedDict()
        self.objects_centroids_buffer=OrderedDict()
        self.objects_bbox_buffer=OrderedDict()
        self.objects_position=OrderedDict()
        self.objects_velocity=OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.selectObjectID=selectObjectID
        self.nextObjectID = 0
        self.L=3.0409374
        self.alpha= 0.92117816
        self.camera_velocity=camera_velocity

    def register(self, centroid, rect):
        self.objects_centroid[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.objects_centroids_buffer[self.nextObjectID]=[]
        self.objects_centroids_buffer[self.nextObjectID].append(centroid)
        self.objects_bbox_buffer[self.nextObjectID] = []
        self.objects_bbox_buffer[self.nextObjectID].append(rect)
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects_centroid[objectID]
        del self.disappeared[objectID]

    def update_centroid(self, rects):
        if len(rects) == 0:
            disappearedcopy=self.disappeared.copy()
            for objectID in disappearedcopy.keys():
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects_centroid

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects_centroid) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i],rects[i])



        else:
            objectIDs = list(self.objects_centroid.keys())
            objectCentroids = list(self.objects_centroid.values())
            D = dis.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects_centroid[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                self.objects_centroids_buffer[objectID].append(inputCentroids[col])
                self.objects_bbox_buffer[objectID].append(rects[col])
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])

        return self.objects_centroid

    def camera_standardize(self):

        '''note:the field of view set as 53 degree
           what i selectID:2 is a car ,the length of the car set as the experimental value 3m,
           after by iterazation 30 frames, recording the bounding box of the car(ID:2),we using the formula below about
           the relationship between the pixel distance and real objects distance
           the measuring value of L and alpha are respectively 3.0409374, 0.92117816
           :param alpha:camera slop angle
           :param L:the vertical distance between camera center and object '''
        n=200
        y=53*pi/180
        AB=3
        P_A = self.objects_bbox_buffer[self.selectObjectID][0][0]
        P_B = self.objects_bbox_buffer[self.selectObjectID][0][2]
        P_C = self.objects_bbox_buffer[self.selectObjectID][-1][0]
        P_D = self.objects_bbox_buffer[self.selectObjectID][-1][2]

        def f(x):
            return (n**2-n*np.tan(x)*np.tan(y)*(P_A+P_B-2*n)+(P_A-n)*(P_B-n)*np.tan(x)**2*np.tan(y)**2)\
                   /(n**2-n*np.tan(x)*np.tan(y)*(P_C+P_D-2*n)+(P_C-n)*(P_D-n)*np.tan(x)**2*np.tan(y)**2)-(P_B-P_A)/(P_D-P_C)
        root=fsolve(f,[.7])
        self.alpha=root
        L=AB*(n*2-n*np.tan(root)*np.tan(y)*(P_A+P_B-2*n)+(P_A-n)*(P_B-n)*np.tan(root)**2*np.tan(y)**2)/\
          ((P_B-P_A)*n*np.tan(y)*(1/np.cos(root))**2)
        self.L=L

    def update_localization(self):
        L=self.L
        alpha=self.alpha
        beta = 0.925
        n=200
        for (objectID,centroid) in self.objects_centroid.items():
            P_A = centroid[0]
            GA=L*(n*np.tan(alpha)-(n-P_A)*np.tan(beta))/(n+(n-P_A)*np.tan(alpha)*np.tan(beta))
            R=sqrt(L**2+GA**2)
            offset_angle=pi/2-alpha-atan(L/GA)
            angle=180+(offset_angle/pi)*180
            self.objects_position[objectID]=[angle,R]

    def update_speed(self):
        L = self.L
        alpha = self.alpha
        beta = 0.925
        n = 200
        for (objectID, centroids) in self.objects_centroids_buffer.items():
            if len(centroids)>=4:
                P_S = centroids[-4][0]
                P_E = centroids[-1][0]
                G_S = L * (n * np.tan(alpha) - (n - P_S) * np.tan(beta)) / (n + (n - P_S) * np.tan(alpha) * np.tan(beta))
                G_E = L * (n * np.tan(alpha) - (n - P_E) * np.tan(beta)) / (n + (n - P_E) * np.tan(alpha) * np.tan(beta))
                S=G_E-G_S
                #frames=len(self.objects_centroids_buffer[objectID])
                frames=3
                FPS=5581/93
                T=frames/FPS
                V=S/T+self.camera_velocity
                self.objects_velocity[objectID]=V
