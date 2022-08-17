# -*- coding = "utf-8" -*-

import numpy as np
class TrackState:
    Tentative =1
    Confirmed =2
    Deleted = 3



class Tracklet:
    def __init__(self,id,
                 kf,
                 mean,
                 covariance,
                 c1_id =-1,
                 c2_id = -1,
                 feature = None):
        self.id = id
        self.mean = mean
        self.covariance = covariance
        self.c1_id = c1_id
        self.c2_id = c2_id
        self.features =[]+[feature]
        self.kf = kf


        self.tracklet_len = 0
        self.max_age  = 5
        #self.n_init  = 0
        self.time_since_update =0
        self.state = TrackState.Tentative

    def predict(self):
        self.mean,self.covariance = self.kf.predict(self.mean,self.covariance)
        self.tracklet_len+=1
        self.time_since_update+=1

    def update(self,detection):
        self.mean,self.covariance =self.kf.update(self.mean,self.covariance,
                                                             detection.bbox)
        self.c1_id = detection.c1_id
        self.c2_id = detection.c2_id

        self.time_since_update =0
        self.features.append(detection.feature)

        if self.state ==TrackState.Tentative:
            self.state = TrackState.Confirmed



    def topview_location(self):
        # if self.state ==TrackState.Deleted:
        #     return np.array([-100,-100])
        point = self.mean[:2].copy()
        return point

    def mark_missed(self):
        self.c1_id = -1
        self.c2_id = -1
        #self.time_since_update +=1
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def max_age(self):
        return self.max_age