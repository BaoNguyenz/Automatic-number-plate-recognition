# sort/sort.py

from __future__ import print_function
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def iou_batch(bb_test, bb_gt):
  """
  Tính IoU (Intersection over Union) giữa các bounding box.
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  

def convert_bbox_to_z(bbox):
  """
  Chuyển bounding box [x1,y1,x2,y2] thành dạng đo lường z [x,y,s,r] cho bộ lọc Kalman,
  trong đó x,y là tâm, s là scale (diện tích), và r là tỷ lệ khung hình.
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
  """
  Chuyển từ trạng thái dự đoán x của bộ lọc Kalman về lại dạng bounding box [x1,y1,x2,y2].
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker(object):
  """
  Lớp này đại diện cho việc theo dõi một đối tượng duy nhất.
  """
  count = 0
  def __init__(self, bbox):
    # Khởi tạo bộ lọc Kalman
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. 
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self, bbox):
    """
    Cập nhật trạng thái bằng bounding box quan sát được.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Tiến hành dự đoán trạng thái tiếp theo và trả về bounding box dự đoán.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update > 0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Trả về bounding box hiện tại.
    """
    return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
  """
  So khớp các detections mới với các trackers đã có bằng IoU.
  Trả về các cặp đã khớp, các detection không khớp, và các tracker không khớp.
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      # Sử dụng thuật toán Hungarian để tìm cặp tối ưu
      row_ind, col_ind = linear_sum_assignment(-iou_matrix)
      matched_indices = np.array(list(zip(row_ind, col_ind)))
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]] < iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Cài đặt các tham số cho SORT.
    max_age: Số frame tối đa một tracker tồn tại mà không được cập nhật.
    min_hits: Số lần khớp tối thiểu để một tracker được coi là hợp lệ.
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Hàm chính để cập nhật trạng thái tracking với các detection mới.
    `dets` là một mảng numpy [[x1,y1,x2,y2,score],...]
    """
    self.frame_count += 1
    
    # BƯỚC 1: DỰ ĐOÁN
    # Lấy các bounding box dự đoán từ các tracker hiện có
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    
    # BƯỚC 2: SO KHỚP
    # So khớp các detection mới với các tracker đã dự đoán
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # BƯỚC 3: CẬP NHẬT
    # Cập nhật các tracker đã được khớp với detection tương ứng
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # Tạo tracker mới cho các detection không được khớp
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        # Chỉ trả về các tracker đã "trưởng thành" (đủ min_hits) và chưa quá "già" (time_since_update)
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) 
        i -= 1
        
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
          
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))