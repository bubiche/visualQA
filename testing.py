from image2vec.yolo import YOLO

path = './image2vec/'
cfg = path + 'yolo-full.cfg'
wgt = path + 'yolo-full.weights'
net = YOLO(cfg, wgt, 100)