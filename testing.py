from horse.yolo_top import YOLOtop
from image2vec.yolo import YOLO

net = YOLOtop(
	'image2vec/yolo-full.cfg', 
	'image2vec/yolo-full.weights',
	from_layer = 28)

# net = YOLO(
# 	'image2vec/yolo-full.cfg', 
# 	'image2vec/yolo-full.weights',
# 	up_to = 100)