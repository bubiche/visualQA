from image2vec import yolo

net = yolo.YOLO(
	'image2vec/yolo-full.cfg', 
	'image2vec/yolo-full.weights',
	up_to = 29)

img_list = [
	'image2vec/person.jpg',
	'image2vec/dog.jpg'
]

vec = net.forward(img_list)
print(vec, vec.shape)