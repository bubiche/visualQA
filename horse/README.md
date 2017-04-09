HorseNet:

YOLOv1 - based augmented with attention to count the number of horse.

Basically:

```
trainable_patch[32x32x3] -> YOLOv1 -> ref_vector[1x1x1024] 
img[448x448x3] -> YOLOv1 -> feature_vectors[7x7x1024]

feature_vectors * cosine_similarity(ref_vector, feature_vectors)
-> focused_vectors

focused_vectors -> conv() -> [3x3x2048]
-> reduce_sum() -> [2048] -> FC() -> 1 
```