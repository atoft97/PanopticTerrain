import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(figsize=(15, 6))
import json

classes = []

plt.style.use('ggplot')

#x = ['car', 'motorcycle', 'person', 'truck', 'bicycle', 'rider', 'bus', 'scooter']
#energy = [9925, 0, 7242, 123, 570, 1577, 449, 0]
#energy = [608, 0, 348, 0, 165, 263, 83, 0]
#energy = [126, 0, 664, 0, 0, 46, 0, 0]

jsonFilePath = "/home/potetsos/skule/2021Host/compleateDataset/annotations/instances_Validation.json"
f = open(jsonFilePath)
data = json.load(f)

counts = [0 for i in range(13)]
categories = ["Forrest", "CameraEdge","Vehicle","Person","Bush","Puddle","Dirtroad","Sky","Large_stone","Grass","Gravel","Tree","Building"]

for annotation in data['annotations']:
	catID = annotation['category_id']
	counts[catID-1] += 1

print(counts)

x = ['All','car', 'bus',  'bicycle', 'person','rider']
#energy = [9925, 0, 7242, 123, 570, 1577, 449, 0]
#energy = [608, 0, 348, 0, 165, 263, 83, 0]
#energy = [0.552, 0.927, 0.506, 0.411, 0.303, 0.614]

x_pos = [i for i, _ in enumerate(categories)]

plt.bar(x_pos, counts, color='green')
plt.xlabel("Classes")
plt.ylabel("mAP")
#plt.title()

plt.xticks(x_pos, categories)

plt.show()
