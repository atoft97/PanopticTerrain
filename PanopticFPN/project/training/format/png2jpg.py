from PIL import Image
import json
import os

for s in ["train", "test", "validation"]:
    with open('annotations/instances_'+s+'.json') as json_file:
        data = json.load(json_file)
        for json_img in data['images']:
            fileName = json_img['file_name']
            print(fileName)
            if (fileName.endswith('.png')):
                img = Image.open("images"+"/"+fileName)
                newFIleName = fileName.replace(".png", ".jpg")
                img.save("/lhome/asbjotof/work/project/training/dataset_"+s+"/images"+"/"+newFIleName)

                json_img['file_name'] = newFIleName

    with open('/lhome/asbjotof/work/project/training/dataset_'+s+'/annotations/instances_'+s+'.json', 'w') as outfile:
        json.dump(data, outfile)

