import cv2
import os

from os import listdir

'''
for i in range(9,10):
    print(i)

    output_fname = "/home/potetsos/skule/2021Host/visuelInt/eksamen/data/LiDAR-videos/Video0000"+str(i)+"_Ano/video"+str(i)+"AIR.mp4"
    frames_per_second = 20
    codec, file_ext = ("mp4v", ".mp4")

    output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    fourcc=cv2.VideoWriter_fourcc(*codec),
                    fps=float(frames_per_second),
                    #frameSize=(3840, 1080),
                    frameSize=(1024, 128),
                    #frameSize=(1536, 4096),
                    #frameSize=(2048, 540),
                    isColor=True,
            )

    #output_file = cv2.VideoWriter(output_fname, 0, 1, (4096,1536))

    print(output_file)
    startPath = "/home/potetsos/skule/2021Host/visuelInt/eksamen/data/LiDAR-videos/Video0000"+str(i)+"_Ano/combined"
    files = listdir(startPath)
    files.sort()
    print(files)

    index = 0
    for filename in files:
        print(index)
        print(filename)
        #frame = read_image(startPath + "/" +filename, 'RGB')
        frame = cv2.imread(startPath + "/" +filename)
        #print(frame.shape)
        output_file.write(frame)
        index +=1
        #cv2.imwrite(filename, frame)
        #video.write(cv2.imread(os.path.join(image_folder, image)))
'''



output_fname = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/alleLabled/maskFormer000002.mp4"
frames_per_second = 30
codec, file_ext = ("mp4v", ".mp4")

output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                #frameSize=(3840, 1080),
                frameSize=(4096, 1536),
                #frameSize=(1536, 4096),
                #frameSize=(2048, 540),
                isColor=True,
        )

#output_file = cv2.VideoWriter(output_fname, 0, 1, (4096,1536))

print(output_file)
startPath = "/home/potetsos/skule/2021Host/prosjekt/dataFFI/ffiBilder/alleLabled/000002"
files = listdir(startPath)
files.sort()
print(files)

index = 0
for filename in files:
    print(index)
    print(filename)
    #frame = read_image(startPath + "/" +filename, 'RGB')
    frame = cv2.imread(startPath + "/" +filename)
    #print(frame.shape)
    output_file.write(frame)
    index +=1
    #cv2.imwrite(filename, frame)
    #video.write(cv2.imread(os.path.join(image_folder, image)))