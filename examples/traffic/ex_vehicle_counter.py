import imageio
import numpy as np
import cv2
import traceback

from cr import vision

from cr.vision.traffic import TrafficCounter

VIDEO_SOURCE = r'road_traffic_video_karol_majek.mp4'


counter = TrafficCounter(source=VIDEO_SOURCE)
writer = imageio.get_writer('vehicles.mp4', fps=20)

counter.warmup()
counter.next_frame()
frame = counter.draw_vehicles()



count = 0
while True:
    try:
        cv2.imshow('Sliding Windows', frame)
        key = cv2.waitKey(5) & 0xFF
        count += 1
        print("+", end="", flush=True)
        frame = vision.bgr_to_rgb(frame)
        writer.append_data(frame)
        counter.next_frame()
        frame = counter.draw_vehicles()
    except:
        traceback.print_exc()
        break
writer.close()
key = cv2.waitKey(0) & 0xFF
