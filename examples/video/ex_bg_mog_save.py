'''
Capture and save live video with background subtraction
'''
import cv2
import numpy as np
import time



def main(title='Video Recorder'):

    # frame rate estimation
    # Start time
    start = time.time()
    cap = cv2.VideoCapture(0)
    num_frames = 10
    for i in range(num_frames):
        ret, frame = cap.read()
    # end time
    end = time.time()
    # time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))
    # Frame rate
    fps  = num_frames / seconds;
    print("Estimated frames per second : {0}".format(fps))

    output_file = 'video.avi'
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 12
    is_color = False
    size = (640,480)
    out = cv2.VideoWriter(output_file,fourcc, fps, size, is_color)
    ''' Background Subtraction Demo '''
    fg_bg = cv2.bgsegm.createBackgroundSubtractorMOG()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame2 = fg_bg.apply(frame)
            cv2.imshow(title, frame2)
            # frame = cv2.flip(frame,0)
            # write the flipped frame
            out.write(frame2)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        else:
            break
    cv2.destroyAllWindows()
    cap.release()
    out.release()

if __name__ == '__main__':
    main()
