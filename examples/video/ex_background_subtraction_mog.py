'''
Demo of background subtraction
'''
import cv2

def main(title='Background Subtraction'):
    ''' Background Subtraction Demo '''
    fg_bg = cv2.bgsegm.createBackgroundSubtractorMOG()
    cap = cv2.VideoCapture(0)
    while True:   
        _, frame = cap.read()
        foreground_mask = fg_bg.apply(frame)
        cv2.imshow(title, foreground_mask)
        k = cv2.waitKey(40) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()
