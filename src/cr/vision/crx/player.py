import asyncio
import cv2
import datetime

class EventLoopPlayer:

    def __init__(self, loop, window_name="Display", stop_on_esc=True):
        self.loop = loop
        self.window_name = window_name
        self.stop_on_esc = stop_on_esc

    def on_next(self, image):
        print(datetime.datetime.now())
        loop = self.loop
        future = asyncio.run_coroutine_threadsafe(self.show_coroutine(image), loop)
        if not self.stop_on_esc:
            return
        key = future.result()
        if key == ord('q') or key == 27:
            loop.stop()


    def on_completed(self):
        # Make sure that the window created by the player is destroyed
        cv2.destroyAllWindows()
        return

    def on_error(self):
        return

    async def show_coroutine(self, image):
        cv2.imshow(self.window_name, image)
        return cv2.waitKey(1) & 0xff


