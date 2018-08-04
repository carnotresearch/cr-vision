'''
Base class for thread based active nodes
'''
import logging
from threading import Thread

#pylint: disable=W0703

class ActiveNode:
    '''Executes some operations in a thread'''
    name = ''
    _stopped = False

    def __init__(self, name='Node'):
        self.name = name
        self._stopped = False
        self.thread = Thread(target=self._mainloop, name=self.name, args=())
        self.thread.daemon = True

    def start(self):
        '''Starts the thread'''
        self.thread.start()

    def stop(self):
        '''Indicates that the thread should stop'''
        self._stopped = True
        self.thread.join()

    def _mainloop(self):
        # preparation work at the beginning of the thread
        self._begin()
        # If an exception is thrown at the beginning, we won't continue
        while True:
            if self._stopped:
                # stop execution
                break
            try:
                # Next step in the thread
                self._operation()
            except Exception:
                logging.exception(self.name)
        try:
            # release any resources at the end
            # end is always called
            self._end()
        except Exception:
            logging.exception(self.name)

    def _begin(self):
        '''Some operation at the beginning of the thread'''
        logging.info('%s is being started.', self.name)

    def _operation(self):
        raise NotImplementedError

    def _end(self):
        '''Some operation at the end of the thread'''
        logging.info('%s is being finished.', self.name)


class ActiveMap(ActiveNode):
    '''Maps data from input queue to output queue'''

    def __init__(self, input_queue, output_queue, map_function, name='Map'):
        super().__init__(name)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.map_function = map_function

    def _operation(self):
        data = self.input_queue.get(timeout = 1)
        data = self.map_function(data)
        self.output_queue.put(data)
