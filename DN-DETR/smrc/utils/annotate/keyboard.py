# Left: 2424832
# Up: 2490368
# Right: 2555904
# Down: 2621440
import cv2
import sys


def get_platform():
    platforms = {
        'linux': 'Linux',
        'linux1': 'Linux',
        'linux2': 'Linux',
        'darwin': 'OS X',  # Mac OS X
        'win32': 'Windows'
    }
    if sys.platform not in platforms:
        return sys.platform
    return platforms[sys.platform]


class KeyboardCoding:
    def __init__(self):
        # initialize the system platform (Linux, Windows, ...)
        self.platform = get_platform()
        self.DELAY = 50  # keyboard delay (in milliseconds)

        self.keyboard = {}
        if self.platform == 'Windows':
            # ================================
            # for cv2.waitKeyEx()
            # cv2.waitKey() cannot recognize F1 ~ F12 (and Ins, Del keys for note computer)
            # ================================
            # arrow key
            self.keyboard['LEFT'] = 2424832
            self.keyboard['UP'] = 2490368
            self.keyboard['RIGHT'] = 2555904
            self.keyboard['DOWN'] = 2621440

            # edit key
            self.keyboard['HOME'] = 2359296
            self.keyboard['PAGEUP'] = 2162688
            self.keyboard['PAGEDOWN'] = 2228224
            self.keyboard['END'] = 2293760

            self.keyboard['DELETE'] = 3014656
            self.keyboard['INSERT'] = 2949120
            self.keyboard['SPACE'] = 32
            self.keyboard['TAB'] = 9
            self.keyboard['BACKSPACE'] = 8

            ''' F1 ~ F5
            pressed_key= 7340032 pressed_key & 0xFF = 0
            pressed_key= 7405568 pressed_key & 0xFF = 0
            pressed_key= 7471104 pressed_key & 0xFF = 0
            pressed_key= 7536640 pressed_key & 0xFF = 0
            pressed_key= 7602176 pressed_key & 0xFF = 0
            '''
            self.keyboard['F1'] = 7340032
            self.keyboard['F5'] = 7602176

            """ F9 - F12
            pressed_key= 7864320 pressed_key & 0xFF = 0
            pressed_key= 7929856 pressed_key & 0xFF = 0
            pressed_key= 7995392 pressed_key & 0xFF = 0
            pressed_key= 8060928 pressed_key & 0xFF = 0
            """
            self.keyboard['F9'] = 7864320
            self.keyboard['F10'] = 7929856
            self.keyboard['F11'] = 7995392
            self.keyboard['F12'] = 8060928

            # self.keyboard['SHIFT'] = 65505  # not recognizable in Windows, but combination key (Shift + o) is OK
            # self.keyboard['ALT'] = 65513  # not recognizable in Windows
            self.keyboard['-'] = 45  # tested
            self.keyboard['+'] = 43  # tested
        elif self.platform == 'Linux':
            # ================================
            # cv2.waitKeyEx()
            # ================================
            # arrow key
            self.keyboard['LEFT'] = 65361
            self.keyboard['UP'] = 65362
            self.keyboard['RIGHT'] = 65363
            self.keyboard['DOWN'] = 65364

            # edit key
            self.keyboard['HOME'] = 65360
            self.keyboard['PAGEUP'] = 65365
            self.keyboard['PAGEDOWN'] = 65366
            self.keyboard['END'] = 65367

            # not delete key, but break key
            self.keyboard['DELETE'] = 65535
            self.keyboard['INSERT'] = 65379
            self.keyboard['TAB'] = 8  # 65513
            # can not be recognized by Linux, so we use 'ALT'

            self.keyboard['BACKSPACE'] = 8
            self.keyboard['SPACE'] = 32

            self.keyboard['F1'] = 65470
            self.keyboard['F5'] = 65474

            """
            pressed_key= 65478 pressed_key & 0xFF = 198
            pressed_key= 65479 pressed_key & 0xFF = 199
            pressed_key= 65480 pressed_key & 0xFF = 200
            pressed_key= 65481 pressed_key & 0xFF = 201
            """
            self.keyboard['F9'] = 65478
            self.keyboard['F10'] = 65479
            self.keyboard['F11'] = 65480
            self.keyboard['F12'] = 65481

            self.keyboard['SHIFT'] = 65505
            # self.keyboard['ALT'] = 65513
            self.keyboard['-'] = 45
            self.keyboard['+'] = 43
            # do not use cv2.waitKey(), as there are few available keys
            # compared with waitKeyEx())
            # self.keyboard['LEFT'] = 81
            # self.keyboard['UP'] = 82
            # self.keyboard['RIGHT'] = 83
            # self.keyboard['DOWN'] = 84
            #
            # # edit key
            # self.keyboard['HOME'] = 149
            # self.keyboard['PAGEUP'] = 154
            # self.keyboard['PAGEDOWN'] = 155
            # self.keyboard['END'] = 156
            #
            # # not delete key, but break key
            # self.keyboard['DELETE'] = 8
            # self.keyboard['INSERT'] = 99
            # self.keyboard['SPACE'] = 32
            #
            # self.keyboard['F5'] = 194
            #
            # self.keyboard['SHIFT'] = 226
            # self.keyboard['ALT'] = 233
            # # self.keyboard['CTRL'] = None # Not recognizable

    def read_pressed_key(self):
        ''' Key Listeners'''
        # to use arrow key, we have to use cv2.waitKeyEx() for Windows system, otherwise, the array key is not recognizable
        if self.platform == 'Windows':
            pressed_key = cv2.waitKeyEx(self.DELAY)  # cv2.waitKeyEx(self.DELAY)
        elif self.platform == 'Linux':  # waitKeyEx(self.DELAY) & 0xFF is same to waitKey(self.DELAY)
            # do not use cv2.waitKey(), as there are few available keys
            # compared with waitKeyEx())
            pressed_key = cv2.waitKeyEx(self.DELAY)
        else:  # other system
            pressed_key = cv2.waitKey(self.DELAY)
        # print('pressed_key=', pressed_key)  # ('pressed_key=', -1) if no key is pressed.
        return pressed_key
