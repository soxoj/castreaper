#!/usr/bin/env python3
import asyncio
import argparse
import cv2
import json
import math
import os
import sys
import numpy as np
import re

from tqdm import tqdm
from PIL import Image
import pytesseract
from pytesseract import Output

QUEUE_LIMIT = 10


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 3)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


MODES = {
    'as much text as possible': (lambda x: x, '--psm 11'),
    'single uniform block': (lambda x: x, '--psm 6'),
    'single text line': (lambda x: x, '--psm 7'),
    'remove noise, single line': (remove_noise, '--psm 7'),
    'remove noise, single line': (opening, '--psm 7'),
    'remove noise, single line': (canny, '--psm 7'),
    'invert, single line': (lambda x: cv2.cvtColor(cv2.bitwise_not(x), cv2.COLOR_BGR2GRAY), '--psm 7'),
}


class Entity:
    def __init__(self, text, sec):
        self.text = text
        self.sec = sec

    def __repr__(self):
        return self.text

    def __str__(self):
        return f'{self.sec}: {self.text}'

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other):
        return self.text == other.text

    def __gt__(self, other):
        return self.text > other.text

    def __lt__(self, other):
        return self.text < other.text


class Box:
    def __init__(self, name):
        self.name = name
        self.coords = None
        self.preprocess = lambda x: x
        self.config = '--psm 11'


class TextEntitiesStorage:
    def __init__(self, trigger_words, trigger_regexps, entities_regexps):
        self.words = trigger_words
        self.t_regexps = trigger_words
        self.e_regexps = entities_regexps
        self.boxes = {}
        self.entities = set()

    def add_recognized_word(self, word, second):
        self.entities.add(Entity(word.strip(), second))

    def add_box(self, name, coords):
        b = Box(name)
        x = coords[0]
        y = coords[1]
        w = coords[2]
        h = coords[3]
        b.coords = (y-5, y+h+5, x-5, x+w+355)
        self.boxes[name] = b
        return b

    def match_entity(self, word):
        for w in self.words:
            if w in word:
                return True

        return False

    def get_entities_words(self):
        entities = set()
        words = set()
        for e in list(self.entities):
            added = False
            for r in self.e_regexps:
                result = re.search(r, e.text)
                if result:
                    entities.add(Entity(result.group(0), e.sec))
                    if result.group(0) != e.text:
                        # print(f'{result.group(0)} - {e}')
                        words.add(e)
                    added = True
                    break
            
            if not added:
                words.add(e)

        return list(entities), list(words)


def get_video_params(c):
    fps = c.get(cv2.CAP_PROP_FPS)

    c.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    ms = c.get(cv2.CAP_PROP_POS_MSEC)
    frames = c.get(cv2.CAP_PROP_POS_FRAMES) or v.get(cv.CAP_PROP_FRAME_COUNT)
    c.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

    if ms < 1:
        s = frames / fps
        ms = s * 1000

    return {
        'fps': int(fps),
        'ms_total': round(ms, 3),
        's_total': round(s, 3),
        'frames': int(frames),
    }


async def writer(frame_queue, queue):
    while True:
        if frame_queue.empty():
            # print('writer exited')
            break

        sec_str, text = await queue.get()
        if sec_str == 'exit':
            print('Exit from writing results')
            break
        with open('image_to_text.txt', 'a') as outputFile:
            outputFile.write(f'{sec_str},{text}\n')
        queue.task_done()
        await asyncio.sleep(0)


async def _recognize_frame(queue, text_queue, storage):
    sec_str, im_pil = await queue.get()
    if sec_str == 'exit':
        await text_queue.put(('exit', None))
        print('Exit from recognizing frames')
        return False

    img = np.asarray(im_pil)
    # img = cv2.cvtColor(cv2.bitwise_not(img), cv2.COLOR_BGR2GRAY)

    # gray = cv2.convertScaleAbs(img, 1.5, 1)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # extracting text from image frame
    if storage.boxes:
        for name, b in storage.boxes.items():
            print(f'Trying to extract URL from box "{name}""...')
            d = b.coords
            box_img = b.preprocess(img[d[0]:d[1], d[2]:d[3]])
            data = pytesseract.image_to_data(box_img, lang='eng', config=b.config, output_type=Output.DICT)

            s = ' '.join(data['text']).strip()
            conf = max(data['conf'])
            print(f'Text: {s}, confidence: {conf}')
            storage.add_recognized_word(s, sec_str)

            # cv2.imshow('img', box_img)
            # cv2.waitKey(0)

    text_in_image = pytesseract.image_to_data(img, lang='eng', config='--psm 11', output_type=Output.DICT)

    interesting_results = []
    text_list = text_in_image['text']

    for i, t in enumerate(text_list):
        if len(t) > 2:
            # save all long enough text to log
            interesting_results.append(t)

            # conf = int(text_in_image['conf'][i])
            # print(conf)
            if storage.match_entity(t):
                d = text_in_image
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                new_img = img[y-5:y+h+5, x-h:x+w+355]
                box_name = f'{x}:{y}:{w}:{h}'

                if not box_name in storage.boxes:
                    b = Box(box_name)
                    b.coords = (y-5, y+h+5, x-5, x+w+355)
                    storage.boxes[box_name] = b

                    print(f'Added new box {box_name}')

                    new_img = cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.imshow('img', new_img)
                    # cv2.waitKey(0)

                    source_text = t
                    print(f'Text: {t}')
                    max_len = len(t)
                    source = 'source'

                    for name, m in MODES.items():
                        try:
                            text = pytesseract.image_to_string(m[0](new_img), lang='eng', config=m[1])
                        except Exception as e:
                            print(f'Error: {e}')
                        else:
                            # print(f'{name} mode: {text}')
                            if len(text) > max_len:
                                max_len = len(text)
                                source = name
                                t = text

                    if source != 'source':
                        print(f'Found optimal preprocess mode: {source}')
                        b.preprocess = m[0]
                        b.config = m[1]
                        
                    storage.add_recognized_word(t, sec_str)

    text_in_image = ' '.join(interesting_results)

    # replacing \n and \f character
    text_in_image = text_in_image.replace('\n', '')
    text_in_image = text_in_image.replace('\f', '')

    # Removing unwanted spaces
    text_in_image = text_in_image.strip()

    # removing unicode characters
    text_in_image = text_in_image.encode('ascii', 'ignore').decode()

    await text_queue.put((sec_str, text_in_image))
    # appending output to dictionary (only those frames which contain text)
    # if(len(text_in_image) != 0):        
        # my_dict[str(frame)] = text_in_image

    queue.task_done()
    return True


async def recognize_frame(frame_queue, queue, text_queue, pbar, storage):
    while True:
        try:
            res = await _recognize_frame(queue, text_queue, storage)
        except KeyboardInterrupt:
            res = False

            frame_queue._queue.clear()

            while not frame_queue.empty():
                await frame_queue.get()
                frame_queue.task_done()

        if not res:
            # print('recognize frame exited')
            break

        await asyncio.sleep(0)
        pbar.update(1)


def get_sec_by_frame(fps):
    def calcucale(frame):
        sec = frame / fps
        minute_full = math.trunc(sec // 60)
        sec = math.trunc(sec - (minute_full * 60))
        return f'{minute_full}:{sec:02d}'
    return calcucale

async def put_frames(frame_queue, queue, pbar, video, path, calc_frame_fun, skip_frames):
    while True:
        if frame_queue.empty():
            await queue.put(('exit', None))
            # print(f'no frames, put_frames exited')
            break

        frame_number = await frame_queue.get()

        if frame_number == -1:
            await queue.put(('exit', None))
            print(f'End')
            break

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = video.read()
        if not ret:
            await queue.put(('exit', None))
            print(f'Exit from reading video, invalid frame {frame_number}')
            break

        pbar.update(1)
        im_pil = Image.fromarray(frame)
        await queue.put((calc_frame_fun(frame_number), im_pil))

        frame_queue.task_done()
        await asyncio.sleep(0)


async def run_all(processes):
    await asyncio.gather(*processes)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('./screenripper.py <VIDEOFILE NAME>')
        sys.exit(-1)

    parser = argparse.ArgumentParser(description='Screenripper')
    parser.add_argument('video_filename', type=str, help='Filename')
    parser.add_argument(
        '-s',
        '--skip-frames',
        dest='skip_frames',
        type=int,
        default=24,
        help='Count of frames to skip')

    args = parser.parse_args()

    filename = args.video_filename
    output_path = filename + '_output'

    video = cv2.VideoCapture(sys.argv[1])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    info = get_video_params(video)
    fps = info.get("fps")

    print(f'Video: {filename}')
    print(f'Framerate: {fps}')
    print(f'Total frames: {info.get("frames")}')
    print(f'Total seconds: {info.get("s_total")}')

    print(f'Starting to extract and recognize text on each {args.skip_frames} frame...')

    queue = asyncio.Queue(QUEUE_LIMIT)
    frame_queue = asyncio.Queue()
    text_queue = asyncio.Queue()
    calc_frame_fun = get_sec_by_frame(info.get("fps"))

    frame_count = info.get("frames")

    frame_queue.put_nowait(1)
    frame_queue.put_nowait(frame_count - 1)
    frame_queue.put_nowait(1 + fps)
    frame_queue.put_nowait(frame_count - fps - 1)

    for steps in [3, 5]:
        step = int(frame_count // steps)
        for i in range(1, steps):
            frame_queue.put_nowait(1 + i*step)

    frame_queue.put_nowait(-1)
    pbar = tqdm(total=frame_queue.qsize()*2, file=sys.stdout)

    trigger_words = [
        'http',
        '.com',
        '//',
    ]
    trigger_regexps = []
    entities_regexps = [
        r'https?:\/\/\S+',
        r'\w+\.\w+\/\S+',
        r'\S+@\S+\.\S+',
    ]

    storage = TextEntitiesStorage(trigger_words, trigger_regexps, entities_regexps)
    # # storage.add_box('URL address bar', (143, 48, 79, 12))
    # b = storage.add_box('URL address bar 2', (120, 90, 530, 15))
    # # b.preprocess = MODES['remove ']
    # b.config = '--psm 7'

    processes = [
                    writer(frame_queue, text_queue), # should be 1
                    put_frames(frame_queue, queue, pbar, video, output_path, calc_frame_fun, args.skip_frames), # should be 1
                    recognize_frame(frame_queue, queue, text_queue, pbar, storage),
                ]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(run_all(processes))
    finally:
        pbar.close()
        video.release()
        cv2.destroyAllWindows()

    entities, words = storage.get_entities_words()
    print('\nRecognized entities:\n'+'\n'.join(str(w) for w in sorted(entities)))
    print('\nRaw entities and other words:\n'+'\n'.join(str(w) for w in sorted(words)))
