#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

def limits():
    # Generated from 18 layer stylegan (x1024)
    # Was done on the fly previously, but this is easier for UI.
    # If the architecture changes, just run any input up to S space and fill in the input sizes.
    num_s = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32]
    return {
        'layer': len(num_s),
        'channel': num_s
    }

def known_presets():
    return {
        'gaze': (9, 409),
        'smile': (6, 259),
        'eyebrows_1': (8, 28),
        'eyebrows_2': (12, 455),
        'eyebrows_3': (6, 179),
        'hair_color': (12, 266),
        'fringe': (6, 285),
        'lipstick': (15, 45),
        'eye_makeup': (12, 414),
        'eye_roll': (14, 239),
        'asian_eyes': (9, 376),
        'gray_hair': (6, 364),
        'eye_size': (12, 110),
        'goatee': (9, 421),
        'fat': (6, 104),
        'gender': (6, 128),
        'chin': (6, 131),
        'double_chin': (6, 144),
        'sideburns': (6, 234),
        'forehead_hair': (6, 322),
        'curly_hair': (6, 364),
        'nose_up_down': (9, 86),
        'eye_wide': (9, 63),
        'gender_2': (9, 6),
        'demon_eyes': (14, 319),
        'sunken_eyes': (14, 380),
        'pupil_1': (14, 414),
        'pupil_2': (14, 419),
    }