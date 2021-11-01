#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

from resources import resource_list


def pretrained_models():
    return {
        'afro': resource_list['styleclip_afro'],
        'bobcut': resource_list['styleclip_bobcut'],
        'mohawk': resource_list['styleclip_mohawk'],
        'bowlcut': resource_list['styleclip_bowlcut'],
        'curly_hair': resource_list['styleclip_curly_hair'],
    }