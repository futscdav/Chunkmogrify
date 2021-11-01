#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

import torch

# The second number is the W dimension up to which the change applies. 8 is a good default.
# (Lower number means only low level features will be affected)
# The expected format is a numpy array of the shape (18, 512).


# Synthetic because they are just found by mucking around in Z space.
synthetic_glasses = torch.zeros((1, 512, ))
synthetic_glasses[:, 30] = -1
synthetic_glasses[:, 36] = -1

# Empirical glasses I manually labelled about 200 images and took the difference in means
# as direction.

# Beard was found through gradient descent modification and subsequent PCA of differences.

def known_directions():
    return {
        'smile': ('directions/smile_styleganv2.npy', 0, 5),
        'age': ('directions/age.npy', 0, 8),
        'gender': ('directions/gender.npy', 0, 5),
        'eyes_open': ('directions/eyes_open.npy', 0, 12),
        'nose_ratio': ('directions/nose_ratio.npy', 0, 8),
        'nose_tip': ('directions/nose_tip.npy', 0, 12),
        'lip_ratio': ('directions/lip_ratio.npy', 0, 12),
        'eye_distance': ('directions/eye_distance.npy', 0, 8),
        'eye_to_eyebrow_distance': ('directions/eye_eyebrow_distance.npy', 0, 5),
        'eye_ratio': ('directions/eye_ratio.npy', 0, 8),
        'mouth_open': ('directions/mouth_open.npy', 0, 8),
        'pitch': ('directions/pitch.npy', 0, 5),
        # 'roll': ('directions/roll.npy', 5),
        'synthetic_glasses': (synthetic_glasses, 2, 4),
        'empirical_glasses': ('directions/empirical_glasses.npy', 2, 4),
        'beard': ('directions/beard.npy', 6, 9),
        'yaw': ('directions/yaw.npy', 0, 5),
        'light': ('directions/light.npy', 5, 11),
        'eye_color': ('directions/light.npy', 14, 15),
        # 'eig_all_0': (torch.load('components/all.pt')[:, 0].view(1, 512), 0, 18),
        # 'eig_all_1': (torch.load('components/all.pt')[:, 1].view(1, 512), 0, 18),
        # 'eig_all_2': (torch.load('components/all.pt')[:, 2].view(1, 512), 0, 18),
        # 'eig_all_3': (torch.load('components/all.pt')[:, 3].view(1, 512), 0, 18),
        # 'eig_all_4': (torch.load('components/all.pt')[:, 4].view(1, 512), 0, 18),
        # 'eig_all_5': (torch.load('components/all.pt')[:, 5].view(1, 512), 0, 18),
        # 'eig_all_6': (torch.load('components/all.pt')[:, 6].view(1, 512), 0, 18),
        # 'eig_all_7': (torch.load('components/all.pt')[:, 7].view(1, 512), 0, 18),
        # 'eig_all_8': (torch.load('components/all.pt')[:, 8].view(1, 512), 0, 18),
        # 'eig_all_9': (torch.load('components/all.pt')[:, 9].view(1, 512), 0, 18),
        # 'eig_all_10': (torch.load('components/all.pt')[:, 10].view(1, 512), 0, 18),

        # 'eig_64_0': (torch.load('components/b64_conv0.pt')[:, 0].view(1, 512), 0, 18),
        # 'eig_64_1': (torch.load('components/b64_conv0.pt')[:, 1].view(1, 512), 0, 18),
        # 'eig_64_2': (torch.load('components/b64_conv0.pt')[:, 2].view(1, 512), 0, 18),
        # 'eig_64_3': (torch.load('components/b64_conv0.pt')[:, 3].view(1, 512), 0, 18),
        # 'eig_64_4': (torch.load('components/b64_conv0.pt')[:, 4].view(1, 512), 0, 18),
        # 'eig_64_5': (torch.load('components/b64_conv0.pt')[:, 5].view(1, 512), 0, 18),
        # 'eig_64_6': (torch.load('components/b64_conv0.pt')[:, 6].view(1, 512), 0, 18),
        # 'eig_64_7': (torch.load('components/b64_conv0.pt')[:, 7].view(1, 512), 0, 18),
        # 'eig_64_8': (torch.load('components/b64_conv0.pt')[:, 8].view(1, 512), 0, 18),
        # 'eig_64_9': (torch.load('components/b64_conv0.pt')[:, 9].view(1, 512), 0, 18),
        # 'eig_64_10': (torch.load('components/b64_conv0.pt')[:, 10].view(1, 512), 0, 18),

        # 'eig_16_0': (torch.load('components/b16_conv0.pt')[:, 0].view(1, 512), 0, 18), # Hair ?
        # 'eig_16_1': (torch.load('components/b16_conv0.pt')[:, 1].view(1, 512), 0, 18), # Gender ?
        # 'eig_16_2': (torch.load('components/b16_conv0.pt')[:, 2].view(1, 512), 0, 18),
        # 'eig_16_3': (torch.load('components/b16_conv0.pt')[:, 3].view(1, 512), 0, 18), # Roll
        # 'eig_16_4': (torch.load('components/b16_conv0.pt')[:, 4].view(1, 512), 0, 11), # Age
        # 'eig_16_5': (torch.load('components/b16_conv0.pt')[:, 5].view(1, 512), 0, 18),
        # 'eig_16_6': (torch.load('components/b16_conv0.pt')[:, 6].view(1, 512), 0, 18),
        # 'eig_16_7': (torch.load('components/b16_conv0.pt')[:, 7].view(1, 512), 0, 18),
        # 'eig_16_8': (torch.load('components/b16_conv0.pt')[:, 8].view(1, 512), 0, 18),
        # 'eig_16_9': (torch.load('components/b16_conv0.pt')[:, 9].view(1, 512), 0, 18), # Weird age
        # 'eig_16_10': (torch.load('components/b16_conv0.pt')[:, 10].view(1, 512), 0, 18), # Pitch
    }