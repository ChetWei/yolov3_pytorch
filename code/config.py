# -*- coding: utf-8 -*-

anchors_mask1 = [[116, 90], [156, 198], [373, 326]]
anchors_mask2 = [[30, 61], [62, 45], [59, 119]]
anchors_mask3 = [[10, 13], [16, 30], [33, 23]]
#   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
#   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
#   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
anchors_mask_list = [anchors_mask1, anchors_mask2, anchors_mask3]