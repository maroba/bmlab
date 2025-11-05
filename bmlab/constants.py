from enum import Enum


c = 299792458  # [m/s] speed of light


class ExtractionMethod(Enum):
    ARC_FROM_PTS_OF_AVG_IMG = "arc_from_pts_of_avg_img"
    ARC_FROM_PTS_OF_ALL_IMGS = "arc_from_pts_of_all_imgs"
