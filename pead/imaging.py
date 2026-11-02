from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass(frozen=True)
class Recipe:
    # ROI is normalized x, y, width, height in the original camera frame.
    roi: tuple[float, float, float, float] = (0, 0, 1, 1)
    clahe: bool = False
    glare_level: int = 248
    max_glare: float = 0.02
    tile_size: int = 512
    overlap: float = 0.25
    min_area: int = 20  # pixels in the cropped full-resolution ROI

    def __post_init__(self):
        x, y, w, h = self.roi
        if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x+w <= 1.000001 and y+h <= 1.000001):
            raise ValueError('ROI must lie within the image')
        if self.tile_size not in (256, 512, 1024) or not 0 <= self.overlap <= 0.5:
            raise ValueError('Invalid tile settings')
        if not 200 <= self.glare_level <= 255 or not 0 <= self.max_glare <= 1 or self.min_area < 1:
            raise ValueError('Invalid glare or area settings')

    def dict(self):
        return asdict(self)


def roi_box(shape, recipe):
    h, w = shape[:2]
    x, y, rw, rh = recipe.roi
    x0, y0 = int(x*w), int(y*h)
    x1, y1 = min(w, int((x+rw)*w)), min(h, int((y+rh)*h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError('ROI is empty at this image resolution')
    return x0, y0, x1, y1


def crop(frame, recipe):
    x0, y0, x1, y1 = roi_box(frame.shape, recipe)
    return frame[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def prepare(frame, recipe):
    image, box = crop(frame, recipe)
    # Detect clipping before normalization. Never inpaint or erase potential defects.
    glare = np.min(image, axis=2) >= recipe.glare_level
    glare = cv2.dilate(glare.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
    if recipe.clahe:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(lab[:,:,0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image, glare, box


def tile_boxes(shape, recipe):
    """Plan native-resolution crops without decoding or copying image pixels."""
    h, w = shape[:2]
    size = recipe.tile_size
    def positions(length):
        if length <= size:
            return [0]
        return sorted(set([*range(0, length-size+1, int(size*(1-recipe.overlap))), length-size]))
    xs, ys = positions(w), positions(h)
    for y in ys:
        for x in xs:
            yield x,y,min(x+size,w),min(y+size,h)


def tiles(image, recipe):
    for x,y,x1,y1 in tile_boxes(image.shape, recipe):
        yield (x,y,x1,y1), image[y:y1,x:x1]


def regions(score, threshold, min_area):
    mask = (score >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    result = []
    for i in range(1, count):
        x,y,w,h,area = stats[i]
        if area >= min_area:
            result.append({'x':int(x),'y':int(y),'width':int(w),'height':int(h),'area':int(area),'score':float(score[labels==i].max())})
    return result


def overlay(frame, score, glare, box, recipe):
    x0,y0,x1,y1 = box
    # Fixed calibrated scale: blue -> green -> yellow -> red (never per-frame min/max).
    norm = np.clip(score,0,1)
    heat = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_JET)
    out = frame.copy()
    out[y0:y1,x0:x1] = cv2.addWeighted(frame[y0:y1,x0:x1],.55,heat,.45,0)
    out[y0:y1,x0:x1][glare] = (190,50,190)
    valid = score.copy()
    valid[glare] = 0
    red = regions(valid,1.,recipe.min_area)
    yellow = regions(valid,.65,recipe.min_area)
    for items,color in [(yellow,(0,220,255)),(red,(0,0,255))]:
        for r in items:
            cv2.rectangle(out,(x0+r['x'],y0+r['y']),(x0+r['x']+r['width'],y0+r['y']+r['height']),color,2)
    fraction = float(glare.mean())
    state = 'UNINSPECTABLE' if fraction > recipe.max_glare else 'ANOMALY' if red else 'REVIEW' if yellow or fraction > 0 else 'NORMAL'
    cv2.rectangle(out,(x0,y0),(x1-1,y1-1),(210,210,210),1)
    return out, {'decision':state,'glare_fraction':fraction,'regions':red,'doubtful_regions':yellow,'score':float(valid.max()),'coordinate_space':'ROI pixels'}
