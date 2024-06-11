MIN_APPLE_WIDTH = 4
MIN_APPLE_HEIGHT = 4

def merge_overlapping_boxes(boxes):
    if not boxes:
        return []

    merged_boxes = []
    boxes = sorted(boxes, key=lambda b: b[0])

    while boxes:
        a = boxes.pop(0)
        merged = False
        for i in range(len(merged_boxes)):
            b = merged_boxes[i]
            if (a[0] <= b[0] + b[2] and a[0] + a[2] >= b[0] and
                    a[1] <= b[1] + b[3] and a[1] + a[3] >= b[1]):
                # Merge boxes
                new_x = min(a[0], b[0])
                new_y = min(a[1], b[1])
                new_w = max(a[0] + a[2], b[0] + b[2]) - new_x
                new_h = max(a[1] + a[3], b[1] + b[3]) - new_y
                merged_boxes[i] = (new_x, new_y, new_w, new_h)
                merged = True
                break
        if not merged:
            merged_boxes.append(a)

    return merged_boxes

def check_image_dimension(w, h):
    return w > MIN_APPLE_WIDTH and h > MIN_APPLE_HEIGHT