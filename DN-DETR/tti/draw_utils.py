import os.path
import cv2
import smrc.utils


LINE_THICKNESS = 2


def draw_single_bbox(tmp_img, bbox, rectangle_color=smrc.utils.color.YELLOW,
                     anchor_rect_color=None, text_shadow_color=None,
                     text_color=(0, 0, 0), caption=None):
    # specify the colors
    bbox = list(map(int, bbox))
    class_idx, xmin, ymin, xmax, ymax = bbox
    # class_color = self.CLASS_BGR_COLORS[class_idx].tolist()

    if anchor_rect_color is None: anchor_rect_color = rectangle_color
    if text_shadow_color is None: text_shadow_color = rectangle_color

    draw_bbox_with_specialized_shape(
        tmp_img=tmp_img, bbox=bbox, color=rectangle_color,
        anchor_rect_color=anchor_rect_color, thickness=2
    )

    if caption is not None:
        # caption = self.CLASS_LIST[class_idx]  # default caption of the bbox is class_name
        # text_shadow_color = class_color, text_color = (0, 0, 0), i.e., black
        draw_class_name(tmp_img, (xmin, ymin), caption, text_shadow_color, text_color)  #


def draw_bbox_with_specialized_shape(tmp_img, bbox, color,
                                     thickness, **kwargs):
    class_idx, xmin, ymin, xmax, ymax = bbox

    # draw the rectangle
    cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color, thickness)


def draw_class_name(tmp_img, location_to_draw, text_content, text_shadow_color, text_color):
    font = cv2.FONT_HERSHEY_SIMPLEX  # FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    margin = 3
    text_width, text_height = cv2.getTextSize(text_content, font, font_scale, LINE_THICKNESS)[0]

    xmin, ymin = int(location_to_draw[0]), int(location_to_draw[1])
    cv2.rectangle(tmp_img, (xmin, ymin), (xmin + text_width + margin, ymin - text_height - margin),
                  text_shadow_color, -1)

    cv2.putText(tmp_img, text_content, (xmin, ymin - 5), font, font_scale, text_color, LINE_THICKNESS,
                int(cv2.LINE_AA))

    # # draw resizing anchors
    # if 'anchor_rect_color' in kwargs:
    #     anchor_color = kwargs['anchor_rect_color']
    # else:
    #     anchor_color = color
    #
            # if __name__ == '__main__':

