from pipeline import *
from moviepy.editor import VideoFileClip

camera = cam.Camera()

def process_image(image):
    pipeline = Pipeline(image, camera)
    out = pipeline\
        .undistort()\
        .gradient_binary() \
        .mask_region_of_interest() \
        .warp() \
        .draw_lanes() \
        .unwarp() \
        .add_weighted() \
        .display_params() \
        .image

    return out


if __name__ == "__main__":
    """
    image = mpimg.imread('test_images/test3.jpg')
    out = process_image(image)
    cv2.imwrite('final.jpg', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    plt.imshow(out)
    plt.show()

    """
    video_output = 'out.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    out_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    out_clip.write_videofile(video_output, audio=False)

