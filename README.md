# traffic-monitor

Example run

```
from pathlib import Path
import cv2
import moviepy.editor as mpy

from traffic_monitor.detection import TrafficDetector

path = Path('stream_data', 'streamlink_20210303_145004.mp4')
td = TrafficDetector(path, step=5)
td.detect_vehicles()
boxes = td.draw_boxes()
pred_imgs = lambda x: cv2.cvtColor(next(boxes), cv2.COLOR_BGR2RGB)
clip = mpy.VideoClip(pred_imgs, duration=20)  # fps * duration < len(frames)
clip.write_videofile("out.mp4", fps=3)
```
