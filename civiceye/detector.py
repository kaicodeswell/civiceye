import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

model = YOLO("yolov8n.pt")

STOP_TIME = 5
MOVEMENT_THRESHOLD = 12

os.makedirs("incidents", exist_ok=True)


def process_video(video_path, zone):

    video = cv2.VideoCapture(video_path)

    previous_positions = {}
    stop_timers = {}
    alerted_ids = set()

    while True:

        success, frame = video.read()

        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml"
        )

        display = results[0].plot()

        zone_array = np.array(zone, dtype=np.int32)

        cv2.polylines(
            display,
            [zone_array.reshape((-1, 1, 2))],
            True,
            (0, 0, 255),
            4
        )

        vehicles_detected = 0
        active_obstructions = 0

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, class_id in zip(boxes, ids, classes):

                if class_id not in {2, 3, 5, 7}:
                    continue

                vehicles_detected += 1

                x1, y1, x2, y2 = box

                bottom_center = (
                    int((x1 + x2) / 2),
                    int(y2)
                )

                inside = cv2.pointPolygonTest(
                    zone_array,
                    bottom_center,
                    False
                )

                previous_position = previous_positions.get(track_id)

                movement = 0

                if previous_position is not None:

                    movement = np.linalg.norm(
                        np.array(bottom_center) -
                        np.array(previous_position)
                    )

                previous_positions[track_id] = bottom_center

                if inside >= 0:

                    cv2.circle(
                        display,
                        bottom_center,
                        7,
                        (255, 0, 0),
                        -1
                    )

                    if movement <= MOVEMENT_THRESHOLD:

                        if track_id not in stop_timers:
                            stop_timers[track_id] = time.time()

                        stopped_for = (
                            time.time() -
                            stop_timers[track_id]
                        )

                        if stopped_for < STOP_TIME:

                            cv2.putText(
                                display,
                                f"STOPPED {stopped_for:.1f}s",
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2
                            )

                        else:

                            active_obstructions += 1

                            cv2.rectangle(
                                display,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 0, 255),
                                4
                            )

                            cv2.putText(
                                display,
                                "OBSTRUCTION",
                                (int(x1), int(y1) - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 0, 255),
                                3
                            )

                            cv2.putText(
                                display,
                                f"Vehicle ID {track_id} - {stopped_for:.1f}s",
                                (int(x1), int(y2) + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2
                            )

                            if track_id not in alerted_ids:

                                alerted_ids.add(track_id)

                                filename = (
                                    f"incidents/"
                                    f"vehicle_{track_id}_"
                                    f"{int(time.time())}.jpg"
                                )

                                cv2.imwrite(
                                    filename,
                                    display
                                )

                    else:

                        stop_timers.pop(track_id, None)

                        cv2.putText(
                            display,
                            "MOVING",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                else:

                    stop_timers.pop(track_id, None)

        yield display, vehicles_detected, active_obstructions

    video.release()
