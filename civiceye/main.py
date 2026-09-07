from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import winsound

model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("videos/traffic.mp4")

points = []
zone_created = False

previous_positions = {}
stop_timers = {}
alerted_ids = set()

MOVEMENT_THRESHOLD = 12
STOP_TIME = 5

os.makedirs("incidents", exist_ok=True)


def draw_zone(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and not zone_created:
        points.append((x, y))


cv2.namedWindow("CIVIC EYE")
cv2.setMouseCallback("CIVIC EYE", draw_zone)

while True:

    success, frame = video.read()

    if not success:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        previous_positions.clear()
        stop_timers.clear()
        alerted_ids.clear()
        continue

    display = frame.copy()

    for point in points:
        cv2.circle(display, point, 6, (0, 0, 255), -1)

    if len(points) > 1:
        cv2.polylines(
            display,
            [np.array(points, dtype=np.int32)],
            False,
            (0, 0, 255),
            3
        )

    if len(points) >= 3:
        cv2.line(
            display,
            points[-1],
            points[0],
            (0, 0, 255),
            3
        )

    if zone_created:

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml"
        )

        display = results[0].plot()

        zone = np.array(points, dtype=np.int32)

        cv2.polylines(
            display,
            [zone.reshape((-1, 1, 2))],
            True,
            (0, 0, 255),
            3
        )

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, class_id in zip(boxes, ids, classes):

                if class_id not in {2, 3, 5, 7}:
                    continue

                x1, y1, x2, y2 = box

                bottom_center = (
                    int((x1 + x2) / 2),
                    int(y2)
                )

                inside = cv2.pointPolygonTest(
                    zone,
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

                        stopped_for = time.time() - stop_timers[track_id]

                        cv2.putText(
                            display,
                            f"STOPPED: {stopped_for:.1f}s",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                        if stopped_for >= STOP_TIME:

                            if track_id not in alerted_ids:

                                alerted_ids.add(track_id)

                                filename = (
                                    f"incidents/"
                                    f"vehicle_{track_id}_"
                                    f"{int(time.time())}.jpg"
                                )

                                cv2.imwrite(filename, display)

                                winsound.Beep(1000, 700)

                            cv2.rectangle(
                                display,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 0, 255),
                                4
                            )

                            cv2.putText(
                                display,
                                "!!! OBSTRUCTION !!!",
                                (int(x1), int(y1) - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0, 0, 255),
                                3
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

    cv2.imshow("CIVIC EYE", display)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("d") and len(points) >= 3:
        zone_created = True

    if key == ord("r"):
        points = []
        zone_created = False
        previous_positions.clear()
        stop_timers.clear()
        alerted_ids.clear()

    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
