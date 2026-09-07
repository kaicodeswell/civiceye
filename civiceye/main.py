import streamlit as st
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="CIVIC EYE",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 CIVIC EYE")
st.caption("Emergency Zone & Vehicle Obstruction Detection")

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "videos/traffic.mp4"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

if not os.path.exists(VIDEO_PATH):
    st.error(f"Video not found: {VIDEO_PATH}")
    st.stop()

os.makedirs("incidents", exist_ok=True)

st.sidebar.header("⚙️ Controls")

if "points" not in st.session_state:
    st.session_state.points = []

if "zone_created" not in st.session_state:
    st.session_state.zone_created = False

if "running" not in st.session_state:
    st.session_state.running = False

if st.sidebar.button("🔄 Reset Zone"):
    st.session_state.points = []
    st.session_state.zone_created = False
    st.session_state.running = False
    st.rerun()

st.sidebar.markdown("### Zone Instructions")
st.sidebar.write("Click points on the video to create the emergency zone.")

st.sidebar.write("After selecting at least 3 points, click **Create Zone**.")

st.sidebar.write("Vehicles stopped for 5+ seconds will be marked as obstructions.")

st.subheader("Emergency Zone")

uploaded_frame = cv2.VideoCapture(VIDEO_PATH)
success, first_frame = uploaded_frame.read()
uploaded_frame.release()

if not success:
    st.error("Could not read traffic video.")
    st.stop()

first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

st.image(
    first_frame_rgb,
    caption="Select your emergency zone using the controls below.",
    use_container_width=True
)

st.markdown("### 📍 Zone Coordinates")

col1, col2, col3 = st.columns(3)

with col1:
    x = st.number_input(
        "X coordinate",
        min_value=0,
        max_value=first_frame.shape[1],
        value=100
    )

with col2:
    y = st.number_input(
        "Y coordinate",
        min_value=0,
        max_value=first_frame.shape[0],
        value=100
    )

with col3:
    if st.button("➕ Add Point"):
        st.session_state.points.append((int(x), int(y)))

if st.session_state.points:
    st.write("Selected points:", st.session_state.points)

    preview = first_frame.copy()

    for point in st.session_state.points:
        cv2.circle(
            preview,
            point,
            7,
            (0, 0, 255),
            -1
        )

    if len(st.session_state.points) > 1:
        cv2.polylines(
            preview,
            [np.array(st.session_state.points, dtype=np.int32)],
            False,
            (0, 0, 255),
            3
        )

    if len(st.session_state.points) >= 3:
        cv2.line(
            preview,
            st.session_state.points[-1],
            st.session_state.points[0],
            (0, 0, 255),
            3
        )

    st.image(
        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
        caption="Current Emergency Zone",
        use_container_width=True
    )

if len(st.session_state.points) >= 3:

    if st.button("🚨 Create Emergency Zone"):
        st.session_state.zone_created = True
        st.session_state.running = True
        st.rerun()

if not st.session_state.zone_created:
    st.info("Select at least 3 points to create the emergency zone.")
    st.stop()

st.success("Emergency zone created successfully!")

st.subheader("🎥 Live Detection")

video_placeholder = st.empty()
status_placeholder = st.empty()

zone = np.array(
    st.session_state.points,
    dtype=np.int32
)

video = cv2.VideoCapture(VIDEO_PATH)

previous_positions = {}
stop_timers = {}
alerted_ids = set()

MOVEMENT_THRESHOLD = 12
STOP_TIME = 5

frame_count = 0

while st.session_state.running:

    success, frame = video.read()

    if not success:
        break

    frame_count += 1

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    display = results[0].plot()

    cv2.polylines(
        display,
        [zone.reshape((-1, 1, 2))],
        True,
        (0, 0, 255),
        3
    )

    obstruction_detected = False

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
                    np.array(bottom_center)
                    -
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
                        time.time()
                        -
                        stop_timers[track_id]
                    )

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

                        obstruction_detected = True

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

                    stop_timers.pop(
                        track_id,
                        None
                    )

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

                stop_timers.pop(
                    track_id,
                    None
                )

    if obstruction_detected:
        status_placeholder.error(
            "🚨 OBSTRUCTION DETECTED — VEHICLE STOPPED IN EMERGENCY ZONE"
        )
    else:
        status_placeholder.success(
            "🟢 Emergency zone clear"
        )

    video_placeholder.image(
        cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
        channels="RGB",
        use_container_width=True
    )

    time.sleep(0.03)

video.release()

st.session_state.running = False

st.success("Video processing completed.")

if st.button("▶️ Run Detection Again"):
    st.session_state.running = True
    st.rerun()
