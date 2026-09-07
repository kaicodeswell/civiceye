import streamlit as st
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(
    page_title="CIVIC EYE",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 CIVIC EYE")
st.caption("Emergency Zone & Vehicle Obstruction Detection")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(
    BASE_DIR,
    "videos",
    "traffic.mp4"
)

st.session_state.setdefault("points", [])
st.session_state.setdefault("zone_created", False)
st.session_state.setdefault("running", False)

os.makedirs(
    os.path.join(BASE_DIR, "incidents"),
    exist_ok=True
)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()


if not os.path.exists(VIDEO_PATH):
    st.error(f"Video not found: {VIDEO_PATH}")
    st.stop()


video = cv2.VideoCapture(VIDEO_PATH)

success, first_frame = video.read()

video.release()

if not success:
    st.error("Could not read traffic.mp4")
    st.stop()


first_frame_rgb = cv2.cvtColor(
    first_frame,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# ZONE SELECTION
# --------------------------------------------------

if not st.session_state.zone_created:

    st.subheader("📍 Select Emergency Zone")

    st.info(
        "Click directly on the video to add points. "
        "Select at least 3 points to create the zone."
    )

    preview = first_frame_rgb.copy()

    points = st.session_state.points

    for point in points:

        cv2.circle(
            preview,
            point,
            7,
            (255, 0, 0),
            -1
        )

    if len(points) > 1:

        cv2.polylines(
            preview,
            [
                np.array(
                    points,
                    dtype=np.int32
                )
            ],
            False,
            (255, 0, 0),
            3
        )

    if len(points) >= 3:

        cv2.line(
            preview,
            points[-1],
            points[0],
            (255, 0, 0),
            3
        )

    clicked = streamlit_image_coordinates(
        preview,
        key="zone_selector"
    )

    if clicked is not None:

        click_x = int(clicked["x"])
        click_y = int(clicked["y"])

        last_point = (
            click_x,
            click_y
        )

        if (
            len(points) == 0
            or points[-1] != last_point
        ):

            st.session_state.points.append(
                last_point
            )

            st.rerun()


    st.write(
        f"**Points selected:** "
        f"{len(st.session_state.points)}"
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "🚨 Create Emergency Zone",
            disabled=len(st.session_state.points) < 3
        ):

            st.session_state.zone_created = True
            st.session_state.running = True

            st.rerun()


    with col2:

        if st.button("🔄 Reset Points"):

            st.session_state.points = []

            st.rerun()


    if len(points) >= 3:

        st.success(
            "Zone ready! Click 'Create Emergency Zone'."
        )

    st.stop()


# --------------------------------------------------
# DETECTION
# --------------------------------------------------

st.success("🚨 Emergency zone activated")

col1, col2 = st.columns(2)

with col1:

    if st.button("🔄 Reset Zone"):

        st.session_state.points = []
        st.session_state.zone_created = False
        st.session_state.running = False

        st.rerun()


with col2:

    if st.button("⏹ Stop Detection"):

        st.session_state.running = False


st.subheader("🎥 CIVIC EYE Detection")


video_placeholder = st.empty()

status_placeholder = st.empty()


zone = np.array(
    st.session_state.points,
    dtype=np.int32
)


previous_positions = {}
stop_timers = {}
alerted_ids = set()


MOVEMENT_THRESHOLD = 12
STOP_TIME = 5


video = cv2.VideoCapture(
    VIDEO_PATH
)


while st.session_state.running:

    success, frame = video.read()

    if not success:

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            0
        )

        previous_positions.clear()
        stop_timers.clear()
        alerted_ids.clear()

        continue


    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )


    display = results[0].plot()


    cv2.polylines(
        display,
        [
            zone.reshape(
                (-1, 1, 2)
            )
        ],
        True,
        (0, 0, 255),
        3
    )


    obstruction_detected = False


    if results[0].boxes.id is not None:

        boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        ids = (
            results[0]
            .boxes
            .id
            .cpu()
            .numpy()
            .astype(int)
        )

        classes = (
            results[0]
            .boxes
            .cls
            .cpu()
            .numpy()
            .astype(int)
        )


        for box, track_id, class_id in zip(
            boxes,
            ids,
            classes
        ):

            # car, motorcycle, bus, truck
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


            previous_position = (
                previous_positions.get(
                    track_id
                )
            )


            movement = 0


            if previous_position is not None:

                movement = np.linalg.norm(
                    np.array(bottom_center)
                    -
                    np.array(previous_position)
                )


            previous_positions[
                track_id
            ] = bottom_center


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

                        stop_timers[
                            track_id
                        ] = time.time()


                    stopped_for = (
                        time.time()
                        -
                        stop_timers[track_id]
                    )


                    cv2.putText(
                        display,
                        f"STOPPED: {stopped_for:.1f}s",
                        (
                            int(x1),
                            int(y1) - 10
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )


                    if stopped_for >= STOP_TIME:

                        obstruction_detected = True


                        if track_id not in alerted_ids:

                            alerted_ids.add(
                                track_id
                            )


                            filename = os.path.join(
                                BASE_DIR,
                                "incidents",
                                f"vehicle_{track_id}_"
                                f"{int(time.time())}.jpg"
                            )


                            cv2.imwrite(
                                filename,
                                display
                            )


                        cv2.rectangle(
                            display,
                            (
                                int(x1),
                                int(y1)
                            ),
                            (
                                int(x2),
                                int(y2)
                            ),
                            (0, 0, 255),
                            4
                        )


                        cv2.putText(
                            display,
                            "!!! OBSTRUCTION !!!",
                            (
                                int(x1),
                                int(y1) - 40
                            ),
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
                        (
                            int(x1),
                            int(y1) - 10
                        ),
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
            "🚨 OBSTRUCTION DETECTED"
        )

    else:

        status_placeholder.success(
            "🟢 Emergency zone clear"
        )


    display_rgb = cv2.cvtColor(
        display,
        cv2.COLOR_BGR2RGB
    )


    video_placeholder.image(
        display_rgb,
        channels="RGB",
        use_container_width=True
    )


    time.sleep(0.03)


video.release()
