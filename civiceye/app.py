import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
from detector import process_video

st.set_page_config(
    page_title="CIVIC EYE",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 CIVIC EYE")
st.subheader("Emergency Route Monitoring System")

VIDEO_PATH = "videos/traffic.mp4"

if "zone_points" not in st.session_state:
    st.session_state.zone_points = []

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if not st.session_state.monitoring:

    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        st.error("Could not open traffic video.")
        st.stop()

    video.set(cv2.CAP_PROP_POS_FRAMES, 100)

    success, frame = video.read()

    video.release()

    if not success:
        st.error("Could not read video.")
        st.stop()

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    display = frame_rgb.copy()

    points = st.session_state.zone_points

    for point in points:

        cv2.circle(
            display,
            point,
            8,
            (255, 0, 0),
            -1
        )

    if len(points) >= 2:

        pts = np.array(
            points,
            dtype=np.int32
        )

        cv2.polylines(
            display,
            [pts],
            False,
            (255, 0, 0),
            4
        )

    if len(points) >= 3:

        pts = np.array(
            points,
            dtype=np.int32
        )

        overlay = display.copy()

        cv2.fillPoly(
            overlay,
            [pts],
            (255, 0, 0)
        )

        display = cv2.addWeighted(
            overlay,
            0.20,
            display,
            0.80,
            0
        )

        cv2.polylines(
            display,
            [pts],
            True,
            (255, 0, 0),
            4
        )

    st.header("📍 Define Emergency Zone")

    st.write(
        "Click at least 3 points around the emergency lane."
    )

    clicked = streamlit_image_coordinates(
        Image.fromarray(display),
        key="zone_selector"
    )

    if clicked:

        point = (
            int(clicked["x"]),
            int(clicked["y"])
        )

        if point not in st.session_state.zone_points:

            st.session_state.zone_points.append(point)

            st.rerun()

    st.write(
        f"Zone points selected: "
        f"{len(st.session_state.zone_points)}"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button("↩️ Undo Last Point"):

            if st.session_state.zone_points:

                st.session_state.zone_points.pop()

                st.rerun()

    with col2:

        if st.button("🗑️ Clear Zone"):

            st.session_state.zone_points = []

            st.rerun()

    if len(st.session_state.zone_points) >= 3:

        st.success("✅ Emergency zone ready!")

        if st.button(
            "🚨 START CIVIC EYE MONITORING",
            type="primary"
        ):

            st.session_state.monitoring = True
            st.rerun()

    else:

        st.warning(
            "Select at least 3 points before starting."
        )

else:

    st.header("📹 CIVIC EYE — LIVE MONITORING")

    points = st.session_state.zone_points

    col1, col2, col3 = st.columns(3)

    vehicle_metric = col1.empty()
    obstruction_metric = col2.empty()
    status_metric = col3.empty()

    video_placeholder = st.empty()

    st.divider()

    st.header("🚨 Incident Status")

    incident_placeholder = st.empty()

    for (
        frame,
        vehicle_count,
        obstruction_count
    ) in process_video(
        VIDEO_PATH,
        points
    ):

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        video_placeholder.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

        vehicle_metric.metric(
            "Vehicles Detected",
            vehicle_count
        )

        obstruction_metric.metric(
            "Active Obstructions",
            obstruction_count
        )

        if obstruction_count > 0:

            status_metric.metric(
                "System Status",
                "🚨 ALERT"
            )

            incident_placeholder.error(
                "🚨 VEHICLE OBSTRUCTION DETECTED"
            )

        else:

            status_metric.metric(
                "System Status",
                "🟢 ACTIVE"
            )

            incident_placeholder.success(
                "No active obstruction."
            )
