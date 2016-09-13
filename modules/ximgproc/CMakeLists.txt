set(the_description "Extended image processing module. It includes edge-aware filters and etc.")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)

ocv_define_module(ximgproc opencv_imgproc opencv_core opencv_highgui opencv_calib3d opencv_ml WRAP python)

target_link_libraries(opencv_ximgproc)

file(COPY samples/rcGroundTruth.png DESTINATION ${OpenCV_BINARY_DIR}/modules/ximgproc)
file(COPY samples/rcImage.png DESTINATION ${OpenCV_BINARY_DIR}/modules/ximgproc)
file(COPY samples/rcSeed.png DESTINATION ${OpenCV_BINARY_DIR}/modules/ximgproc)