set(the_description "Extended image processing module. It includes edge-aware filters and etc.")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)
ocv_define_module(ximgproc opencv_imgproc opencv_core opencv_highgui opencv_ml)

target_link_libraries(opencv_ximgproc)

file(COPY samples/grabcut_cow.png DESTINATION ${OpenCV_BINARY_DIR}/modules/ximgproc)
file(COPY samples/grabcut_cow_mask.bmp DESTINATION ${OpenCV_BINARY_DIR}/modules/ximgproc)
file(COPY samples/grabcut_cow_mask.png DESTINATION ${OpenCV_BINARY_DIR}/modules/ximgproc)
file(COPY samples/grabcut_cow_mask_skel.png DESTINATION ${OpenCV_BINARY_DIR}/modules/ximgproc)
