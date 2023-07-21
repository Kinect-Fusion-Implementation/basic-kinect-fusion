#include "KinectSensor.h"
#include <iostream>

void callback_depth(freenect_device* dev, void* data, uint32_t timestamp)
{
	cv::waitKey(30);
	printf("Received depth frame at %d\n", timestamp);

}

void callback_rgb(freenect_device* dev, void* data, uint32_t timestamp)
{
	cv::waitKey(30);
	printf("Received video frame at %d\n", timestamp);
}

KinectSensor::KinectSensor() {
	freenect_context* fn_ctx;
	int ret = freenect_init(&fn_ctx, NULL);
	if (ret < 0)
		return;
	freenect_set_log_level(fn_ctx, FREENECT_LOG_DEBUG);
	freenect_select_subdevices(fn_ctx, FREENECT_DEVICE_CAMERA);
	// Find out how many devices are connected.
	int num_devices = freenect_num_devices(fn_ctx);
	if (ret < 0)
		return;
	if (num_devices == 0)
	{
		printf("No device found!\n");
		freenect_shutdown(fn_ctx);
		return;
	}

	// Open the first device.
	freenect_device* fn_dev;
	ret = freenect_open_device(fn_ctx, &fn_dev, 0);
	if (ret < 0)
	{
		freenect_shutdown(fn_ctx);
		return;
	}

	// Set depth and video modes.
	ret = freenect_set_depth_mode(fn_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_REGISTERED));
	if (ret < 0)
	{
		freenect_shutdown(fn_ctx);
		return;
	}
	ret = freenect_set_video_mode(fn_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));
	if (ret < 0)
	{
		freenect_shutdown(fn_ctx);
		return;
	}

	float* depth;
	depth = (float*) malloc(sizeof(float) * 640 * 480);
	freenect_set_depth_buffer(fn_dev, depth);
	// Set frame callbacks.
	freenect_set_depth_callback(fn_dev, callback_depth);
	freenect_set_video_callback(fn_dev, callback_rgb);

	cv::Mat frame(480, 640, CV_8UC3, depth);
	cv::imshow("Test", frame);
	cv::waitKey(30);

	// Start depth and video.
	ret = freenect_start_depth(fn_dev);
	if (ret < 0)
	{
		freenect_shutdown(fn_ctx);
		return;
	}
	ret = freenect_start_video(fn_dev);
	if (ret < 0)
	{
		freenect_shutdown(fn_ctx);
		return;
	}

	// Run until interruption or failure.
	while (freenect_process_events(fn_ctx) >= 0)
	{

	}

	printf("Shutting down\n");

	// Stop everything and shutdown.
	freenect_stop_depth(fn_dev);
	freenect_stop_video(fn_dev);
	freenect_close_device(fn_dev);
	freenect_shutdown(fn_ctx);

	printf("Done!\n");
}

bool KinectSensor::init()
{
	return false;
}

bool KinectSensor::processNextFrame(){
	return false;
}

unsigned int KinectSensor::getCurrentFrameCnt()
{
	return -1;
}

// get current color data
BYTE *KinectSensor::getColorRGBX()
{
	return nullptr;
}

// get current depth data
float *KinectSensor::getDepth()
{
	return nullptr;
}

// color camera info
Eigen::Matrix3f KinectSensor::getColorIntrinsics()
{
	return Eigen::Matrix3f::Identity();
}

Eigen::Matrix4f KinectSensor::getColorExtrinsics()
{
	return Eigen::Matrix4f::Identity();
}

unsigned int KinectSensor::getColorImageWidth()
{
	return -1;
}

unsigned int KinectSensor::getColorImageHeight()
{
	return -1;
}

// depth (ir) camera info
Eigen::Matrix3f KinectSensor::getDepthIntrinsics()
{
	return Eigen::Matrix3f::Identity();
}

Eigen::Matrix4f KinectSensor::getDepthExtrinsics()
{
	return Eigen::Matrix4f::Identity();
}

unsigned int KinectSensor::getDepthImageWidth()
{
	return -1;
}

unsigned int KinectSensor::getDepthImageHeight()
{
	return -1;
}