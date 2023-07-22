#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <math.h>

#include <GL/glut.h>
#include "libfreenect.h"
#include "libfreenect_registration.h"
#include "./sensor/VirtualSensor.h"
#include "./kinect_fusion/Eigen.h"
#include "./kinect_fusion/ICPOptimizer.h"
#include "./kinect_fusion/PointCloud.h"
#include "./kinect_fusion/PointCloudPyramid.h"
#include "./kinect_fusion/ICPOptimizer.h"
#include "./configuration/Configuration.h"
#include "./kinect_fusion/ICPOptimizer.h"
#include "CudaVoxelGrid.h"
#include "../visualization/MarchingCubes.h"
#include "./visualization/PointCloudToMesh.h"

void runKinectFusion(uint16_t* depth, uint8_t* rgb) {

}

/**
 * Code for visualization from https://github.com/OpenKinect/libfreenect/blob/master/examples/glview
 */

int g_argc;
char **g_argv;

pthread_mutex_t buffer_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t gl_frame_cond = PTHREAD_COND_INITIALIZER;

bool got_rgb = false;
bool got_depth = false;

// back: owned by libfreenect (implicit for depth)
// mid: owned by callbacks, "latest frame ready"
// front: owned by GL, "currently being drawn"
uint8_t *depth_img_mid, *depth_img_front;
// Contains the actual depth data
float *depth_mid, *depth_front;
uint8_t *rgb_back, *rgb_mid, *rgb_front;
GLuint gl_depth_tex;
GLuint gl_rgb_tex;

uint16_t t_gamma[2048];

void DrawGLScene()
{
	pthread_mutex_lock(&buffer_mutex);

	while ((!got_depth && !got_rgb))
	{
		pthread_cond_wait(&gl_frame_cond, &buffer_mutex);
	}
    std::chrono::time_point point = std::chrono::high_resolution_clock::now();

	uint8_t *tmp;
	float *tmp_depth;

	if (got_depth)
	{
		// Swap front and mid buffer -> Mid buffer contains the new data, that is why we swap
		tmp = depth_img_front;
		depth_img_front = depth_img_mid;
		depth_img_mid = tmp;
        tmp_depth = depth_front;
		depth_front = depth_mid;
		depth_mid = tmp_depth;
		got_depth = false;

	}
	if (got_rgb)
	{
		// Swap front and mid buffer -> Mid buffer contains the new data, that is why we swap
		tmp = rgb_front;
		rgb_front = rgb_mid;
		rgb_mid = tmp;
		got_rgb = false;
	}
    // Run kinect fusion on the images that will be drawn!
    // runKinectFusion(depth_front, rgb_front);
    
    pthread_mutex_unlock(&buffer_mutex);
	glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, depth_img_front);

	glLoadIdentity();
	glPushMatrix();
	glTranslatef((640.0 / 2.0), (480.0 / 2.0), 0.0);
	// glRotatef(camera_angle, 0.0, 0.0, 1.0);
	glTranslatef(-(640.0 / 2.0), -(480.0 / 2.0), 0.0);
	glBegin(GL_TRIANGLE_FAN);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glTexCoord2f(0, 1);
	glVertex3f(0, 0, 1.0);
	glTexCoord2f(1, 1);
	glVertex3f(640, 0, 1.0);
	glTexCoord2f(1, 0);
	glVertex3f(640, 480, 1.0);
	glTexCoord2f(0, 0);
	glVertex3f(0, 480, 1.0);
	glEnd();
	glPopMatrix();

	glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb_front);

	glPushMatrix();
	glTranslatef(640 + (640.0 / 2.0), (480.0 / 2.0), 0.0);
	// glRotatef(camera_angle, 0.0, 0.0, 1.0);
	glTranslatef(-(640 + (640.0 / 2.0)), -(480.0 / 2.0), 0.0);

	glBegin(GL_TRIANGLE_FAN);
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glTexCoord2f(0, 1);
	glVertex3f(640, 0, 0);
	glTexCoord2f(1, 1);
	glVertex3f(1280, 0, 0);
	glTexCoord2f(1, 0);
	glVertex3f(1280, 480, 0);
	glTexCoord2f(0, 0);
	glVertex3f(640, 480, 0);
	glEnd();
	glPopMatrix();
	glutSwapBuffers();
}

void resizeScene(int Width, int Height)
{
	glViewport(0, 0, Width, Height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1280, 0, 480, -5.0f, 5.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void init(int Width, int Height)
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	// glClearDepth(0.0);
	// glDepthFunc(GL_LESS);
	// glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	glDisable(GL_ALPHA_TEST);
	glEnable(GL_TEXTURE_2D);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glShadeModel(GL_FLAT);

	glGenTextures(1, &gl_depth_tex);
	glBindTexture(GL_TEXTURE_2D, gl_depth_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glGenTextures(1, &gl_rgb_tex);
	glBindTexture(GL_TEXTURE_2D, gl_rgb_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	resizeScene(Width, Height);
}

void *gl_threadfunc(void *arg)
{
	printf("GL thread\n");

	glutInit(&g_argc, g_argv);

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH);
	glutInitWindowSize(1280, 480);
	glutInitWindowPosition(0, 0);

	int window = glutCreateWindow("LibFreenect");

	glutDisplayFunc(&DrawGLScene);
	glutIdleFunc(&DrawGLScene);
	glutReshapeFunc(&resizeScene);

	init(1280, 480);

	glutMainLoop();

	return NULL;
}
std::chrono::time_point start = std::chrono::steady_clock::now();
freenect_device *fn_dev;
freenect_context *fn_ctx;
freenect_registration *fn_registration;
pthread_t freenect_thread;

void callback_depth(freenect_device *dev, void *v_depth, uint32_t timestamp)
{
    std::cout << "Callback frame at: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start) << std::endl;
	uint16_t *depth = (uint16_t *)v_depth;

	pthread_mutex_lock(&buffer_mutex);

	for (int i = 0; i < 640 * 480; i++)
	{
        depth_mid[i] = depth[i] / 1000.0;
        if(depth[i] == 0) {
            depth_img_mid[3 * i + 0] = 255;
            depth_img_mid[3 * i + 1] = 255;
            depth_img_mid[3 * i + 2] = 255;
        } else if (depth[i] < 2000) {
            std::cout << depth[i] << std::endl;
            depth_img_mid[3 * i + 0] = 255 * (depth[i] / 2000.0f);
            depth_img_mid[3 * i + 1] = 0;
            depth_img_mid[3 * i + 2] = 0;
        } else if (depth[i] < 4000) {
            depth_img_mid[3 * i + 0] = 0;
            depth_img_mid[3 * i + 1] = 255 * ((depth[i] / 2000.0f) - 1);
            depth_img_mid[3 * i + 2] = 0;
        } else if (depth[i] < 8000) {
            depth_img_mid[3 * i + 0] = 0;
            depth_img_mid[3 * i + 1] = 0;
            depth_img_mid[3 * i + 2] = 255 * ((depth[i] / 4000.0f) - 1);
        } else {
            depth_img_mid[3 * i + 0] = 255;
            depth_img_mid[3 * i + 1] = 255;
            depth_img_mid[3 * i + 2] = 255;
        }
        // Clamp back
        /*
		int pval = t_gamma[((depth[i])];
		int lb = pval & 0xff;
		switch (pval >> 8)
		{
		case 0:
			depth_img_mid[3 * i + 0] = 255;
			depth_img_mid[3 * i + 1] = 255 - lb;
			depth_img_mid[3 * i + 2] = 255 - lb;
			break;
		case 1:
			depth_img_mid[3 * i + 0] = 255;
			depth_img_mid[3 * i + 1] = lb;
			depth_img_mid[3 * i + 2] = 0;
			break;
		case 2:
			depth_img_mid[3 * i + 0] = 255 - lb;
			depth_img_mid[3 * i + 1] = 255;
			depth_img_mid[3 * i + 2] = 0;
			break;
		case 3:
			depth_img_mid[3 * i + 0] = 0;
			depth_img_mid[3 * i + 1] = 255;
			depth_img_mid[3 * i + 2] = lb;
			break;
		case 4:
			depth_img_mid[3 * i + 0] = 0;
			depth_img_mid[3 * i + 1] = 255 - lb;
			depth_img_mid[3 * i + 2] = 255;
			break;
		case 5:
			depth_img_mid[3 * i + 0] = 0;
			depth_img_mid[3 * i + 1] = 0;
			depth_img_mid[3 * i + 2] = 255 - lb;
			break;
		default:
			depth_img_mid[3 * i + 0] = 0;
			depth_img_mid[3 * i + 1] = 0;
			depth_img_mid[3 * i + 2] = 0;
			break;
		}
        */
	}
	got_depth = true;
	pthread_cond_signal(&gl_frame_cond);
	pthread_mutex_unlock(&buffer_mutex);
}

void callback_rgb(freenect_device *dev, void *rgb, uint32_t timestamp)
{
	pthread_mutex_lock(&buffer_mutex);

	// swap buffers
	rgb_back = rgb_mid;
	freenect_set_video_buffer(dev, rgb_back);
	rgb_mid = (uint8_t *)rgb;

	got_rgb = true;
	pthread_cond_signal(&gl_frame_cond);
	pthread_mutex_unlock(&buffer_mutex);
}

void *freenect_threadfunc(void *arg)
{
	freenect_set_depth_callback(fn_dev, callback_depth);
	freenect_set_video_callback(fn_dev, callback_rgb);
	freenect_set_video_mode(fn_dev, freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));
    // For visualization: FREENECT_DEPTH_11BIT FREENECT_DEPTH_MM
	freenect_set_depth_mode(fn_dev, freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_MM));
	freenect_set_video_buffer(fn_dev, rgb_back);
    start = std::chrono::steady_clock::now();
	freenect_start_depth(fn_dev);
	freenect_start_video(fn_dev);

	while (freenect_process_events(fn_ctx) >= 0)
	{
	}

	printf("\nshutting down streams...\n");

	freenect_stop_depth(fn_dev);
	freenect_stop_video(fn_dev);

	freenect_close_device(fn_dev);
	freenect_shutdown(fn_ctx);

	printf("-- done!\n");
	return NULL;
}

int main()
{
	depth_img_mid = (uint8_t *)malloc(640 * 480 * 3);
	depth_img_front = (uint8_t *)malloc(640 * 480 * 3);
	depth_mid = (float *)malloc(640 * 480 * sizeof(float));
	depth_front = (float *)malloc(640 * 480 * sizeof(float));
	rgb_back = (uint8_t *)malloc(640 * 480 * 3);
	rgb_mid = (uint8_t *)malloc(640 * 480 * 3);
	rgb_front = (uint8_t *)malloc(640 * 480 * 3);

	for (int i = 0; i < 2048; i++)
	{
		float v = i / 2048.0;
		v = powf(v, 3) * 6;
		t_gamma[i] = v * 6 * 256;
	}

	int ret;
	ret = freenect_init(&fn_ctx, NULL);

	freenect_set_log_level(fn_ctx, FREENECT_LOG_DEBUG);
	freenect_select_subdevices(fn_ctx, FREENECT_DEVICE_CAMERA);

	// Find out how many devices are connected.
	int num_devices = freenect_num_devices(fn_ctx);
	if (ret < 0)
		return 1;
	if (num_devices == 0)
	{
		printf("No device found!\n");
		freenect_shutdown(fn_ctx);
		return 1;
	}

	// Open the first device.
	ret = freenect_open_device(fn_ctx, &fn_dev, 0);
	if (ret < 0)
	{
		freenect_shutdown(fn_ctx);
		return 1;
	}
    
    // we can do this either way, gl in main thread or pthread in main
	pthread_create(&freenect_thread, NULL, freenect_threadfunc, NULL);
	gl_threadfunc(NULL);

	printf("Done!\n");
}

/*
int main()
{
    int result = 0;
    // return icp_accuracy_test();
    
    VirtualSensor sensor;
    sensor.init(Configuration::getDataSetPath());

    float sigmaS(2.0);
    float sigmaR(2.0);
    std::cout << "Using sigmaS: " << sigmaS << std::endl;
    std::cout << "Using sigmaR: " << sigmaR << std::endl;

    // Number of subsampling levels
    const unsigned levels = 2;
    // Size of smoothing window
    const unsigned windowSize = 7;
    // Size of subsampling window
    const unsigned blockSize = 3;

    int roomWidthMeter = 6;
    int roomHeightMeter = 6;
    int roomDepthMeter = 6;
    float voxelsPerMeter = 100;
    float scale = 1 / voxelsPerMeter;
    float truncation = 0.125f;
    int numberVoxelsWidth = roomWidthMeter * voxelsPerMeter;
    int numberVoxelsHeight = roomHeightMeter * voxelsPerMeter;
    int numberVoxelsDepth = roomDepthMeter * voxelsPerMeter;
#if EVAL_MODE
    auto gridGenStart = std::chrono::high_resolution_clock::now();
#endif
    VoxelGrid grid(Vector3f(-2.0, -1.0, -2.0), numberVoxelsWidth, numberVoxelsHeight, numberVoxelsDepth, sensor.getDepthImageHeight(), sensor.getDepthImageWidth(), scale, truncation);
#if EVAL_MODE
    auto gridGenEnd = std::chrono::high_resolution_clock::now();
    std::cout << "Setting up grid took: " << std::chrono::duration_cast<std::chrono::milliseconds>(gridGenEnd - gridGenStart).count() << " ms" << std::endl;
#endif
    int idx = 0;
    Matrix4f trajectoryOffset;
    auto totalComputeStart = std::chrono::high_resolution_clock::now();
    while (sensor.processNextFrame())
    {
        auto frameComputeStart = std::chrono::high_resolution_clock::now();
        float *depth = sensor.getDepth();
        // Trajectory:       world -> view space (Extrinsics)
        // InvTrajectory:    view -> world space (Pose)

        if (idx == 0)
        {
            // We express our world space based on the first trajectory (we set the first trajectory to eye matrix, and express all further camera positions relative to that first camera position)
            trajectoryOffset = sensor.getTrajectory().inverse();
        }
        idx++;

#if EVAL_MODE
        auto updateTSDFStart = std::chrono::high_resolution_clock::now();
#endif
        grid.updateTSDF(sensor.getTrajectory() * trajectoryOffset, sensor.getDepthIntrinsics(), depth, sensor.getDepthImageWidth(), sensor.getDepthImageHeight());

#if EVAL_MODE
        auto updateTSDFEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Computing the TSDF update (volumetric fusion) took: " << std::chrono::duration_cast<std::chrono::milliseconds>(updateTSDFEnd - updateTSDFStart).count() << " ms" << std::endl;
        auto pyramidComputeStart = std::chrono::high_resolution_clock::now();
#endif
        PointCloudPyramid pyramid(sensor.getDepth(), sensor.getDepthIntrinsics(), sensor.getTrajectory() * trajectoryOffset, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), levels, windowSize, blockSize, sigmaR, sigmaS);
#if EVAL_MODE
        auto pyramidComputeEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Computing the pyramid took: " << std::chrono::duration_cast<std::chrono::milliseconds>(pyramidComputeEnd - pyramidComputeStart).count() << " ms" << std::endl;
        const std::vector<PointCloud> &cloud = pyramid.getPointClouds();
        auto raycastStart = std::chrono::high_resolution_clock::now();
#endif
        // RaycastImage raycast = grid.raycastVoxelGrid(sensor.getTrajectory() * trajectoryOffset, sensor.getDepthIntrinsics());
#if SAVE_IMAGE_MODE
        if(idx % 50 == 0 || idx > 70 && idx < 100) {
            ImageUtil::saveNormalMapToImage((float*) raycast.normalMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), std::string("RaycastedImage_") + std::to_string(idx), "");
            writeMesh(raycast.vertexMap, sensor.getDepthImageWidth(), sensor.getDepthImageHeight(), Configuration::getOutputDirectory() + "mesh_" + std::to_string(idx) + ".off");
        }
#endif
#if EVAL_MODE
        auto raycastStop = std::chrono::high_resolution_clock::now();
        auto frameComputeEnd = std::chrono::high_resolution_clock::now();
        std::cout << "Computing raycasting took: " << std::chrono::duration_cast<std::chrono::milliseconds>(raycastStop - raycastStart).count() << " ms" << std::endl;
        std::cout << "Computing the frame took: " << std::chrono::duration_cast<std::chrono::milliseconds>(frameComputeEnd - frameComputeStart).count() << " ms" << std::endl;
#endif
    }

    auto totalComputeStop = std::chrono::high_resolution_clock::now();
    std::cout << "Computing for all frames took: " << std::chrono::duration_cast<std::chrono::milliseconds>(totalComputeStop - totalComputeStart).count() << " ms" << std::endl;
#if SAVE_IMAGE_MODE
    auto marchingCubesStart = std::chrono::high_resolution_clock::now();
    run_marching_cubes(grid, idx);
    auto marchingCubesStop = std::chrono::high_resolution_clock::now();
    std::cout << "Computing marching cubes took: " << std::chrono::duration_cast<std::chrono::milliseconds>(marchingCubesStop - marchingCubesStart).count() << " ms" << std::endl;
#endif
    return result;
}
*/
