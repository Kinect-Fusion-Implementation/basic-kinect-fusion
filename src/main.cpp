#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <math.h>

#include <GL/glut.h>
#include "libfreenect.h"
#include "libfreenect_registration.h"
#include "./kinect_fusion/Eigen.h"
#include "PointCloud.h"
#include "PointCloudPyramid.h"
#include "./configuration/Configuration.h"
#include "ICPOptimizer.h"
#include "CudaVoxelGrid.h"
#include "../visualization/MarchingCubes.h"
#include "./visualization/PointCloudToMesh.h"

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

float sigmaS(2.0);
float sigmaR(2.0);

// Number of subsampling levels -> Without the basic level, the pyramid will contain subLevels + 1 point clouds
const unsigned subLevels = 2;
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
// x,y,z: width, height, depth
VoxelGrid grid;
// On level 0, 1, 2
std::vector<int> iterations_per_level = {4, 4, 3};
ICPOptimizer optimizer;

Matrix4f prevFrameToGlobal = Matrix4f::Identity();

unsigned int idx = 0;

int run_kinect_fusion(float vertexDiffThreshold, float normalDiffThreshold, float *depth, uint8_t *image)
{
        Matrix3f intrinsics;
        intrinsics << 525.0f, 0.0f, 319.5f,
            0.0f, 525.0f, 239.5f,
            0.0f, 0.0f, 1.0f;
        // Trajectory:       world -> view space (Extrinsics)
        // InvTrajectory:    view -> world space (Pose)

        if (idx == 0)
        {
                // Update the TSDF w.r.t. the first camera frame C0 (all other poses/extrinsics are expressions relative to C0)
                grid.updateTSDF(Matrix4f::Identity(), intrinsics, depth, 640, 480);
                idx++;
                return 0;
        }

        PointCloudPyramid pyramid(depth, intrinsics, 640, 480, 2, windowSize, blockSize, sigmaR, sigmaS);
        ImageUtil::saveDepthMapToImage(depth, 640, 480, "test", "");
        RaycastImage raycast = grid.raycastVoxelGrid(prevFrameToGlobal.inverse(), intrinsics);

#pragma omp parallel for collapse(2)
        for (size_t i = 0; i < 640; i++)
        {
                for (size_t j = 0; j < 480; j++)
                {
                        size_t index = i + j * 640;
                        image[3 * index + 0] = std::min(std::max(0, (int)(255.0f * pyramid.getPointClouds().at(0).getNormalsCPU()[(index)].x())), 255);
                        image[3 * index + 1] = std::min(std::max(0, (int)(255.0f * pyramid.getPointClouds().at(0).getNormalsCPU()[(index)].y())), 255);
                        image[3 * index + 2] = std::min(std::max(0, (int)(255.0f * pyramid.getPointClouds().at(0).getNormalsCPU()[(index)].z())), 255);
                }
        }
        // Estimate the pose of the current frame
        Matrix4f estPose;

        estPose = optimizer.optimize(pyramid, raycast.m_vertexMapGPU, raycast.m_normalMapGPU, prevFrameToGlobal, idx);
        // Use estimated pose as prevPose for next frame
        prevFrameToGlobal = estPose;

        grid.updateTSDF(estPose.inverse(), intrinsics, depth, 640, 480);
        return 0;
}


std::vector<float> thresholds = {0.2, 0.1, 0.08, 0.06, 0.04, 0.02, 0.01, 0.008};
// 0.02 is best for xyz (5, 5), 0.04 best for rpy (4, 4)
float pointThreshold = thresholds.at(5);
float normalThreshold = thresholds.at(5);

void DrawGLScene()
{
        pthread_mutex_lock(&buffer_mutex);

        while ((!got_depth && !got_rgb))
        {
                pthread_cond_wait(&gl_frame_cond, &buffer_mutex);
        }

        uint8_t *tmp;
        float *tmp_depth;

        if (got_depth)
        {
                // Swap front and mid buffer -> Mid buffer contains the new data, that is why we swap
                tmp_depth = depth_front;
                depth_front = depth_mid;
                depth_mid = tmp_depth;
                got_depth = false;
                run_kinect_fusion(pointThreshold, normalThreshold, depth_front, depth_img_front);
                tmp = depth_img_front;
                depth_img_front = depth_img_mid;
                depth_img_mid = tmp;
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
        uint16_t *depth = (uint16_t *)v_depth;

        pthread_mutex_lock(&buffer_mutex);

#pragma omp parallel for
        for (int i = 0; i < 640 * 480; i++)
        {
                depth_mid[i] = depth[i] / 1000.0;
                // std::cout << depth_mid[i];
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
        Matrix3f intrinsics;
        intrinsics << 525.0f, 0.0f, 319.5f,
            0.0f, 525.0f, 239.5f,
            0.0f, 0.0f, 1.0f;
        grid = VoxelGrid(Vector3f(-3.0, -3.0, -3.0), numberVoxelsWidth, numberVoxelsHeight, numberVoxelsDepth, 480, 640, scale, truncation);
        optimizer = ICPOptimizer(intrinsics, 640, 480, pointThreshold, normalThreshold, iterations_per_level, 0.2f);

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
}
