#include<iostream>
#include<conio.h>
#include<fstream>
#include<string>
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>

#include "version.h"
#include "Matrix.hpp"
#include "HomographyMatrix.hpp"

#define THREE_VIDEO_STITCHING 0
#define USE_OPENGL_PBO 0

using namespace std;
using namespace cv;
using namespace inastitch::json;

void frameBufferCallback(GLFWwindow*, int width, int height);
void processInput(GLFWwindow*);
string readFile(const char*);
void createShader(GLenum, uint32_t&, const char*);
void computeHomography(Mat, Mat, float[][3], bool);

constexpr float ratio = 0.75;
constexpr float reprojThresh = 4.0;
const auto minMatchCount = 25;
const uint32_t windowHeight = 1080;
const uint32_t windowWidth = 1920;

GLFWwindow* window;
uint32_t VAO = 0,
VBO = 0,
vertexShader = 0,
fragmentShader = 0,
shaderProgram = 0;
uint32_t textures[3];

static const GLchar* vertexShaderSource = R""""(
#version 330 core

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 texCoord;

out vec2 texCoordVar;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main() {
   gl_Position = proj * view * model * vec4(position.x, position.y, 0.0f, 1.0f);
   texCoordVar = texCoord;
}
)"""";
static const GLchar* fragmentShaderSource = R""""(
#version 330 core

in vec2 texCoordVar;
out vec4 fragColor;

uniform sampler2D texture1;
uniform mat3 warp;

void main() {
   vec3 dst = warp * vec3(texCoordVar.x + 1, texCoordVar.y, 1.0);
   fragColor = texture(texture1, vec2(dst.x / dst.z, dst.y / dst.z));
}
)"""";
GLfloat offX = 0, offY = 0;

static const GLfloat topRightX = -0.480f - offX, topRightY = 0.360f + offY;
static const GLfloat bottomRightX = -0.480f - offX, bottomRightY = -0.360f - offY;
static const GLfloat bottomLeftX = 0.480f + offX, bottomLeftY = -0.360f - offY;
static const GLfloat topLeftX = 0.480f + offX, topLeftY = 0.360f + offY;
static const GLfloat vertices[] = {
	// position (2D)            // texCoord
	topRightX,    topRightY,    0.0f, 0.0f,
	bottomRightX, bottomRightY, 0.0f, 1.0f,
	bottomLeftX,  bottomLeftY,  1.0f, 1.0f,

	bottomLeftX,  bottomLeftY,  1.0f, 1.0f,
	topLeftX,     topLeftY,     1.0f, 0.0f,
	topRightX,    topRightY,    0.0f, 0.0f
};

int main(int argc, const char** argv)
{
	GLuint glShaderProgram, glVextexBufferObject;
	GLint glShaderPositionAttrib, glShaderTexCoordAttrib;
	GLint glShaderModelMatrixUni, glShaderViewMatrixUni, glShaderProjMatrixUni;
	GLint glShaderWarpMatrixUni;
	GLFWwindow* glWindow;

    VideoCapture vid1 = VideoCapture("C:\\Users\\adhar\\source\\repos\\TriangleV4\\x64\\Release\\1.mp4");
    VideoCapture vid2 = VideoCapture("C:\\Users\\adhar\\source\\repos\\TriangleV4\\x64\\Release\\2.mp4");
    VideoCapture vid3 = VideoCapture("C:\\Users\\adhar\\source\\repos\\TriangleV4\\x64\\Release\\3.mp4");
    Mat center, right, left;
    if (!(vid1.isOpened() && vid2.isOpened() && vid3.isOpened())) { cout << "Unable to open video files!" << endl; return -1; }
    vid1.read(center);
    vid2.read(right);
    vid3.read(left);

    cout << "[*]Video Files opened!" << endl;
    uint32_t textureWidth = center.rows;
    uint32_t textureHeight = center.cols;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glWindow = glfwCreateWindow(windowWidth, windowHeight, "Image Stitch", NULL, NULL);
    glfwSetFramebufferSizeCallback(glWindow, frameBufferCallback);
    glfwMakeContextCurrent(glWindow);
    //glfwSwapInterval(1); // VSync enabled. Comment this line to disable VSync.

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) return -1;

    const auto glVersion = glGetString(GL_VERSION);
    const auto glRenderer = glGetString(GL_RENDERER);

    std::cout << "GL_VERSION  : " << glVersion << std::endl;
    std::cout << "GL_RENDERER : " << glRenderer << std::endl;

    if ((glVersion == nullptr) || (glRenderer == nullptr))
    {
        std::cerr << "No OpenGL" << std::endl
            << "Note 1: if you run this command in a SSH terminal, you can specify "
            << "the display with DISPLAY=:0 for example." << std::endl
            << "Note 2: if you run this command on a headless system (i.e., no screen attached), "
            << "hardware acceleration might not be possible. "
            << "As a workaround, use LIBGL_ALWAYS_SOFTWARE=1 to force software rendering." << std::endl
            << "Aborting..." << std::endl;
        std::abort();
    }


    
    glShaderProgram = glCreateProgram();
    createShader(GL_VERTEX_SHADER, vertexShader, vertexShaderSource);
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        return -1;
    };
    cout << "[*]Vertex Shader Compiled!" << endl;
    createShader(GL_FRAGMENT_SHADER, fragmentShader, fragmentShaderSource);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        return -1;
    };
    cout << "[*]Fragment Shader Compiled!" << endl;
    glAttachShader(glShaderProgram, vertexShader);
    glAttachShader(glShaderProgram, fragmentShader);
    glLinkProgram(glShaderProgram);
    glGetProgramiv(glShaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        return -1;
    }
    cout << "[*]Shader Program Linked!" << endl;
    glUseProgram(glShaderProgram);
    glShaderPositionAttrib = glGetAttribLocation(glShaderProgram, "position");
    glShaderTexCoordAttrib = glGetAttribLocation(glShaderProgram, "texCoord");

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glViewport(0, 0, windowWidth, windowHeight);

    uint32_t VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glGenBuffers(1, &glVextexBufferObject);
    glBindBuffer(GL_ARRAY_BUFFER, glVextexBufferObject);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(glShaderPositionAttrib, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (GLvoid*)0);
    glEnableVertexAttribArray(glShaderPositionAttrib);
    glVertexAttribPointer(glShaderTexCoordAttrib, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (GLvoid*)(2 * sizeof(float)));
    glEnableVertexAttribArray(glShaderTexCoordAttrib);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GLuint texture0;
    glGenTextures(1, &texture0);
    glBindTexture(GL_TEXTURE_2D, texture0); // bind
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    cout << "[*]Texture 0 created!" << endl;

    const auto pixelSize = 4; // RGBA
    const auto pboBufferSize = windowWidth * windowHeight * pixelSize;
#if USE_OPENGL_PBO
    // PBO
    const auto pboCount = 2;
    unsigned int pboIds[pboCount];
    glGenBuffers(pboCount, pboIds);
    for (int i = 0; i < pboCount; i++)
    {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pboIds[i]);
        glBufferData(GL_PIXEL_PACK_BUFFER, pboBufferSize, nullptr, GL_STREAM_READ);
    }
#endif

    const glm::mat4 identMat4 = glm::mat4(1.0f);
    // Model
    glm::mat4 modelMat[3] = { identMat4, identMat4, identMat4 };

    // View
    const glm::mat4 initialViewMat = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -0.460f));
    glm::mat4 viewMat[3] = { initialViewMat, initialViewMat, initialViewMat };
    // glm matrice element access: matrix[colIdx][rowIdx] = value
    // shift left
    viewMat[0][3][0] = 0.480f;
    viewMat[1][3][0] = -0.480f;

    // Projection
    glm::mat4 projMat = glm::perspective(glm::radians(100.0f), static_cast<float>(windowWidth) / windowHeight, 0.1f, 200.0f);

    const glm::mat3 identMat3 = glm::mat3(1.0f);
    glm::mat3 texWarpMat[3] = { identMat3, identMat3, identMat3 };

    float matrix1[3][3];
   
    resize(center, center, Size(textureWidth, textureHeight), INTER_AREA);
    resize(right, right, Size(textureWidth, textureHeight), INTER_AREA);
    
    cvtColor(center, center, COLOR_BGR2RGB);
    cvtColor(right, right, COLOR_BGR2RGB);
   
    flip(center, center, 1);
    flip(right, right, 1);
    

    computeHomography(center, right, matrix1, false);

    modelMat[1][0][0] = -1.0f;

    for (uint32_t rowIdx = 0; rowIdx < 3; rowIdx++)
    {
        for (uint32_t colIdx = 0; colIdx < 3; colIdx++)
        {
            // Note: texWarpMat is "column first"

            texWarpMat[1][colIdx][rowIdx] = matrix1[rowIdx][colIdx];

            //texWarpMat[2][colIdx][rowIdx] = matrix2[rowIdx][colIdx];
        }
    }

    

    cout << "[*]Entering render loop" << endl;
    VideoWriter output("output.mp4", VideoWriter::fourcc('X', '2', '6', '4'), 30, Size(windowWidth, windowHeight));
    bool isRecording = true;
    while (!glfwWindowShouldClose(glWindow))
    {
        auto t1 = chrono::high_resolution_clock::now(); // start time for FPS
        glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(glShaderProgram);
        // vertex shader
        glShaderModelMatrixUni = glGetUniformLocation(glShaderProgram, "model");
        glShaderViewMatrixUni = glGetUniformLocation(glShaderProgram, "view");
        glShaderProjMatrixUni = glGetUniformLocation(glShaderProgram, "proj");
        // pixel shader
        glShaderWarpMatrixUni = glGetUniformLocation(glShaderProgram, "warp");

        glBindTexture(GL_TEXTURE_2D, texture0);

        if ((!vid1.read(center)) || (!vid2.read(right)) /* || (!vid3.read(left)) */)
        {
            vid1.release();
            vid2.release();
            vid3.release();
            vid1 = VideoCapture("C:\\Users\\adhar\\source\\repos\\TriangleV4\\x64\\Release\\1.mp4");
            vid2 = VideoCapture("C:\\Users\\adhar\\source\\repos\\TriangleV4\\x64\\Release\\2.mp4");
            //vid3 = VideoCapture("C:\\Users\\adhar\\source\\repos\\TriangleV4\\x64\\Release\\3.mp4");
            if (isRecording)
            {
                output.release();
                isRecording = false;
                cout << "[*]Recording completed!" << endl;
            }
            continue;
        }
        resize(center, center, Size(textureWidth, textureHeight), INTER_AREA);
        resize(right, right, Size(textureWidth, textureHeight), INTER_AREA);
        //resize(left, left, Size(textureWidth, textureHeight), INTER_AREA);
        cvtColor(center, center, COLOR_BGR2RGBA);
        cvtColor(right, right, COLOR_BGR2RGBA);
        //cvtColor(left, left, COLOR_BGR2RGBA);
        //flip(center, center, 1);
        flip(right, right, 1);
        //flip(left, left, 1);
          
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureWidth, textureHeight, GL_RGBA, GL_UNSIGNED_BYTE, center.data);
        glUniformMatrix4fv(glShaderModelMatrixUni, 1, GL_FALSE, glm::value_ptr(modelMat[0]));
        glUniformMatrix4fv(glShaderViewMatrixUni, 1, GL_FALSE, glm::value_ptr(viewMat[0]));
        glUniformMatrix4fv(glShaderProjMatrixUni, 1, GL_FALSE, glm::value_ptr(projMat));
        glUniformMatrix3fv(glShaderWarpMatrixUni, 1, GL_FALSE, glm::value_ptr(texWarpMat[0]));
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureWidth, textureHeight, GL_RGBA, GL_UNSIGNED_BYTE, right.data);
        glUniformMatrix4fv(glShaderModelMatrixUni, 1, GL_FALSE, glm::value_ptr(modelMat[1]));
        glUniformMatrix4fv(glShaderViewMatrixUni, 1, GL_FALSE, glm::value_ptr(viewMat[1]));
        glUniformMatrix4fv(glShaderProjMatrixUni, 1, GL_FALSE, glm::value_ptr(projMat));
        glUniformMatrix3fv(glShaderWarpMatrixUni, 1, GL_FALSE, glm::value_ptr(texWarpMat[1]));
        glDrawArrays(GL_TRIANGLES, 0, 6);
        
        /*
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureWidth, textureHeight, GL_RGBA, GL_UNSIGNED_BYTE, left.data);
        glUniformMatrix4fv(glShaderModelMatrixUni, 1, GL_FALSE, glm::value_ptr(modelMat[2]));
        glUniformMatrix4fv(glShaderViewMatrixUni, 1, GL_FALSE, glm::value_ptr(viewMat[2]));
        glUniformMatrix4fv(glShaderProjMatrixUni, 1, GL_FALSE, glm::value_ptr(projMat));
        glUniformMatrix3fv(glShaderWarpMatrixUni, 1, GL_FALSE, glm::value_ptr(texWarpMat[2]));
        glDrawArrays(GL_TRIANGLES, 0, 6);
        */

        glfwSwapBuffers(glWindow);
        auto t2 = chrono::high_resolution_clock::now(); // end time for FPS
        if (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count() % 10 == 0) // so that FPS is not calculated every cycle.
        {
            auto duration = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
            double fps = (double)1000 / duration;
            for (int i = 0; i < 80; i++) cout << "\b";
            
            cout << "FPS: " << fps << " Execution Time: " << duration << " ms";
        }
        //waitKey(5000);
        
        if (isRecording)
        {
            vector<unsigned char> pixels((windowWidth) * (windowHeight) * 3);
            glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, &pixels[0]);
            Mat frame(windowHeight, windowWidth, CV_8UC3, pixels.data());
            cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            flip(frame, frame, 0);
            //imshow("Test", frame);
            output.write(frame);
        }
    }
    //output.release();
    glfwTerminate();
    cout << "[*]Resources released" << endl;

	return 0;
}

void frameBufferCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, windowWidth, windowHeight);
	glfwSwapBuffers(window);
}
void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}
string readFile(const char* fileName)
{
	ifstream inFile(fileName, ios::in);
	string result = "", temp = "";
	while (getline(inFile, temp))
	{
		result += temp + "\n";
	}
	inFile.close();
	return result;
}
void createShader(GLenum type, uint32_t& shader, const char* source)
{
	shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, NULL);
	glCompileShader(shader);
}
void computeHomography(Mat center, Mat right, float matrix[][3], bool isFlipped)
{
    //float matrix[3][3];

    std::function detectAndDescribe = [](const cv::Mat& image, const char* imgDesc)
    {
        // convert the image to grayscale
        cv::Mat grayImage;
        cv::cvtColor(image, grayImage, cv::COLOR_RGB2GRAY);

        // detect and extract features from the image
        // https://docs.opencv.org/4.4.0/d7/d60/classcv_1_1SIFT.html
        auto desc = cv::SIFT::create();

        const auto t1 = std::chrono::high_resolution_clock::now();
        // https://docs.opencv.org/4.4.0/d0/d13/classcv_1_1Feature2D.html
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat features;
        desc->detectAndCompute(grayImage, cv::noArray(), keypoints, features);
        const auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "detectAndCompute:"
            << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us"
            << std::endl;

        return std::make_tuple(keypoints, features);
    };

    auto [kpsC, featC] = detectAndDescribe(center, "C");
    auto [kpsR, featR] = detectAndDescribe(right, "R");
    cout << "[*]Keypoints calculated" << endl;

    std::cout << "center: keypointCount=" << kpsC.size() << std::endl;
    std::cout << "right : keypointCount=" << kpsR.size() << std::endl;

    auto matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::MatcherType
        ::BRUTEFORCE);
    const auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<cv::DMatch>> rawMatchesR;
    matcher->knnMatch(featR /* query */, featC /* train */, rawMatchesR, 2 /* two best matches */);
    const auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "knnMatch:"
        << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us"
        << std::endl;

    std::cout << "rawMatchCount=" << rawMatchesR.size() << std::endl;

    std::function loopOverRawMatches = [&](
        const std::vector<std::vector<cv::DMatch>>& rawMatches,
        float ratio)
    {
        std::vector<cv::DMatch> matches;

        for (const auto& m : rawMatches)
        {
         
            if ((m.size() == 2) && (m[0].distance < m[1].distance * ratio))
            {
                matches.push_back(m[0]);
            }
        }

        return matches;
    };

    auto matchesR = loopOverRawMatches(rawMatchesR, 0.75);
    const auto minMatchCount = 25;
    std::cout << "matchCount=" << matchesR.size() << std::endl;

    std::function computeHomographyMatrix = [minMatchCount](
        const std::vector<cv::DMatch>& matches,
        const std::vector<cv::KeyPoint>& kpsC,
        const std::vector<cv::KeyPoint>& kpsR,
        float reprojThresh)
    {
        cv::Mat homoM;
        // homography matrix requires at least 4 matches
        if (matches.size() > minMatchCount)
        {
            // construct the two sets of points
            // https://docs.opencv.org/trunk/d2/d29/classcv_1_1KeyPoint.html
            std::vector<cv::Point2f> ptsC, ptsR;
            for (const auto& m : matches)
            {
                ptsC.push_back(kpsC.at(m.queryIdx).pt);
                ptsR.push_back(kpsR.at(m.trainIdx).pt);
            }

            std::cout << "PointsC=" << ptsC.size() << std::endl;
            std::cout << "PointsR=" << ptsR.size() << std::endl;

            // compute the homography between the two sets of points
            // https://docs.opencv.org/4.4.0/d9/d0c/group__calib3d.html
            const auto t1 = std::chrono::high_resolution_clock::now();
            homoM = cv::findHomography(ptsR, ptsC, cv::RANSAC, reprojThresh);
            const auto t2 = std::chrono::high_resolution_clock::now();
            std::cout << "findHomography:"
                << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << "us"
                << std::endl;
        }

        return homoM;
    };

    auto homoMatrixR = computeHomographyMatrix(matchesR, kpsR, kpsC, reprojThresh);
    cout << "[*]Homography Matrix calculated" << endl;

    std::cout << "Original OpenCV homography matrix:" << std::endl;
    std::cout << homoMatrixR << std::endl;


    cv::Mat homoMatrixRInv;
    cv::invert(homoMatrixR, homoMatrixRInv);

    std::cout << "Inverted OpenCV homography matrix:" << std::endl;
    std::cout << homoMatrixRInv << std::endl;


    cv::Mat panoR;
    cv::warpPerspective(
        right, panoR, homoMatrixRInv,
        { center.cols + right.cols, right.rows },
        // cv::Size = ( maxCols, maxRows )
        cv::INTER_LINEAR //| cv::WARP_INVERSE_MAP
    );
    cv::imwrite("sample_panoR.jpg", panoR);

    // Build panorama
    const auto panoWidth = center.cols + right.cols;
    const auto panoHeight = center.rows;
    cv::Mat pano(panoHeight, panoWidth, center.type());
    pano.setTo(0);

    panoR.copyTo(pano);

    center.copyTo(pano(
        cv::Rect(0, 0, center.cols, center.rows)));

    cv::imwrite("sample_pano.jpg", pano);


    float xFlipAndShiftMatrix_data[] = {
         -1.0, 0.0, 1.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0
    };
    cv::Mat xFlipAndShiftMatrix(3, 3, CV_32F, xFlipAndShiftMatrix_data);

    float scale1Matrix_data[] = {
         (right.cols * 1.0f), 0.0, 0.0,
         0.0, (right.rows * 1.0f), 0.0,
         0.0, 0.0, 1.0
    };
    cv::Mat scale1Matrix(3, 3, CV_32F, scale1Matrix_data);

    float descaleMatrix_data[] = {
         1.0f / (right.cols * 1.0f), 0.0, 0.0,
         0.0, 1.0f / (right.rows * 1.0f), 0.0,
         0.0, 0.0, 1.0
    };
    cv::Mat descaleMatrix(3, 3, CV_32F, descaleMatrix_data);

    // Mat needs to be of the same type
    homoMatrixR.convertTo(homoMatrixR, CV_32F);

    cv::Mat homoMatrixRInvNorm = descaleMatrix * homoMatrixR * scale1Matrix;

    if(isFlipped) homoMatrixRInvNorm = xFlipAndShiftMatrix * homoMatrixRInvNorm;
    cout << "[*]OpenGL ready homography matrix calculated" << endl;

    std::cout << "OpenGL-ready homography matrix:" << std::endl;
    std::cout << homoMatrixRInvNorm << std::endl;

    // write back result to output parameter
    for (uint32_t row = 0; row < 3; row++) {
        for (uint32_t col = 0; col < 3; col++) {
            matrix[row][col] = homoMatrixRInvNorm.at<float>(row, col);
        }
    }
}

