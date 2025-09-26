#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

struct CostFunction {    //代价函数
    CostFunction(double t, double x_obs, double y_obs, double x0, double y0) : t_(t), x_obs_(x_obs), y_obs_(y_obs), x0_(x0), y0_(y0) {}

    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        const T& vx = params[0];
        const T& vy = params[1];
        const T& g = params[2];
        const T& k = params[3];

        residual[0] = x_obs_ - (x0_ + vx/k * (T(1) - exp(-k * t_)));
        residual[1] = y_obs_ - (y0_ + (vy + g/k)/k * (T(1) - exp(-k * t_)) - g * t_ / k);

        return true;
    }

private:
    double t_, x_obs_, y_obs_, x0_, y0_;
};

std::pair<double, double> detectBallPosition(const cv::Mat& frame) {  //检测球的位置，使用HSV颜色空间掩码和轮廓查找
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    //球颜色检测
    cv::Scalar lower_blue(90, 50, 50);
    cv::Scalar upper_blue(130, 255, 255);

    cv::Mat mask;
    cv::inRange(hsv, lower_blue, upper_blue, mask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return {-1, -1};
    }

    // 最大轮廓
    size_t maxIndex = 0;
    double maxArea = cv::contourArea(contours[0]);
    for (size_t i = 1; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIndex = i;
        }
    }

    cv::Rect boundingBox = cv::boundingRect(contours[maxIndex]);
    cv::rectangle(frame, boundingBox, cv::Scalar(0, 255, 0), 2);

    double centerX = boundingBox.x + boundingBox.width / 2.0;
    double centerY = boundingBox.y + boundingBox.height / 2.0;

    return {centerX, centerY};
}

void processVideo(double& x0, double& y0, std::vector<std::tuple<double, double, double>>& observations, int frame_skip = 5) {  //处理视频，提取球的位置，使用稀疏采样
    cv::VideoCapture cap("resources/video.mp4", cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Video FPS: " << fps << std::endl;
    std::cout << "Using frame skip: " << frame_skip << " (processing every " << frame_skip << "th frame)" << std::endl;

    cv::Mat frame;
    int frameCount = 0;
    bool firstFrame = true;

    while (cap.read(frame)) {
        double time = frameCount / fps;

        
        if (frameCount % frame_skip == 0) {
            auto [x, y] = detectBallPosition(frame);

            if (x >= 0 && y >= 0) {
                if (firstFrame) {
                    x0 = x;
                    y0 = y;
                    firstFrame = false;
                    std::cout << "Initial position: x0 = " << x0 << ", y0 = " << y0 << std::endl;
                } else {
                    observations.emplace_back(time, x, y);
                }
            }
        }

        frameCount++;
    }

    cap.release();
}

void optimizeParameters(double x0, double y0, const std::vector<std::tuple<double, double, double>>& observations, double* params) {  //优化参数，添加残差块
    ceres::Problem problem;

    for (const auto& obs : observations) {
        double t, x_obs, y_obs;
        std::tie(t, x_obs, y_obs) = obs;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CostFunction, 2, 4>(new CostFunction(t, x_obs, y_obs, x0, y0)),
            nullptr,
            params);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;
}

void createVisualizationVideo(double x0, double y0, double vx, double vy, double g, double k, double fps) {  //可视化函数
    cv::VideoCapture cap("resources/video.mp4", cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return;
    }

    int frameWidth = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);  // 获取总帧数

    
    std::vector<cv::Point> trajectoryPoints;
    for (int i = 0; i < totalFrames; ++i) {
        double ti = i / fps;
        double x_pred = x0 + vx / k * (1 - exp(-k * ti));
        double y_pred = y0 + (vy + g / k) / k * (1 - exp(-k * ti)) - g * ti / k;
        trajectoryPoints.push_back(cv::Point(x_pred, y_pred));
    }

    cv::VideoWriter writer("resources/output_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frameWidth, frameHeight));

    cv::Mat frame;
    int frameCount = 0;

    while (cap.read(frame)) {
        
        auto [x_obs, y_obs] = detectBallPosition(frame);

        
        if (trajectoryPoints.size() > 1) {
            for (size_t i = 1; i < trajectoryPoints.size(); ++i) {
                cv::line(frame, trajectoryPoints[i-1], trajectoryPoints[i], cv::Scalar(255, 0, 0), 2);
            }
        }

        // 加参数文本在右下角
        std::string text = cv::format("vx: %.3f vy: %.3f g: %.3f k: %.3f", vx, vy, g, k);
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);
        cv::Point textOrg(frameWidth - textSize.width - 10, frameHeight - textSize.height - 10);
        cv::putText(frame, text, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);

        writer.write(frame);
        frameCount++;
    }

    cap.release();
    writer.release();
    std::cout << "Visualization video saved to resources/output_video.mp4" << std::endl;

    // 转换为 GIF
    std::string cmd = "ffmpeg -i resources/output_video.mp4 -vf \"fps=60,scale=1280:-1:flags=lanczos\" resources/output.gif -y";
    int ret = system(cmd.c_str());
    if (ret == 0) {
        std::cout << "GIF saved to resources/output.gif" << std::endl;
    } else {
        std::cout << "Failed to convert to GIF" << std::endl;
    }
}

int main() {
    double x0, y0;
    std::vector<std::tuple<double, double, double>> observations;

    int frame_skip = 5;  
    processVideo(x0, y0, observations, frame_skip);

    std::cout << "Collected " << observations.size() << " observations from sparse sampling." << std::endl;

    double params[4] = {5.0, 10.0, 9.8, 0.1};
    optimizeParameters(x0, y0, observations, params);

    std::cout << "vx: " << params[0] << " vy: " << params[1] << " g: " << params[2] << " k: " << params[3] << std::endl;

    //可视化（可选）
    //createVisualizationVideo(x0, y0, params[0], params[1], params[2], params[3], 60.0);

    return 0;
}
