#include <mvg.h>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

const Mat SOBEL_X = (Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
const Mat SOBEL_Y = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

Mat construct_box_filter(int size) {
  return Mat(size, size, CV_32FC1, Scalar(1.0 / (size * size)));
}

// Assuming kernel is odd and square
Mat convolve(Mat image, Mat kernel) {
  assert(image.channels() == 1);

  int kernel_size = kernel.size().width;
  int half_kernel_size = kernel_size / 2;

  Mat padded;

  copyMakeBorder(
      image,
      padded,
      half_kernel_size,
      half_kernel_size,
      half_kernel_size,
      half_kernel_size,
      BORDER_REPLICATE,
      0);

  int height = padded.size().height;
  int width = padded.size().width;
  int result_height = height - half_kernel_size * 2;
  int result_width = width - half_kernel_size * 2;

  Mat result = Mat(result_height, result_width, padded.type(), Scalar(0));;

  for(int r = half_kernel_size; r <= result_height; r++) {
    for(int c = half_kernel_size; c <= result_width; c++) {
      for(int k_r = 0; k_r < kernel_size; k_r++) {
        for(int k_c = 0; k_c < kernel_size; k_c++) {
          result.at<uint8_t>(r - half_kernel_size, c - half_kernel_size) += (
              padded.at<uint8_t>(r + half_kernel_size - k_r, c + half_kernel_size - k_c) *
              kernel.at<float>(k_r, k_c));
        }
      }
    }
  }

  return result;
}

Mat convolve_color(Mat image, Mat kernel) {
  assert(image.channels() == 3);

  Mat result, src_channels[3];
  vector<Mat> dst_channels;

  split(image, src_channels);

  for(int i = 0; i < 3; i++) {
    Mat result = convolve(src_channels[i], kernel);
    dst_channels.push_back(result);
  }

  merge(dst_channels, result);

  return result;
}

Mat construct_color_image() {
  Mat channel = (Mat_<float>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);

  vector<Mat> channels;
  for(int i = 0; i < 3; i++) {
    channels.push_back(channel);
  }

  Mat result;
  merge(channels, result);

  return result;
}

void print_color_image(Mat image) {
  assert(image.channels() == 3);

  Mat channels[3];
  split(image, channels);

  for(int i = 0; i < 3; i++) {
    cout << channels[i] << endl;
  }
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("usage: detect_corner <Image>\n");
    return -1;
  }

  Mat image = imread(argv[1], 1);

  Mat grayscale_img;
  cvtColor(image, grayscale_img, COLOR_RGB2GRAY);

  Mat Ix = convolve(grayscale_img, SOBEL_X);
  Mat Iy = convolve(grayscale_img, SOBEL_Y);

  Mat Ixx, Iyy, Ixy;

  multiply(Ix, Ix, Ixx);
  multiply(Ix, Iy, Ixy);
  multiply(Iy, Iy, Iyy);

  // What's a good value for the size of the filter?
  Mat box_filter = construct_box_filter(5);

  Ixx = convolve(Ixx, box_filter);
  Ixy = convolve(Ixy, box_filter);
  Iyy = convolve(Iyy, box_filter);

  float k = 0.05;
  float threshold = 0.01;
  Size size = Ixx.size();
  vector<KeyPoint> corners;

  for(int r = 0; r < size.height; r++) {
    for(int c = 0; c < size.width; c++) {
      int M11 = Ixx.at<uint8_t>(r, c);
      int M12 = Ixy.at<uint8_t>(r, c);
      int M22 = Iyy.at<uint8_t>(r, c);
      int determinant = M11 * M22 - M12 * M12;
      int trace = M11 + M22;
      float R = determinant - k * trace * trace;
      if(R > threshold) {
        corners.push_back(KeyPoint(Point2f(r, c), 1.f));
      }
    }
  }

  Mat result;
  cout << corners.size() << endl;
  drawKeypoints(image, corners, result, Scalar(0));

  display(result);

  return 0;
}

// TODO: Gradient can be negative, can't use unsigned int
// Haar for scale?
// TODO: Write code to convole with linearly separable filters
// TODO: Harris with scale?
