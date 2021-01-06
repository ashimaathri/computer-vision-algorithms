#include <mvg.h>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

const Mat SOBEL_X = (Mat_<float>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
const Mat SOBEL_Y = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

// Assuming kernel is odd and square
Mat convolve(Mat image, Mat kernel, int type) {
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
      BORDER_DEFAULT,
      0);

  int height = padded.size().height;
  int width = padded.size().width;
  int result_height = height - half_kernel_size * 2;
  int result_width = width - half_kernel_size * 2;

  Mat result = Mat(result_height, result_width, type, Scalar(0));;

  for(int r = half_kernel_size; r <= result_height; r++) {
    for(int c = half_kernel_size; c <= result_width; c++) {
      double value = 0;
      for(int k_r = 0; k_r < kernel_size; k_r++) {
        for(int k_c = 0; k_c < kernel_size; k_c++) {
          value += (
              padded.at<uint8_t>(r + half_kernel_size - k_r, c + half_kernel_size - k_c) *
              kernel.at<float>(k_r, k_c));
        }
      }
      switch(type) {
        case CV_32FC1:
          result.at<float>(r - half_kernel_size, c - half_kernel_size) = value;
          break;
        case CV_64FC1:
          result.at<double>(r - half_kernel_size, c - half_kernel_size) = value;
          break;
        default:
          result.at<uint8_t>(r - half_kernel_size, c - half_kernel_size) = value;
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
    Mat result = convolve(src_channels[i], kernel, src_channels[i].type());
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

void real_harris(Mat image) {
  Mat grayscale_img, normalized;
  float k = 0.05;
  float R = 151;
  cvtColor(image, grayscale_img, COLOR_RGB2GRAY);
  Mat result = Mat::zeros(grayscale_img.size(), CV_32FC1);
  cornerHarris(grayscale_img, result, 2, 3, k);
  normalize(result, normalized, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
  vector<KeyPoint> corners;
  for(int i = 0; i < normalized.rows; i++) {
    for(int j = 0; j < normalized.cols; j++) {
      if(normalized.at<float>(i,j) > R) {
        corners.push_back(KeyPoint(Point2f(j, i), 1.f));
      }
    }
  }
  cout << corners.size() << endl;
  drawKeypoints(image, corners, result, Scalar(0));
  display(result);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("usage: detect_corner <Image>\n");
    return -1;
  }

  Mat image = imread(argv[1], 1);
  //real_harris(image);

  Mat grayscale_img;
  cvtColor(image, grayscale_img, COLOR_RGB2GRAY);

  Mat Ix = convolve(grayscale_img, SOBEL_X, CV_32FC1);
  Mat Iy = convolve(grayscale_img, SOBEL_Y, CV_32FC1);

  Mat Ixx, Iyy, Ixy;

  multiply(Ix, Ix, Ixx);
  multiply(Ix, Iy, Ixy);
  multiply(Iy, Iy, Iyy);

  boxFilter(Ixx, Ixx, Ixx.depth(), Size(2, 2), Point(-1, -1), true, BORDER_DEFAULT);
  boxFilter(Ixy, Ixy, Ixy.depth(), Size(2, 2), Point(-1, -1), true, BORDER_DEFAULT);
  boxFilter(Iyy, Iyy, Iyy.depth(), Size(2, 2), Point(-1, -1), true, BORDER_DEFAULT);

  //TODO: Take these as input params
  /*
   * Pair 1
  float k = 0.064;
  float threshold = 67;
  */
  /*
   * Checkerboard
  float k = 0.03;
  float threshold = 120;
  */
  /*
   * Truck 1
  float k = 0.08;
  float threshold = 109;
  */
  float k = 0.072;
  float threshold = 109;
  Size size = Ixx.size();
  Mat R = Mat(image.size(), CV_32FC1);

  for(int r = 0; r < size.height; r++) {
    for(int c = 0; c < size.width; c++) {
      float M11 = Ixx.at<float>(r, c);
      float M12 = Ixy.at<float>(r, c);
      float M22 = Iyy.at<float>(r, c);
      float determinant = M11 * M22 - M12 * M12;
      float trace = M11 + M22;
      R.at<float>(r, c) = determinant - (k * trace * trace);
    }
  }
  Mat R_normalized;
  normalize(R, R_normalized, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
  vector<KeyPoint> corners;
  for(int i = 0; i < R_normalized.rows; i++) {
    for(int j = 0; j < R_normalized.cols; j++) {
      if(R_normalized.at<float>(i,j) > threshold) {
        corners.push_back(KeyPoint(Point2f(j, i), 1.f));
      }
    }
  }

  Mat result;
  cout << corners.size() << endl;
  drawKeypoints(image, corners, result, Scalar(50));

  display(result);
  write(argv[1], result);

  return 0;
}
