#include <mvg.h>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

const Mat SOBEL_X = (Mat_<char>(3, 3) << 1, 0, -1, 2, 0, -2, 1, 0, -1);
const Mat SOBEL_Y = (Mat_<char>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);

// Assuming kernel is odd and square
Mat convolve(Mat image, Mat kernel) {
  assert(image.channels() == 1);

  int kernel_size = kernel.size().width;
  int half_kernel_size = kernel_size / 2;
  int height = image.size().height;
  int width = image.size().width;
  int result_height = height - half_kernel_size * 2;
  int result_width = width - half_kernel_size * 2;

  Mat result = Mat(result_height, result_width, image.type(), Scalar(0));;

  for(int r = half_kernel_size; r <= result_height; r++) {
    for(int c = half_kernel_size; c <= result_width; c++) {
      for(int k_r = 0; k_r < kernel_size; k_r++) {
        for(int k_c = 0; k_c < kernel_size; k_c++) {
          result.at<char>(r - half_kernel_size, c - half_kernel_size) += (
              image.at<char>(r + half_kernel_size - k_r, c + half_kernel_size - k_c) *
              kernel.at<char>(k_r, k_c));
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

  int pad_length = kernel.size().width / 2;

  split(image, src_channels);

  for(int i = 0; i < 3; i++) {
    Mat padded_channel;

    copyMakeBorder(
        src_channels[i],
        padded_channel,
        pad_length,
        pad_length,
        pad_length,
        pad_length,
        BORDER_REPLICATE,
        0);

    Mat result = convolve(padded_channel, kernel);
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

  //Mat test_image = construct_color_image();
  //Mat kernel = (Mat_<float>(3, 3) << 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0);

  // Get gradient in both directions using sobel 
  Mat gradient_x = convolve_color(image, SOBEL_X);
  //Mat gradient_y = convolve_color(image, SOBEL_Y);

  display(gradient_x);

  // Try w = box filter (maybe try it with Gaussian filter too?)
  // Construct M
  // Get trace and determinant of M
  // Find R and compare against threshold
  // Harris with scale?
  return 0;
}

// TODO: Fix memory issue in convolve_color with vector push_back
// TODO: Write code to convole with linearly separable filters
// TODO: Convert image to grayscale before calculating gradient
