#include <mvg.h>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

Mat convolve(Mat image, Mat kernel) {
  Mat result;

  return result;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("usage: detect_corner <Image>\n");
    return -1;
  }

  Mat image = imread(argv[1], 1);
  // Get gradient in both directions using sobel 
  // Try w = box filter (maybe try it with Gaussian filter too?)
  // Construct M
  // Get trace and determinant of M
  // Find R and compare against threshold
  // Harris with scale?
  return 0;
}
