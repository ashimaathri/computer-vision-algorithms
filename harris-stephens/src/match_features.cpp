#include <mvg.h>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

// OpenCV harris for comparison of results
// Both algos give exactly the same results
void opencv_harris(Mat image) {
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
  if (argc != 5) {
    printf("usage: match_features <Image1> <Image2> <k> <threshold>\n");
    return -1;
  }

  Mat image1 = imread(argv[1], 1);
  Mat image2 = imread(argv[2], 1);
  Mat grayscale_img1, grayscale_img2;
  cvtColor(image1, grayscale_img1, COLOR_RGB2GRAY);
  cvtColor(image2, grayscale_img2, COLOR_RGB2GRAY);

  float k = atof(argv[3]);
  float threshold = atof(argv[4]);
  vector<KeyPoint> corners1 = harris_stephens_corners(grayscale_img1, k, threshold);
  vector<KeyPoint> corners2 = harris_stephens_corners(grayscale_img2, k, threshold);

  Mat result;
  drawKeypoints(image1, corners1, result, Scalar(50));
  write(argv[1], result, "-harris-corners");
  drawKeypoints(image2, corners2, result, Scalar(50));
  write(argv[2], result, "-harris-corners");

  Mat composite_image = create_composite_image(image1, image2);
  vector<tuple<KeyPoint, KeyPoint>> matches = getMatches(grayscale_img1, corners1, grayscale_img2, corners2, &ssd);

  Point2f offset = Point2f(image1.cols, 0);
  for(auto it = matches.begin(); it < matches.end(); it++) {
    KeyPoint kp1 = get<0>(*it);
    KeyPoint kp2 = get<1>(*it);
    line(composite_image, kp1.pt, kp2.pt + offset, Scalar(255), 2, LINE_8);
  }

  char composite_name[8] = "top-100";
  display(composite_image);
  write(composite_name, composite_image, "-harris-corner-matches");

  return 0;
}
