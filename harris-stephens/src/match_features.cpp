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

  float k = atof(argv[3]);
  float threshold = atof(argv[4]);
  vector<tuple<KeyPoint, vector<uint8_t>>> corners1 = harris_stephens_corners(image1, k, threshold);
  vector<tuple<KeyPoint, vector<uint8_t>>> corners2 = harris_stephens_corners(image2, k, threshold);

  Mat result;
  vector<KeyPoint> keypoints;
  for(int i = 0; i < corners1.size(); i++) {
    keypoints.push_back(get<0>(corners1[i]));
  }
  drawKeypoints(image1, keypoints, result, Scalar(50));
  write(argv[1], result, "-harris-corners");
  keypoints.clear();
  for(int i = 0; i < corners2.size(); i++) {
    keypoints.push_back(get<0>(corners2[i]));
  }
  drawKeypoints(image2, keypoints, result, Scalar(50));
  write(argv[2], result, "-harris-corners");

  Mat composite_image = create_composite_image(image1, image2);
  vector<tuple<KeyPoint, KeyPoint>> matches = getMatches(corners1, corners2);

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
