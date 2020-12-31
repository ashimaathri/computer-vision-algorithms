#include <mvg.h>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

void copyPerspectively(Mat &to, Mat from, Mat homography, std::vector <Point2f> points) {
  int toHeight = to.size().height;
  int toWidth = to.size().width;
  int fromHeight = from.size().height;
  int fromWidth = from.size().width;

  Mat world = Mat(toHeight, toWidth, to.type());
  Mat mapX = Mat(toHeight, toWidth, CV_32FC1);
  Mat mapY = Mat(toHeight, toWidth, CV_32FC1);
  for(int x = 0; x < toWidth; x++) {
    for(int y = 0; y < toHeight; y++) {
      const float homogeneousPoint[3] = {(float)x, (float)y, 1};
      Mat imagePoint = homography * Mat(3, 1, CV_32FC1, (void*)&homogeneousPoint);
      float imageZ = imagePoint.at<float>(2, 0);
      float imageX = imagePoint.at<float>(0, 0)/imageZ;
      float imageY = imagePoint.at<float>(1, 0)/imageZ;
      if(imageX > 0 && imageY > 0 && imageX < toWidth && imageY < toHeight) {
        mapX.at<float>(y, x) = imageX;
        mapY.at<float>(y, x) = imageY;
      } else {
        // Setting the x and y values in the maps to a value outside the image
        // causes them to be treated as "outliers" in the image and the default
        // BORDER_CONSTANT flag maps outliers to 0
        mapX.at<float>(y, x) = toWidth + 1;
        mapY.at<float>(y, x) = toHeight + 1;
      }
    }
  }
  remap(from, world, mapX, mapY, INTER_LINEAR);
  Mat mask(toHeight, toWidth, CV_8UC1, cv::Scalar(0));
  vector<vector<Point>> contours;
  contours.push_back(vector<Point>());
  contours[0].push_back(points[0]);
  contours[0].push_back(points[1]);
  contours[0].push_back(points[3]);
  contours[0].push_back(points[2]);
  drawContours(mask, contours, 0, Scalar(255), FILLED, 8);
  world.copyTo(to, mask);
}

int main(int argc, char** argv )
{
  if (argc != 3) {
    printf("usage: ApplyHomography <Destination Image> <Source Image>\n");
    return -1;
  }

  Mat dst_image = imread(argv[1], 1);
  Mat src_image = imread(argv[2], 1);

  vector<Point2f> dst_points, src_points;
  receivePointCorrespondence(dst_points, "destination");
  receivePointCorrespondence(src_points, "source");

  Mat homography = computeHomography(dst_points, src_points);

  copyPerspectively(dst_image, src_image, homography, dst_points);
  write(argv[1], dst_image);
  return 0;
}
