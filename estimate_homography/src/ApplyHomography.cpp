#include <mvg.h>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

Mat construct_roi_mask(int height, int width, vector<Point2f> points) {
  Mat mask(height, width, CV_8UC1, cv::Scalar(0));
  vector<vector<Point>> contours;
  contours.push_back(vector<Point>());
  // Careful about the order of the points defining the ROI
  contours[0].push_back(points[0]);
  contours[0].push_back(points[1]);
  contours[0].push_back(points[3]);
  contours[0].push_back(points[2]);
  drawContours(mask, contours, 0, Scalar(255), FILLED, 8);
  return mask;
}

void copyPerspectively(Mat &to, Mat from, Mat homography, std::vector <Point2f> points) {
  int dstHeight = to.size().height;
  int dstWidth = to.size().width;

  Mat world = Mat(dstHeight, dstWidth, to.type());
  Mat mapX = Mat(dstHeight, dstWidth, CV_32FC1);
  Mat mapY = Mat(dstHeight, dstWidth, CV_32FC1);

  for(int x = 0; x < dstWidth; x++) {
    for(int y = 0; y < dstHeight; y++) {
      const float homogeneousPoint[3] = {(float)x, (float)y, 1};
      Mat imagePoint = homography * Mat(3, 1, CV_32FC1, (void*)&homogeneousPoint);
      float imageZ = imagePoint.at<float>(2, 0);
      float imageX = imagePoint.at<float>(0, 0)/imageZ;
      float imageY = imagePoint.at<float>(1, 0)/imageZ;
      if(imageX > 0 && imageY > 0 && imageX < dstWidth && imageY < dstHeight) {
        mapX.at<float>(y, x) = imageX;
        mapY.at<float>(y, x) = imageY;
      } else {
        // Setting the x and y values in the maps to a value outside the image
        // causes them to be treated as "outliers" in the image and the default
        // BORDER_CONSTANT flag maps outliers to 0
        mapX.at<float>(y, x) = dstWidth + 1;
        mapY.at<float>(y, x) = dstHeight + 1;
      }
    }
  }
  remap(from, world, mapX, mapY, INTER_LINEAR);
  world.copyTo(to, construct_roi_mask(dstHeight, dstWidth, points));
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
