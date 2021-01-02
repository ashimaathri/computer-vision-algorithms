#include <mvg.h>
#include <math.h>
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

Point2f transform_point(Mat homography, float x, float y) {
  Point2f transformed_point;

  Mat homogeneous_src_point = (Mat_<float>(3, 1) << x, y, 1);
  Mat homogeneous_dst_point = homography * homogeneous_src_point;

  float w = homogeneous_dst_point.at<float>(2, 0);
  transformed_point.x = homogeneous_dst_point.at<float>(0, 0) / w;
  transformed_point.y = homogeneous_dst_point.at<float>(1, 0) / w;

  return transformed_point;
}

// Very slow because it goes pixel by pixel instead of using GPU matrix
// operations
Vec3b interpolate(Mat image, Point2f point) {
  float int_x, int_y;
  float frac_x = modf(point.x, &int_x);
  float frac_y = modf(point.y, &int_y);

  Vec3b f11 = image.at<Vec3b>(int_y, int_x);
  Vec3b f12 = image.at<Vec3b>(int_y + 1, int_x);
  Vec3b f21 = image.at<Vec3b>(int_y, int_x + 1);
  Vec3b f22 = image.at<Vec3b>(int_y + 1, int_x + 1);
  Vec3b interpolated_value;

  Mat x_weights = (Mat_<float>(1, 2) << (1 - frac_x), frac_x);
  Mat y_weights = (Mat_<float>(2, 1) << (1 - frac_y), frac_y);

  for(int i = 0; i < image.channels(); i++) {
    Mat neighbor_intensities = (Mat_<float>(2, 2) << f11[i], f12[i], f21[i], f22[i]);
    Mat result = x_weights * neighbor_intensities * y_weights;
    interpolated_value[i] = result.at<float>(0, 0);
  }

  return interpolated_value;
}

Mat copyPerspectively(Mat to, Mat from, Mat homography, std::vector <Point2f> points) {
  int dstHeight = to.size().height;
  int dstWidth = to.size().width;
  int srcHeight = from.size().height;
  int srcWidth = from.size().width;

  Mat dst_x = Mat(dstHeight, dstWidth, CV_32FC1);
  Mat dst_y = Mat(dstHeight, dstWidth, CV_32FC1);
  Mat transformed_src = Mat(dstHeight, dstWidth, to.type());

  for(int x = 0; x < dstWidth; x++) {
    for(int y = 0; y < dstHeight; y++) {
      Point2f src_point = transform_point(homography, x, y);
      // Effectively doesn't interpolate the borders and replaces them with 0s
      if(src_point.x > 0 && src_point.y > 0 && src_point.x < srcWidth - 1 && src_point.y < srcHeight - 1) {
        dst_x.at<float>(y, x) = src_point.x;
        dst_y.at<float>(y, x) = src_point.y;
        transformed_src.at<Vec3b>(y, x) = interpolate(from, src_point);
      } else {
        // Setting the x and y values in the maps to a value outside the image
        // causes them to be treated as "outliers" in the image and the default
        // BORDER_CONSTANT flag maps outliers to 0
        dst_x.at<float>(y, x) = dstWidth + 1;
        dst_y.at<float>(y, x) = dstHeight + 1;
        transformed_src.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
      }
    }
  }

  Mat final_image = to.clone();
  transformed_src.copyTo(final_image, construct_roi_mask(dstHeight, dstWidth, points));
  return final_image;
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

  Mat modified_image = copyPerspectively(dst_image, src_image, homography, dst_points);
  write(argv[1], modified_image);
  return 0;
}
