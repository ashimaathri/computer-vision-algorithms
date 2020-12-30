#include <math.h>
#include <mvg.h> 
using namespace cv;
using namespace std;

Mat constructPerspectiveTransform() {
  cout << "Enter two imaged ideal points" << endl;

  Mat idealPoint1 = readHomogeneousCoordinates();
  Mat idealPoint2 = readHomogeneousCoordinates();

  Mat vanishingLine = normalize(idealPoint1.cross(idealPoint2));

  return (Mat_<float>(3, 3) << 1, 0, 0, 0, 1, 0,
      vanishingLine.at<float>(0, 0),
      vanishingLine.at<float>(0, 1),
      vanishingLine.at<float>(0, 2));
}

Mat constructAffineTransform(Mat perspectiveTransform) {
  Mat imagedCircle = getConic();
  Mat perspectivelyCorrectedImagedCircle = transformConic(perspectiveTransform, imagedCircle);

  Mat idealPoint1 = (Mat_<float>(3, 1) << 1, 0, 0);
  Mat idealPoint2 = (Mat_<float>(3, 1) << 0, 1, 0);

  Mat circularPoint = intersectLineAndConic(perspectivelyCorrectedImagedCircle,
      idealPoint1, idealPoint2);

  Vec2f x1 = circularPoint.at<Vec2f>(0, 0);
  Vec2f x2 = circularPoint.at<Vec2f>(1, 0);
  float denominator = pow(x2[0], 2) + pow(x2[1], 2);
  float alpha = (x1[0] * x2[0] + x1[1] * x2[1]) / denominator;
  float beta = (x1[1] * x2[0] - x1[0] * x2[1]) / denominator;

  return (Mat_<float>(3, 3) << 1/beta, -alpha/beta, 0, 0, 1, 0, 0, 0, 1);
}

int main(int argc, char** argv ) {
  if (argc != 2) {
    printf("usage: %s <Image_Path>\n", argv[0]);
    return -1;
  }

  Mat image = imread(argv[1], 1);
  if(!image.data) {
    printf("No image data \n");
    return -1;
  }

  Mat perspectiveTransform = constructPerspectiveTransform();
  Mat affineTransform = constructAffineTransform(perspectiveTransform);
  Mat homography = (affineTransform * perspectiveTransform).inv();

  write(argv[1],
      changePerspective(image, homography * computeScaling(image, homography)),
      string("-circle"));
  return 0;
}
