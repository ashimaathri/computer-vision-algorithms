#include "mvg.h"
#include <cstring>
using namespace cv;
using namespace std;

Mat solveConicSystem(vector<Mat> points) {
  int numLines = points.size();

  assert(numLines == 5);
  const int numRows = 5;
  const int numCols = 6;

  float linearSystem[numRows][numCols];

  int sizeOfRow = numCols * sizeof(float);

  for(int i = 0; i < 5; i++) {
    float x1 = points[i].at<float>(0, 0);
    float x2 = points[i].at<float>(0, 1);
    float x3 = points[i].at<float>(0, 2);
    const float row[numCols] = {x1 * x1, x1 * x2, x2 * x2, x1, x2, 1};
    memcpy(linearSystem + i, (void*)&row, sizeOfRow);
  }

  Mat coefficients = getNullSpace((void*)&linearSystem, numRows, numCols);
  float a = coefficients.at<float>(0, 0);
  float b = coefficients.at<float>(0, 1) / 2;
  float c = coefficients.at<float>(0, 2);
  float d = coefficients.at<float>(0, 3) / 2;
  float e = coefficients.at<float>(0, 4) / 2;
  float f = coefficients.at<float>(0, 5);
  Mat conic = (Mat_<float>(3, 3) << a, b, d, b, c, e, d, e, f);
  // Make sure to return positive-definite conic
  return (a > 0 ? conic : -conic);
}

Mat getConic() {
  vector<Mat> points;

  cout << "Enter the 5 points on the imaged circle" << endl;
  for(int i = 0; i < 5; i++) {
    points.push_back(normalize(readHomogeneousCoordinates()));
  }

  return solveConicSystem(points);
}

Mat intersectLineAndConic(Mat conic, Mat P, Mat Q) {
  Mat Pt, Qt;

  transpose(P, Pt);
  transpose(Q, Qt);

  Mat A = Pt * conic * P;
  Mat B = 2 * Pt * conic * Q;
  Mat C = Qt * conic * Q;

  float a = A.at<float>(0, 0);
  float b = B.at<float>(0, 0);
  float c = C.at<float>(0, 0);

  float alpha = -b / (2 * a);
  float beta = sqrt(4 * a * c - b * b) / (2 * a);

  Mat circularPoint;

  Mat channels[2] = {alpha * P + Q, beta * P};
  merge(channels, 2, circularPoint);

  return circularPoint;
}

Mat transformConic(Mat pointTransform, Mat conic) {
  Mat pointTransform_t;

  transpose(pointTransform, pointTransform_t);

  return pointTransform_t.inv() * conic * pointTransform.inv();
}

Mat getConjugate(Mat point) {
  Mat conjugate;
  vector<Mat> channels(2);

  split(point, channels);
  channels = {channels[0], -channels[1]};
  merge(channels, conjugate);

  return conjugate;
}

Mat complexMatMul(Mat matrix1, Mat matrix2) {
  Mat m1, m2;

  int n = matrix1.rows;
  int m = matrix2.cols;
  int numChannels1 = matrix1.channels();
  int numChannels2 = matrix2.channels();
  assert(matrix1.cols == matrix2.rows);
  assert(numChannels1 <= 2 && numChannels2 <= 2);

  if(numChannels1 < 2) {
    Mat tmp[] = {matrix1, Mat::zeros(matrix1.size(), CV_32FC1)};
    merge(tmp, 2, m1);
    m2 = matrix2;
  }

  if(numChannels2 < 2) {
    Mat tmp[] = {matrix2, Mat::zeros(matrix2.size(), CV_32FC1)};
    merge(tmp, 2, m2);
    m1 = matrix1;
  }

  Mat product(n, m, CV_32FC2);

  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
      Mat result, row, column;
      row = m1.row(i);
      transpose(m2.col(j), column);
      mulSpectrums(row, column, result, 0);
      Scalar resultSum = sum(result);
      product.at<Vec2f>(i, j) = Vec2f(resultSum[0], resultSum[1]);
    }
  }

  return product;
}

Mat fullDecomp(Mat symmetricMat) {
  Mat W, U, Ut, W_sqrt, diagonal;
  cv::SVDecomp(symmetricMat, W, U, Ut, cv::SVD::FULL_UV);
  cv::sqrt(W, W_sqrt);
  return U * diagonal.diag(W_sqrt) * Ut;
}

Mat choleskyUpperDecomp(Mat matrix) {
  Mat K, Q, W, U, Ut, W_sqrt, diagonal;

  cv::SVDecomp(matrix, W, U, Ut, cv::SVD::FULL_UV);
  cv::sqrt(W, W_sqrt);
  Mat V = U * diagonal.diag(W_sqrt);
  // We don't have a function to perform RQ decomposition for 2x2 matrix so we
  // convert the 2x2 into a 3x3 such that the result for the 2x2 is the same as
  // it would be for the corresponding 3x3.
  Mat V_3x3 = (Mat_<float>(3, 3) << V.at<float>(0, 0), V.at<float>(0, 1),
      0, V.at<float>(1, 0), V.at<float>(1, 1), 0, 0, 0, 1);
  RQDecomp3x3(V_3x3, K, Q);

  return (Mat_<float>(2, 2) << K.at<float>(0, 0), K.at<float>(0, 1),
      K.at<float>(1, 0), K.at<float>(1, 1));
}

Mat choleskyDecomposition(Mat matrix) {
  Mat upperTriangular = matrix.clone();

  cv::Cholesky((float*)upperTriangular.ptr(), upperTriangular.step,
      upperTriangular.rows, NULL, 0, 0);

  upperTriangular = upperTriangular.t();

  for(int i = 1; i < upperTriangular.rows; i++) {
    for(int j = 0; j < i; j++) {
      upperTriangular.at<float>(i, j) = 0;
    }
  }

  Mat lowerTriangular;
  cv::transpose(upperTriangular, lowerTriangular);
  return lowerTriangular;
}

Mat normalize(Mat coordinates) {
  return coordinates / coordinates.at<float>(2, 0);
}

void printVector(vector <float> v) {
  cout << "Printing vector" << endl;
  for(std::vector<float>::const_iterator i = v.begin(); i != v.end(); ++i) {
    cout << *i << ' ';
  }
  cout << endl;
}

Mat getNullSpace(void* linearSystem, int numRows, int numCols, int finalRows) {
  Mat W, U, Vt;
  cv::SVDecomp(
      Mat(numRows, numCols, CV_32FC1, linearSystem),
      W,
      U,
      Vt,
      cv::SVD::FULL_UV);
  return Vt(Range(numRows, numRows + 1), Range::all()).reshape(0, finalRows);
}

Mat readHomogeneousCoordinates() {
  float x1, x2, x3;
  std::cin >> x1;
  std::cin >> x2;
  std::cin >> x3;
  return (Mat_<float>(3, 1) << x1, x2, x3);
}

Mat computeHomography(std::vector <Point2f> from, std::vector <Point2f> to) {
  int numPoints = from.size();

  bool isMinimal = (numPoints / 4) > 0;
  bool areNumCorrespondencesEqual = (to.size() == numPoints);
  assert(isMinimal && areNumCorrespondencesEqual);

  int numCols = 9;
  int numRows = numPoints * 2;
  Mat result;

  for(auto it_from = from.begin(), it_to = to.begin();
      it_from < from.end();
      it_from++, it_to++) {
    Mat row;

    row = (Mat_<float>(1, numCols) << -(*it_from).x, -(*it_from).y, -1,
        0, 0, 0, (*it_from).x * (*it_to).x, (*it_from).y * (*it_to).x,
        (*it_to).x);
    result.push_back(row);

    row = (Mat_<float>(1, numCols) << 0, 0, 0, -(*it_from).x, -(*it_from).y,
        -1, (*it_from).x * (*it_to).y, (*it_from).y * (*it_to).y, (*it_to).y);
    result.push_back(row);
  }

  Mat W, U, Vt;
  cv::SVDecomp(
      result,
      W,
      U,
      Vt,
      cv::SVD::FULL_UV);
  return Vt(Range(numRows, numRows + 1), Range::all()).reshape(0, 3);
}

Mat changePerspective(Mat image, Mat homography) {
  cout << "Applying homography" << endl;
  // Origin of our coordinate system is top-left of image
  int height = image.size().height;
  int width = image.size().width;

  Mat world = Mat(height, width, image.type());
  Mat mapX = Mat(height, width, CV_32FC1);
  Mat mapY = Mat(height, width, CV_32FC1);
  for(int i = 0; i < width; i++) {
    for(int j = 0; j < height; j++) {
      const float homogeneousPoint[3] = {(float)i, (float)j, 1};
      Mat imagePoint = homography * Mat(3, 1, CV_32FC1, (void*)&homogeneousPoint);
      float imageZ = imagePoint.at<float>(2, 0);
      float imageX = imagePoint.at<float>(0, 0)/imageZ;
      float imageY = imagePoint.at<float>(1, 0)/imageZ;
      if(imageX > 0 && imageY > 0 && imageX < width && imageY < height) {
        mapX.at<float>(j, i) = imageX;
        mapY.at<float>(j, i) = imageY;
      } else {
        // Setting the x and y values in the maps to a value outside the image
        // causes them to be treated as "outliers" in the image and the default
        // BORDER_CONSTANT flag maps outliers to 0
        mapX.at<float>(j, i) = width + 1;
        mapY.at<float>(j, i) = height + 1;
      }
    }
  }
  remap(image, world, mapX, mapY, INTER_LINEAR);
  return world;
}

Mat computeScaling(Mat image, Mat homography) {
  float width = image.size().width;
  float height = image.size().height;

  const float extent[3][4] = {
    {0, width, 0, width},
    {0, 0, height, height},
    {1, 1, 1, 1},
  };

  Mat worldExtent = homography.inv() * Mat(3, 4, CV_32FC1, (void*)&extent);

  float minWidth = std::numeric_limits<float>::infinity();
  float maxWidth = -std::numeric_limits<float>::infinity();
  float minHeight = std::numeric_limits<float>::infinity();
  float maxHeight = -std::numeric_limits<float>::infinity();

  for(int i = 0; i < 4; i++) {
    float z = worldExtent.at<float>(2, i);
    float x = worldExtent.at<float>(0, i) / z;
    float y = worldExtent.at<float>(1, i) / z;
    minWidth =  x < minWidth ? x : minWidth;
    maxWidth =  x > maxWidth ? x : maxWidth;
    minHeight =  y < minHeight ? y : minHeight;
    maxHeight =  y > maxHeight ? y : maxHeight;
  }

  float widthScaleFactor = (maxWidth - minWidth)/width;
  float heightScaleFactor = (maxHeight - minHeight)/height;

  float finalScaleFactor = (widthScaleFactor > heightScaleFactor ?  widthScaleFactor : heightScaleFactor);

  // Scale the smaller image up and translate to the bottom left to match with actual world extent
  return (Mat_<float>(3, 3) << finalScaleFactor, 0, minWidth, 0, finalScaleFactor, minHeight, 0, 0, 1);
}

void write(char* nameOfOriginal, Mat image, std::string suffix) {
  string path = string(nameOfOriginal);
  string filename = path.substr(path.find_last_of("/") + 1, path.length());
  string imagename = filename.substr(0, filename.find_last_of("."));
  string outputFilename = "output/" + imagename + suffix + ".jpg";
  cout << "Writing to " << outputFilename << endl;
  imwrite(outputFilename, image);
}

void display(Mat image) {
  namedWindow("Display Image", WINDOW_AUTOSIZE );
  image.convertTo(image, CV_8U);
  imshow("Display Image", image);
  waitKey(0);
}

void receivePointCorrespondence(std::vector <Point2f> &points, const std::string &type) {
  std::cout << "Enter the x,y coordinates of four points in the " << type << " image\n";
  for(int i = 0; i < 4; i++) {
    float x, y;
    std::cin >> x;
    std::cin >> y;
    points.push_back(Point2f(x, y));
  }
}
