1. To get a line through cross product of points, the selected points must be
   as far apart as possible. If points are close then small pixel errors in
selection will cause wide variation of line that is determined by the points.
2. If the homogeneous coordinates are very large, then normalize them by
   dividing by the third coordinate. I think the numbers are too large to fit
into "float" type.
3. To decompose KKt we can use three methods 
   - Cholesky factorization(upper triangular), 
   - Cholesky decomposition(lower triangular),
   - LDLt.
They will give different, but correct, results. Cholesky decomposition works
but would imply that K is lower triangular. Maybe that is also possible for an
SAP or PAS decomposition of a homography?
4. For the methods to work, the points/lines selected must result in a
   positive definite matrix.
5. The performance of the one step method is very sensitive to the selection of
   points/lines. Only a good selection results in an accurate result. As the
number of lines that need to be selected is relatively large, it's easy to end up with a
non positive definite matrix which the algorithm does not work with.

Errata in book
--------------
Page 43, equation 2.17
P must have $v$ as the element in the third row, third column instead of 1. It
doesn't really matter but either that or the text below it should change.
Page 57, example 2.27
C\_inf^\* must read C'\_inf^\*. The text refers to the *image* of the dual
conic, not the dual conic itself.
