#KalmanFiltering
Kalman Filtering implementations designed for running on MCUs. Currently tested on an ESP32. Should work on most architectures
 using the Arduino Toolchain.

Includes two Kalman Filters:
1) A templated Extended Kalman Filter (EKF) that enables easy extensions for use with nonlinear functions.
    Also includes automatic variable ranges if configured (such as when state variables are angles).
2) An example Pan/Tilt/Range EKF implementation with a single measurement of pan/tilt/range in rad/rad/m.

The example code shows how to use the Pan/Tilt/Range EKF. The numbers are not realistic, so the EKF does not handle
 this example too well. This example mainly demonstrates and tests the math functions.

Debug output is easily included by setting the appropriate debug level and passing a stream object to the EKF. The example
 shows how.

This library requires the forked BasicLinearAlgebra (https://github.com/chacalnoir/BasicLinearAlgebra.git)
 in order to perform submatrix inversions (in order to handle measurements of various sizes for a single EKF).