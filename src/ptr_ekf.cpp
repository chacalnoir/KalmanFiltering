/**
 * MIT License
 *
 * Copyright (c) 2022 Joel Dunham
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 * An EKF implementation for tracking something with a pan/tilt/ranging system.
 * 
 * @author Joel Dunham <joel.ph.dunham@gmail.com>
 * @date 2022/03/05
 * 
 */

#include <ptr_ekf.h>

#define PTREKF_DEFAULT_ANGLE_NOISE 0.001
#define PTREKF_DEFAULT_RANGE_NOISE 0.0001
#define PTREKF_DEFAULT_RATE_NOISE_MULTIPLIER 10.0

namespace KalmanFiltering {

PTREKF::PTREKF() {
    // Set appropriate information to the base class
    // Set up state information
    BLA::Matrix<PTREKF_STATE_SIZE, PTREKF_STATE_SIZE> q;
    q.Fill(0.0);
    for(uint8_t counter = 0; counter < 3; counter++) {
        if(counter < 2) {
            // Angle and angular rate
            q(counter, counter) = PTREKF_DEFAULT_ANGLE_NOISE;
            q(counter + 3, counter + 3) = PTREKF_DEFAULT_ANGLE_NOISE * PTREKF_DEFAULT_RATE_NOISE_MULTIPLIER;
        } else {
            // Range and range rate
            q(counter, counter) = PTREKF_DEFAULT_RANGE_NOISE;
            q(counter + 3, counter + 3) = PTREKF_DEFAULT_RANGE_NOISE * PTREKF_DEFAULT_RATE_NOISE_MULTIPLIER;
        }
    }
    setStateTransition(&stateTransition, &stateJacobian, q);
    setHasRanges(true);

    // Set up measurement information
    ptr_measurement_id_ = defineMeasurement(PTREKF_MAX_MEASUREMENT_SIZE, &measurement, &measurementJacobian);
    // Define the range and wrapping for the angle data in the measurements.
    setMeasurementRange(ptr_measurement_id_, 0, true, true, -1.0 * PI, 1.0 * PI);
    setMeasurementRange(ptr_measurement_id_, 1, true, true, -1.0 * PI, 1.0 * PI);

    // Define the range and wrapping for the angle data in the state.
    setStateRange(0, true, true, -1.0 * PI, 1.0 * PI);
    setStateRange(1, true, true, -1.0 * PI, 1.0 * PI);
}

bool PTREKF::ptrCorrect(float time, BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, 1> z,
                        BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, PTREKF_MAX_MEASUREMENT_SIZE> r) {
    return EKF<PTREKF_STATE_SIZE, PTREKF_MAX_MEASUREMENT_SIZE, PTREKF_INPUT_SIZE>::correct(ptr_measurement_id_, time, z, r);
}

BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, 1> PTREKF::measurement(BLA::Matrix<PTREKF_STATE_SIZE, 1>x) {
    BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, 1> y;
    for(uint8_t counter = 0; counter < 3; counter++) {
        // The measurement is the first 3 states
        y(counter, 0) = x(counter, 0);
    }

    return y;
}

BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, PTREKF_STATE_SIZE> PTREKF::measurementJacobian(BLA::Matrix<PTREKF_STATE_SIZE, 1>x) {
    BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, PTREKF_STATE_SIZE> jacobian;
    /*
     *  1 0 0 0 0 0
     *  0 1 0 0 0 0
     *  0 0 1 0 0 0
     */
    jacobian.Fill(0.0);
    jacobian(0, 0) = 1.0;
    jacobian(1, 1) = 1.0;
    jacobian(2, 2) = 1.0;

    return jacobian;
}

BLA::Matrix<PTREKF_STATE_SIZE, 1> PTREKF::stateTransition(float deltaT, BLA::Matrix<PTREKF_STATE_SIZE, 1>x, BLA::Matrix<PTREKF_INPUT_SIZE, 1>u) {
    // Assuming constant velocity, with noise.
    BLA::Matrix<PTREKF_STATE_SIZE, 1> x_new;
    for(uint8_t counter = 0; counter < 3; counter++) {
        x_new(counter, 0) = x(counter, 0) + x(counter + 3, 0) * deltaT;
        x_new(counter + 3, 0) = x(counter + 3, 0);
    }

    return x_new;
}

BLA::Matrix<PTREKF_STATE_SIZE, PTREKF_STATE_SIZE> PTREKF::stateJacobian(float deltaT, BLA::Matrix<PTREKF_STATE_SIZE, 1>x, BLA::Matrix<PTREKF_INPUT_SIZE, 1>u) {
    BLA::Matrix<PTREKF_STATE_SIZE, PTREKF_STATE_SIZE> jacobian;
    /*
     *  1 0 0   0      0       0
     *  0 1 0   0      0       0
     *  0 0 1   0      0       0
     *  0 0 0 deltaT   0       0
     *  0 0 0   0    deltaT    0
     *  0 0 0   0      0     deltaT
     */
    jacobian.Fill(0.0);
    for(uint8_t counter = 0; counter < 3; counter++) {
        jacobian(counter, counter) = 1.0;
        jacobian(counter + 3, counter + 3) = deltaT;
    }

    return jacobian;
}



}  // namespace KalmanFiltering