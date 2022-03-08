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

#ifndef KALMAN_FILTERING_PTR_EKF_H
#define KALMAN_FILTERING_PTR_EKF_H

#include <ekf.h>

namespace KalmanFiltering {

static const uint8_t PTREKF_STATE_SIZE = 6;
static const uint8_t PTREKF_MAX_MEASUREMENT_SIZE = 3;
static const uint8_t PTREKF_INPUT_SIZE = 0;

/**
 * An EKF specifically for a pan/tilt/range target relative to a sensing platform.
 * Pan and tilt are in radians. Range is in meters.
 * Time is in seconds.
 *  State size 6, max measurement size 3, input size, 0
 */
class PTREKF: public EKF<PTREKF_STATE_SIZE, PTREKF_MAX_MEASUREMENT_SIZE, PTREKF_INPUT_SIZE> {
    public:
        PTREKF();
        virtual ~PTREKF() {};

        /**
         * For providing a PTR (pan/tilt/range) measurement to the EKF. Avoids the external system needing to know the
         *  measurement ID.
         * @param time: float time in seconds of the measurement
         * @param z: BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, 1> measurement of pan/tilt/range in rad/rad/m
         * @param r: BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, PTREKF_MAX_MEASUREMENT_SIZE> measurement covariance for pan/tilt/range
         */
        bool ptrCorrect(float time, BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, 1> z,
                        BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, PTREKF_MAX_MEASUREMENT_SIZE> r);

        /**
         * The function for predicting the PTR measurement from the state
         */
        static BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, 1> measurement(BLA::Matrix<PTREKF_STATE_SIZE, 1>x);

        /**
         * The function for calculating the measurement Jacobian. Is a time invariant matrix.
         */
        static BLA::Matrix<PTREKF_MAX_MEASUREMENT_SIZE, PTREKF_STATE_SIZE> measurementJacobian(BLA::Matrix<PTREKF_STATE_SIZE, 1>x);

        /**
         * The function for calculating the new state from the current state and input/control function.
         */
        static BLA::Matrix<PTREKF_STATE_SIZE, 1> stateTransition(float deltaT, BLA::Matrix<PTREKF_STATE_SIZE, 1>x, BLA::Matrix<PTREKF_INPUT_SIZE, 1>u);

        /**
         * The function for calculating the Jacobian of the state transition function. Is a time invariant matrix.
         */
        static BLA::Matrix<PTREKF_STATE_SIZE, PTREKF_STATE_SIZE> stateJacobian(float deltaT, BLA::Matrix<PTREKF_STATE_SIZE, 1>x, BLA::Matrix<PTREKF_INPUT_SIZE, 1>u);
    private:
        // The ID for all full PTR measurements
        uint8_t ptr_measurement_id_{0};
};

}  // namespace KalmanFiltering

#endif  // KALMAN_FILTERING_PTR_EKF_H