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
 * An example use of the PTR EKF
 * 
 * @author Joel Dunham <joel.ph.dunham@gmail.com>
 * @date 2022/03/05
 * 
 */

#include <ptr_ekf.h>
#include <Arduino.h>

#define ZERO_DELTA 0.00001

// Update rates
uint32_t msSensorCheck{500};
uint64_t msSinceLastSensorCheck{0};
uint64_t msLastTime{0};

// For state estimation
KalmanFiltering::PTREKF ekf;
BLA::Matrix<KalmanFiltering::PTREKF_MAX_MEASUREMENT_SIZE, KalmanFiltering::PTREKF_MAX_MEASUREMENT_SIZE> measurementCovariance;
BLA::Matrix<KalmanFiltering::PTREKF_MAX_MEASUREMENT_SIZE, 1> measurement;  // To avoid constantly recreating this
BLA::Matrix<KalmanFiltering::PTREKF_STATE_SIZE, 1> state;  // To avoid constantly recreating this
BLA::Matrix<KalmanFiltering::PTREKF_STATE_SIZE, KalmanFiltering::PTREKF_STATE_SIZE> covariance;  // To avoid constantly recreating this

// The test cases
const uint8_t TEST_CASES{20};
float pan_deg[TEST_CASES] =
  {-170.0, -150.3, -140.2, -130.3, -120.8, -115.7, -110.0, -100.0, -84.0, -70.0,
   -45.0, -40.0, -35.0, -20.0, -10.0, 0.0, 10.4, 15.5, 22.0, 35.0};
float tilt_deg[TEST_CASES] =
  {0.0, 5.0, 7.4, 9.0, 11.2, 14.0, 20.8, 25.1, 32.7, 40.8,
   35.0, 29.3, 23.8, 21.4, 18.4, 17.2, 12.0, 10.8, 3.0, -1.0};
float range_m[TEST_CASES] =
  {12.0, 11.5, 10.7, 10.4, 9.8, 8.7, 8.1, 7.6, 7.2, 6.5,
   6.1, 5.9, 5.7, 5.3, 4.8, 4.7, 4.5, 4.1, 3.5, 3.1};
uint8_t test_counter{0};

// Function
void setup() {
    Serial.begin(115200);
    while(!Serial) {}

    // Handle the EKF debug output
    ekf.setOutputStream(&Serial);
    ekf.setOutputLevel(KalmanFiltering::OutputLevel::KF_DEBUG);

    // If the EKF requires additional setup from defaults, do so here.
    // Assuming these are static for now
    measurementCovariance.Fill(0.0);
    for(uint8_t idx = 0; idx < 3; idx++) {
        if(idx < 2) {
            measurementCovariance(idx, idx) = KalmanFiltering::PTREKF_DEFAULT_ANGLE_NOISE;
        } else {
            measurementCovariance(idx, idx) = KalmanFiltering::PTREKF_DEFAULT_RANGE_NOISE;
        }
    }
}

void targetEstimation(uint64_t msNow) {
    // Using time units of seconds
    float timeNowS = ((float)msNow) / 1000.0;

    // Create the measurement based on the test cases
    measurement(0, 0) = pan_deg[test_counter] * PI / 180.0;
    measurement(1, 0) = tilt_deg[test_counter] * PI / 180.0;
    measurement(2, 0) = range_m[test_counter];

    // Initialize at the beginning of each set of test cases
    if(test_counter == 0) {
        // Initialize the EKF
        BLA::Matrix<KalmanFiltering::PTREKF_STATE_SIZE, 1> init_state;
        BLA::Matrix<KalmanFiltering::PTREKF_STATE_SIZE, KalmanFiltering::PTREKF_STATE_SIZE> init_cov;
        init_cov.Fill(0.0);
        for(uint8_t idx = 0; idx < 3; idx++) {
            init_state(idx, 0) = measurement(idx, 0);
            init_state(idx + 3, 0) = 0.0;  // Assume no movement to start
            // Initialize noise with an extra factor of MULTIPLIER to enable a first measurement to be wrong
            if(idx < 2) {
                init_cov(idx, idx) = KalmanFiltering::PTREKF_DEFAULT_ANGLE_NOISE *
                KalmanFiltering::PTREKF_DEFAULT_RATE_NOISE_MULTIPLIER;
                init_cov(idx + 3, idx + 3) = KalmanFiltering::PTREKF_DEFAULT_ANGLE_NOISE *
                KalmanFiltering::PTREKF_DEFAULT_RATE_NOISE_MULTIPLIER *
                KalmanFiltering::PTREKF_DEFAULT_RATE_NOISE_MULTIPLIER;
            } else {
                init_cov(idx, idx) = KalmanFiltering::PTREKF_DEFAULT_RANGE_NOISE *
                KalmanFiltering::PTREKF_DEFAULT_RATE_NOISE_MULTIPLIER;
                init_cov(idx + 3, idx + 3) = KalmanFiltering::PTREKF_DEFAULT_RANGE_NOISE *
                KalmanFiltering::PTREKF_DEFAULT_RATE_NOISE_MULTIPLIER *
                KalmanFiltering::PTREKF_DEFAULT_RATE_NOISE_MULTIPLIER;
            }
        }
        ekf.initialize(timeNowS, init_state, init_cov);  // Initialize in seconds
    } else {
        // Update the EKF. First, predict to current time
        ekf.predict(timeNowS);

        // Add the correction
        ekf.ptrCorrect(timeNowS, measurement, measurementCovariance);
    }

    // Index the test counter
    if(++test_counter >= TEST_CASES) {
        test_counter = 0;
    }
}
 
void loop() {
    // Update all the since checks
    uint64_t msNow = millis();
    uint64_t msDeltaT = (msNow > msLastTime) ? msNow - msLastTime : 0;  // Handle rollover
    msSinceLastSensorCheck = msSinceLastSensorCheck + msDeltaT;

    if(msSinceLastSensorCheck >= msSensorCheck) {
        // Update the estimation
        // Everything handled internally in the function
        targetEstimation(msNow);

        msSinceLastSensorCheck = 0;  // Reset for next check
    }

    // For the next round
    msLastTime = msNow;
}
