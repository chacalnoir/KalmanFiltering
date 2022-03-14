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
 * An EKF implementation.
 * 
 * @author Joel Dunham <joel.ph.dunham@gmail.com>
 * @date 2022/03/05
 * 
 */

#ifndef KALMAN_FILTERING_EKF_H
#define KALMAN_FILTERING_EKF_H

#include <BasicLinearAlgebra.h>
#include <vector>

#define KF_ZERO_DELTA 0.0000001

namespace KalmanFiltering {

enum class OutputLevel {
    KF_NONE = 0,
    KF_ERROR = 1,
    KF_WARN = 2,
    KF_INFO = 3,
    KF_DEBUG = 4,
    KF_VERBOSE = 5
};

/**
 * Handles values that can only be in a specified range. Includes limiting and wrapping.
 */
class RangedValue {
    public:
        RangedValue(bool is_ranged = false, bool wrap = false, float lower_limit = 0.0, float upper_limit = 0.0):
            is_ranged_(is_ranged),
            wrap_(wrap),
            lower_limit_(lower_limit),
            upper_limit_(upper_limit) {};
        
        void setRange(bool is_ranged = false, bool wrap = false, float lower_limit = 0.0, float upper_limit = 0.0) {
            is_ranged_ = is_ranged;
            wrap_ = wrap;
            lower_limit_ = lower_limit;
            upper_limit_ = upper_limit;
        }
        
        float updateValue(float value) {
            if(!is_ranged_) {
                return value;
            } else if(!wrap_) {
                // Limit only
                return max(lower_limit_, min(upper_limit_, value));
            } else {
                // Wrap
                float range = upper_limit_ - lower_limit_;
                while(value < lower_limit_) {
                    value += range;
                }
                while(value > upper_limit_) {
                    value -= range;
                }
                return value;
            }
        }
    
    private:
        bool is_ranged_{false};
        bool wrap_{false};
        float lower_limit_{0.0};
        float upper_limit_{0.0};
};

template<uint8_t state_size, uint8_t max_measurement_size>
class MeasurementApplication {
    public:
        typedef BLA::Matrix<max_measurement_size, 1> (*measurementFuncPtr)(BLA::Matrix<state_size, 1>);
        typedef BLA::Matrix<max_measurement_size, state_size> (*hFuncPtr)(BLA::Matrix<state_size, 1>);
        MeasurementApplication(uint8_t measurement_size,
                               BLA::Matrix<max_measurement_size, 1> (*measurementFuncPtr)(BLA::Matrix<state_size, 1>x),
                               BLA::Matrix<max_measurement_size, state_size> (*hFuncPtr)(BLA::Matrix<state_size, 1>x)):
            measurement_size_(measurement_size),
            measurementFuncPtr_(measurementFuncPtr),
            hFuncPtr_(hFuncPtr) {};
        
        // Accessors. Cannot change once set, so no setters.
        uint8_t getMeasurementSize() { return measurement_size_; }
        measurementFuncPtr getMeasurementFunctionPtr() {
            return measurementFuncPtr_;
        }
        hFuncPtr getHFunctionPtr() {
            return hFuncPtr_;
        }

        /**
         * Setter for ranged values
         * @param index uint8_t the index of the value to set the range for
         * @param has_range bool whether the range is specified
         * @param wrap bool whether to wrap values into the range
         * @param lower_limit float the range lower limit
         * @param upper_limit float the range upper limit
         * @return bool whether it was set
         */
        bool setRange(uint8_t index, bool has_range=false, bool wrap=false, float lower_limit = 0.0, float upper_limit = 0.0) {
            if(index >= max_measurement_size) {
                return false;
            }
            ranges_[index].setRange(has_range, wrap, lower_limit, upper_limit);
            return true;
        }

        float rangeValue(uint8_t index, float value) {
            if(index >= max_measurement_size) {
                // TODO: Debug output
                return value;
            } else {
                return ranges_[index].updateValue(value);
            }
        }

    private:
        // The actual size of the measurement since the stored matrix has to be larger
        //  due to templated arguments.
        uint8_t measurement_size_{1};

        /**
         * Function pointer for calculating the predicted measurement.
         *  Could be a linear or non-linear function
         */
        BLA::Matrix<max_measurement_size, 1> (*measurementFuncPtr_)(BLA::Matrix<state_size, 1>x){nullptr};

        /**
         * Function pointer for calculating the H matrix.
         *  Could be static or a linearization of the prediction measurement function.
         */
        BLA::Matrix<max_measurement_size, state_size> (*hFuncPtr_)(BLA::Matrix<state_size, 1>x){nullptr};

        /**
         * For handling ranged values
         */
        RangedValue ranges_[max_measurement_size];
};

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
class EKF {
    public:

        // Constructor
        EKF();

        // Virtual destructor to enable derived classes
        virtual ~EKF();

        /**
         * Set the output level. Can only output if the stream is set via setOutputStream(Stream *out)
         */
        void setOutputLevel(OutputLevel level) {
            output_level_ = level;
        }

        /**
         * Sets the output stream. Will only output if the information is at or above
         *  the output level set by setOutputLevel(OutputLevel level)
         */
        void setOutputStream(Stream *out) {
            ptrOut_ = out;
        }

        /**
         * Set whether to run range checks on state and/or measurement updates
         */
        void setHasRanges(bool has_ranges=false) {
            has_ranges_ = has_ranges;
        }

        /**
         * Set the initial state and convariance
         * @param time: float time of initialization
         * @param x: BLA::Matrix<state_size, 1> initial state
         * @param p: BLA::Matrix<state_size, state_size> initial covariance
         */
        void initialize(float time, BLA::Matrix<state_size, 1> x, BLA::Matrix<state_size, state_size> p);

        /**
         * Reset to the initial state and covariance.
         * @param time: float time of initialization
         * @return whether this call worked since the initial values had to be set previously.
         */
        bool initialize(float time);

        /**
         * Sets state transition functions and the default process noise.
         * @param fFuncPtr: function pointer for f (the state transition matrix with input/control u)
         * @param FfuncPtr: function pointer for the state transition Jacobian. May return a constant matrix if the system is LTI.
         * @param q: BLA::Matrix<state_size, state_size> process noise for the prediction update
         */
        void setStateTransition(BLA::Matrix<state_size, 1> (*transitionFuncPtr)(float deltaT, BLA::Matrix<state_size, 1>x, BLA::Matrix<input_size, 1>u),
                                BLA::Matrix<state_size, state_size> (*jacobianFuncPtr)(float deltaT, BLA::Matrix<state_size, 1>x, BLA::Matrix<input_size, 1>u),
                                BLA::Matrix<state_size, state_size> q) {
            transitionFuncPtr_ = transitionFuncPtr;
            jacobianFuncPtr_ = jacobianFuncPtr;
            q_ = q;
        }

        /**
         * Set a process noise matrix. Note that this can be updated at any time for a dynamic
         *  process noise.
         * @param q: BLA::Matrix<state_size, state_size> process noise for the prediction update
         */
        void setProcessNoise(BLA::Matrix<state_size, state_size> q) {
            q_ = q;
        }

        /**
         * Defines a measurement input and returns an ID for later providing such measurements.
         * @param measurement_size: size of the measurement vector
         * @param measurementFuncPtr: function pointer for calculating the measurement from the state
         * @param hFuncPtr: function pointer for calculating the H matrix from the state (or returning a constant matrix)
         * @return uint8_t measurement type ID - ID to be used when providing measurements.
         */
        uint8_t defineMeasurement(uint8_t measurement_size,
                                  BLA::Matrix<max_measurement_size, 1> (*measurementFuncPtr)(BLA::Matrix<state_size, 1>x),
                                  BLA::Matrix<max_measurement_size, state_size> (*hFuncPtr)(BLA::Matrix<state_size, 1>x));

        /**
         * Defines a measurement range for a specified measurement ID and value index
         * @param measurement_id uint8_t ID of the measurement definition
         * @param index uint8_t the index of the value to set the range for
         * @param has_range bool whether the range is specified
         * @param wrap bool whether to wrap values into the range
         * @param lower_limit float the range lower limit
         * @param upper_limit float the range upper limit
         * @return bool whether it was set
         */
        bool setMeasurementRange(uint8_t measurement_id, uint8_t index, bool has_range=false, bool wrap=false,
                                 float lower_limit = 0.0, float upper_limit = 0.0);
        
        /**
         * Defines a state range for a specified state value index
         * @param index uint8_t the index of the value to set the range for
         * @param has_range bool whether the range is specified
         * @param wrap bool whether to wrap values into the range
         * @param lower_limit float the range lower limit
         * @param upper_limit float the range upper limit
         * @return bool whether it was set
         */
        bool setStateRange(uint8_t index, bool has_range=false, bool wrap=false,
                           float lower_limit = 0.0, float upper_limit = 0.0);

        /**
         * Prediction step. Convenience function when no control/input vector in use.
         * @param time: float time in EKF time units.
         */
        void predict(float time);

        /**
         * Prediction step if using a control/input matrix
         * @param time: float time in EKF time units
         * @param u: BLA::Matrix<input_size, 1> input/control values
         */
        void predict(float time, BLA::Matrix<input_size, 1> u);

        /**
         * Correction step with measurement input. Can be called multiple times with different measurement types
         *  at a single timestep. Just do not move backwards in time.
         * @param measurement_id: uint8_t ID from a previously defined measurement type.
         * @param time: float time of measurement in EKF time units. Cannot go backwards, so will assume current or in the future.
         *              If in the future, runs a prediction step first.
         * @param z: BLA::Matrix<max_measurement_size, 1> measurement. Only fill up to the number of values in the measurement.
         * @param r: BLA::Matrix<max_measurement_size, max_measurement_size> measurement noise. Only fill up to the number of values in
         *           the measurement.
         * @return bool whether the correct step succeeded.
         */
        bool correct(uint8_t measurement_id, float time, 
                     BLA::Matrix<max_measurement_size, 1> z,
                     BLA::Matrix<max_measurement_size, max_measurement_size> r);

        /**
         * Gets the current state
         * @param state: the output variable to receive the state data
         */
        void getState(BLA::Matrix<state_size, 1> &state);

        /**
         * Gets the current covariance
         * @param covariance: the output variable to receive the covariance data
         */
        void getCovariance(BLA::Matrix<state_size, state_size> &covariance);

    protected:
        // Debugging
        OutputLevel output_level_{OutputLevel::KF_NONE};
        Stream* ptrOut_{nullptr};

    private:
        // The last update time
        float last_time_{0.0};

        // The state, sized at compilation
        BLA::Matrix<state_size, 1> x_;

        // Handling wrapped values in the state after updates
        RangedValue ranges_[state_size];
        // Quick way to bypass all range checks if no ranges are specified
        bool has_ranges_{false};

        // The covariance, sized at compilation
        BLA::Matrix<state_size, state_size> p_;

        // The process noise, sized at compilation
        BLA::Matrix<state_size, state_size> q_;

        // State transition function
        BLA::Matrix<state_size, 1> (*transitionFuncPtr_)(float deltaT, BLA::Matrix<state_size, 1>x, BLA::Matrix<input_size, 1>u);

        // State Jacobian function
        BLA::Matrix<state_size, state_size> (*jacobianFuncPtr_)(float deltaT, BLA::Matrix<state_size, 1>x, BLA::Matrix<input_size, 1>u);

        // For predictions with the convenience function
        BLA::Matrix<input_size, 1> dummy_u_;

        // Initialization values
        bool is_initialized_{false};
        BLA::Matrix<state_size, 1> initial_x_;
        BLA::Matrix<state_size, state_size> initial_p_;

        // For measurements
        std::vector<MeasurementApplication<state_size, max_measurement_size>> measurement_data_;
        BLA::Matrix<max_measurement_size, 1> y_;  // Prediction measurement (to avoid reallocating this every correct() call)
        // Measurement - prediction measurement (to avoid reallocating this every correct() call)
        BLA::Matrix<max_measurement_size, 1> residual_;
        // Residual covariance (to avoid reallocating this every correct() call)
        BLA::Matrix<max_measurement_size, max_measurement_size> s_;
        // Measurement Jacobian (to avoid reallocating this every correct() call)
        BLA::Matrix<max_measurement_size, state_size> h_;
        // Kalman gain (to avoid reallocating this every correct() call)
        BLA::Matrix<state_size, max_measurement_size> k_;
        // Identity matrix of state size (to avoid reallocating this every correct() call)
        BLA::Matrix<state_size, state_size> identity_;
};

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
EKF<state_size, max_measurement_size, input_size>::EKF() {
    // These should all be set prior to use.
    //  Initialize to zeros so no undefined behavior
    x_.Fill(0.0);
    p_.Fill(0.0);
    q_.Fill(0.0);
    dummy_u_.Fill(0.0);
    // Create the identity matrix for measurement updates
    identity_.Fill(0.0);
    for(uint8_t counter = 0; counter < state_size; counter++) {
        identity_(counter, counter) = 1.0;
    }
}

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
EKF<state_size, max_measurement_size, input_size>::~EKF() {
    measurement_data_.clear();  // Enable the memory to be reclaimed.
}

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
void EKF<state_size, max_measurement_size, input_size>::initialize(float time,
                                                                   BLA::Matrix<state_size, 1> x,
                                                                   BLA::Matrix<state_size, state_size> p) {
    is_initialized_ = true;
    // Save the initialization values for re-initialization later
    initial_x_ = x;
    initial_p_ = p;
    
    initialize(time);
}

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
bool EKF<state_size, max_measurement_size, input_size>::initialize(float time) {
    if(!is_initialized_) {
        // No data with which to initialize using this function
        return false;
    }
    // Reset the state and covariance to the initial values
    x_ = initial_x_;
    p_ = initial_p_;
    // Save the time of initialization
    last_time_ = time;

    // Debug output
    if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
        ptrOut_->printf("Initial state: [");
        for(uint8_t index = 0; index < state_size; index++) {
            ptrOut_->printf("%.6f%s ", x_(index, 0),
                (index < (state_size - 1)) ? ", " : "]\n");
        }
        ptrOut_->print("Initial covariance: \n");
        for(uint8_t row = 0; row < state_size; row++) {
            ptrOut_->print("[");
            for(uint8_t col = 0; col < state_size; col++) {
                ptrOut_->printf("%.6f", p_(row, col));
                if(col < (state_size - 1)) {
                    ptrOut_->print(", ");
                }
            }
            ptrOut_->print("]\n");
        }
        ptrOut_->println();
    }

    // Has been re-initialized
    return true;
}

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
uint8_t EKF<state_size, max_measurement_size, input_size>::defineMeasurement(uint8_t measurement_size,
        BLA::Matrix<max_measurement_size, 1> (*measurementFuncPtr)(BLA::Matrix<state_size, 1>x),
        BLA::Matrix<max_measurement_size, state_size> (*hFuncPtr)(BLA::Matrix<state_size, 1>x)) {
    // Add the new data
    measurement_data_.push_back(MeasurementApplication<state_size, max_measurement_size>(measurement_size, measurementFuncPtr, hFuncPtr));
    return measurement_data_.size() - 1;  // The index of the measurement data is the ID
}

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
bool EKF<state_size, max_measurement_size, input_size>::setMeasurementRange(uint8_t measurement_id,
        uint8_t index, bool has_range, bool wrap,
        float lower_limit, float upper_limit) {
    if(measurement_id >= measurement_data_.size()) {
        // Invalid index
        if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_WARN)) {
            ptrOut_->printf("EKF::setMeasurementRange invalid measurement id %d\n", measurement_id);
        }
        return false;
    }
    bool valid = measurement_data_[measurement_id].setRange(index, has_range, wrap, lower_limit, upper_limit);
    if(!valid && (ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_WARN)) {
        ptrOut_->printf("EKF::setMeasurementRange invalid index %d for measurement id %d\n", index, measurement_id);
    }
    return valid;
}

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
bool EKF<state_size, max_measurement_size, input_size>::setStateRange(uint8_t index, bool has_range, bool wrap,
        float lower_limit, float upper_limit) {
    if(index >= state_size) {
        // Invalid index
        if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_WARN)) {
            ptrOut_->printf("EKF::setStateRange invalid index %d\n", index);
        }
        return false;
    }
    ranges_[index].setRange(has_range, wrap, lower_limit, upper_limit);
    return true;
}

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
void EKF<state_size, max_measurement_size, input_size>::predict(float time) {
    // Call through with the dummy input vector
    predict(time, dummy_u_);
}

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
void EKF<state_size, max_measurement_size, input_size>::predict(float time, BLA::Matrix<input_size, 1> u) {
    // Run the prediction loop. This is a first order prediction loop.
    // Calculate change in time
    float deltaT = time - last_time_;

    // Debug output
    if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
        ptrOut_->printf("Predict for delta time %.6f\n", deltaT);
    }

    // Calculate the predicted state based on the state transition function
    x_ = transitionFuncPtr_(deltaT, x_, u);

    // Debug output
    if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
        ptrOut_->printf("Predicted state: [");
        for(uint8_t index = 0; index < state_size; index++) {
            ptrOut_->printf("%.6f%s ", x_(index, 0),
                (index < (state_size - 1)) ? ", " : "]\n");
        }
    }
    // Handle range limits
    if(has_ranges_) {
        for(uint8_t state_idx = 0; state_idx < state_size; state_idx++) {
            x_(state_idx, 0) = ranges_[state_idx].updateValue(x_(state_idx, 0));
        }

        // Debug output
        if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
            ptrOut_->printf("Ranged predicted state: [");
            for(uint8_t index = 0; index < state_size; index++) {
                ptrOut_->printf("%.6f%s ", x_(index, 0),
                    (index < (state_size - 1)) ? ", " : "]\n");
            }
        }
    }
    // Calculate the Jacobian at the current time
    BLA::Matrix<state_size, state_size> Fk = jacobianFuncPtr_(deltaT, x_, u);
    // Calculate the predicted covariance
    p_ = Fk*(p_*(~Fk)) + q_;

    // Debug output
    if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
        ptrOut_->print("Predicted covariance: \n");
        for(uint8_t row = 0; row < state_size; row++) {
            ptrOut_->print("[");
            for(uint8_t col = 0; col < state_size; col++) {
                ptrOut_->printf("%.6f", p_(row, col));
                if(col < (state_size - 1)) {
                    ptrOut_->print(", ");
                }
            }
            ptrOut_->print("]\n");
        }
        ptrOut_->println();
    }

    // Store the last time
    last_time_ = time;

    // Debug output
    if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
        ptrOut_->printf("End Predict\n\n");
    }
}

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
bool EKF<state_size, max_measurement_size, input_size>::correct(uint8_t measurement_id, float time,
                  BLA::Matrix<max_measurement_size, 1> z,
                  BLA::Matrix<max_measurement_size, max_measurement_size> r) {
    // Need to know the measurement size for this measurement
    uint8_t measurement_size = measurement_data_[measurement_id].getMeasurementSize();

    // Debug output
    if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
        ptrOut_->printf("Correct\n");
        ptrOut_->print("Measurement: [");
        for(uint8_t counter = 0; counter < measurement_size - 1; counter++) {
            ptrOut_->printf("%.6f", z(counter, 0));
            ptrOut_->print(", ");
        }
        ptrOut_->print(z(measurement_size - 1, 0), 6);
        ptrOut_->print("]\n");
        ptrOut_->print("Measurement Noise: \n");
        for(uint8_t row = 0; row < measurement_size; row++) {
            ptrOut_->print("[");
            for(uint8_t col = 0; col < measurement_size; col++) {
                ptrOut_->printf("%.6f", r(row, col));
                if(col < (measurement_size - 1)) {
                    ptrOut_->print(", ");
                }
            }
            ptrOut_->print("]\n");
        }
        ptrOut_->println();
    }
    
    if(time < last_time_) {
        // Cannot incorporate measurements that are backwards in time
        if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_WARN)) {
            ptrOut_->printf("EKF::correct historical measurement at time %10.6f cannot be used at time %10.6f\n", time,
                last_time_);
            ptrOut_->println("End correct\n");
        }
        return false;
    }

    // Can proceed
    if((time - last_time_) > KF_ZERO_DELTA) {
        // Predict first
        predict(time);  // Assume no control/input vector if predicting here.
    }

    // Calculate the predicted measurement and residual
    y_ = measurement_data_[measurement_id].getMeasurementFunctionPtr()(x_);
    residual_ = z - y_;

    // Debug output
    if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
        ptrOut_->printf("Residuals: [");
        for(uint8_t measurement_index = 0; measurement_index < measurement_size; measurement_index++) {
            ptrOut_->printf("%.6f%s ", residual_(measurement_index, 0),
                (measurement_index < (measurement_size - 1)) ? ", " : "]\n");
        }
    }
    // Handle ranges
    if(has_ranges_) {
        for(uint8_t measurement_index = 0; measurement_index < measurement_size; measurement_index++) {
            residual_(measurement_index, 0) =
                measurement_data_[measurement_id].rangeValue(measurement_index, residual_(measurement_index, 0));
        }

        // Debug output
        if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
            ptrOut_->printf("Ranged residuals: [");
            for(uint8_t measurement_index = 0; measurement_index < measurement_size; measurement_index++) {
                ptrOut_->printf("%.6f%s ", residual_(measurement_index, 0),
                    (measurement_index < (measurement_size - 1)) ? ", " : "]\n");
            }
        }
    }
    // Calculate the measurement Jacobian
    h_ = measurement_data_[measurement_id].getHFunctionPtr()(x_);
    // Calculate the residual covariance
    s_ = h_*(p_*(~h_)) + r;

    // Debug output
    if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_VERBOSE)) {
        ptrOut_->print("Residual covariance: \n");
        for(uint8_t row = 0; row < state_size; row++) {
            ptrOut_->print("[");
            for(uint8_t col = 0; col < measurement_size; col++) {
                ptrOut_->printf("%.6f", s_(row, col));
                if(col < (measurement_size - 1)) {
                    ptrOut_->print(", ");
                }
            }
            ptrOut_->print("]\n");
        }
        ptrOut_->println();
    }
    // Calculate the Kalman gain
    bool succeeded;
    // Only invert the part of the matrix that has useful information.
    // The rest remains as zeros.
    // Since everything is zeros for the rest, the Kalman gain will be zero, the residual will be zero, and the rest will
    //  not affect the overall update.
    k_ = p_*(~h_)*BLA::Inverse(s_, succeeded, measurement_size);
    if(!succeeded) {
        // Singular matrix
        if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_WARN)) {
            ptrOut_->printf("EKF::correct singular matrix - cannot use measurement\n");
        }
        ptrOut_->println("End correct\n");
        return false;
    } else {

        // Debug output
        if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_VERBOSE)) {
            ptrOut_->print("Kalman gain: \n");
            for(uint8_t row = 0; row < state_size; row++) {
                ptrOut_->print("[");
                for(uint8_t col = 0; col < measurement_size; col++) {
                    ptrOut_->printf("%.6f", k_(row, col));
                    if(col < (measurement_size - 1)) {
                        ptrOut_->print(", ");
                    }
                }
                ptrOut_->print("]\n");
            }
            ptrOut_->println();
        }
    }

    // Update the state
    x_ = x_ + k_ * residual_;

    // Debug output
    if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
        ptrOut_->printf("Corrected state: [");
        for(uint8_t index = 0; index < state_size; index++) {
            ptrOut_->printf("%.6f%s ", x_(index, 0),
                (index < (state_size - 1)) ? ", " : "]\n");
        }
    }
    // Handle range limits
    if(has_ranges_) {
        for(uint8_t state_idx = 0; state_idx < state_size; state_idx++) {
            x_(state_idx, 0) = ranges_[state_idx].updateValue(x_(state_idx, 0));
        }

        // Debug output
        if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
            ptrOut_->printf("Ranged corrected state: [");
            for(uint8_t index = 0; index < state_size; index++) {
                ptrOut_->printf("%.6f%s ", x_(index, 0),
                    (index < (state_size - 1)) ? ", " : "]\n");
            }
        }
    }
    // Update the covariance
    p_ = (identity_ - k_*h_)*p_;

    // Debug output
    if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
        ptrOut_->print("Covariance: \n");
        for(uint8_t row = 0; row < state_size; row++) {
            ptrOut_->print("[");
            for(uint8_t col = 0; col < state_size; col++) {
                ptrOut_->printf("%.6f", p_(row, col));
                if(col < (state_size - 1)) {
                    ptrOut_->print(", ");
                }
            }
            ptrOut_->print("]\n");
        }
        ptrOut_->println();
    }

    // Debug output
    if((ptrOut_ != nullptr) && (output_level_ >= OutputLevel::KF_DEBUG)) {
        ptrOut_->printf("End correct\n\n");
    }

    return true;
}

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
void EKF<state_size, max_measurement_size, input_size>::getState(BLA::Matrix<state_size, 1> &state) {
    state = x_;  // Copy and return
}

template<uint8_t state_size, uint8_t max_measurement_size, uint8_t input_size>
void EKF<state_size, max_measurement_size, input_size>::getCovariance(BLA::Matrix<state_size, state_size> &covariance) {
    covariance = p_;  // Copy and return
}

}  // namespace KalmanFiltering

#endif  // KALMAN_FILTERING_EKF_H