#pragma once

#include <cassert>
#include <utility>
#include <vector>

#include <Eigen/Dense>

using scalar_t = double;
using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

class WBCTask {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    WBCTask() = default;

    WBCTask(matrix_t a, vector_t b, matrix_t d, vector_t f)
        : a_(std::move(a)), d_(std::move(d)), b_(std::move(b)), f_(std::move(f)) {}

    explicit WBCTask(size_t numDecisionVars)
        : WBCTask(matrix_t::Zero(0, numDecisionVars),
                  vector_t::Zero(0),
                  matrix_t::Zero(0, numDecisionVars),
                  vector_t::Zero(0)) {}

    WBCTask operator+(const WBCTask& rhs) const
    {
        return {concatenateMatrices(a_, rhs.a_),
                concatenateVectors(b_, rhs.b_),
                concatenateMatrices(d_, rhs.d_),
                concatenateVectors(f_, rhs.f_)};
    }

    WBCTask operator*(scalar_t rhs) const
    {
        return {a_.cols() > 0 ? rhs * a_ : a_,
                b_.cols() > 0 ? rhs * b_ : b_,
                d_.cols() > 0 ? rhs * d_ : d_,
                f_.cols() > 0 ? rhs * f_ : f_};
    }

    matrix_t a_, d_;
    vector_t b_, f_;

private:
    static matrix_t concatenateMatrices(const matrix_t& m1, const matrix_t& m2)
    {
        if (m1.cols() <= 0) return m2;
        if (m2.cols() <= 0) return m1;
        assert(m1.cols() == m2.cols());

        matrix_t res(m1.rows() + m2.rows(), m1.cols());
        res << m1, m2;
        return res;
    }

    static vector_t concatenateVectors(const vector_t& v1, const vector_t& v2)
    {
        if (v1.cols() <= 0) return v2;
        if (v2.cols() <= 0) return v1;
        assert(v1.cols() == v2.cols());

        vector_t res(v1.rows() + v2.rows());
        res << v1, v2;
        return res;
    }
};
