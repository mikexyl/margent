#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Marginals.h>

using namespace gtsam;

class MarginalsExposeBayesTree : public Marginals {
 public:
  MarginalsExposeBayesTree(const NonlinearFactorGraph& graph,
                           const Values& values)
      : Marginals(graph, values), cached_linearization_points_(values) {}

  static MarginalsExposeBayesTree FromOptimalValues(
      const NonlinearFactorGraph& graph,
      const Values& values) {
    auto marginalization = MarginalsExposeBayesTree(graph, values);
    // marginalize all variables and store them
    auto marginals = marginalization.marginalCovariance(values.keySet());
    for (const auto& [key, value] : marginals) {
      marginalization.cached_marginals_[key] = value;
    }
    return marginalization;
  }

  const GaussianBayesTree& getBayesTree() const { return bayesTree_; }
  void setBayesTree(const GaussianBayesTree& bayesTree) {
    bayesTree_ = bayesTree;
  }

  void augment(const NonlinearFactorGraph& newFactors) {}

  static double Chi2(const Vector& mu1,
                     const Matrix& Sigma1,
                     const Vector& mu2) {
    Vector diff = mu2 - mu1;
    return diff.transpose() *
           Sigma1.ldlt().solve(diff);  // More stable than inverse
  }

  std::map<Key, Matrix> marginalCovariance(KeySet keys) {
    return Marginals::marginalCovariance(keys);
  }

  std::map<Key, Matrix> marginalCovariance(KeySet keys,
                                           const Values& updated_values) {
    KeySet skipped_keys, approx_keys;
    for (const auto& key : keys) {
      auto value = updated_values.at<Pose3>(key);
      auto mu = traits<Pose3>::Logmap(value);
      if (cached_linearization_points_.exists(key)) {
        auto cached_value = cached_linearization_points_.at<Pose3>(key);
        auto cached_mu = traits<Pose3>::Logmap(cached_value);
        auto chi2 = Chi2(cached_mu, cached_marginals_.at(key), mu);
        if (chi2 < 4) {
          skipped_keys.insert(key);
        } else if (chi2 < 10) {
          approx_keys.insert(key);
        }
      }
    }

    // erase skipped keys
    for (const auto& key : skipped_keys) {
      keys.erase(key);
    }

    std::map<Key, Matrix> marginals;

    for (const auto& key : approx_keys) {
      auto value = updated_values.at<Pose3>(key);
      auto mu = traits<Pose3>::Logmap(value);
      auto cached_value = cached_linearization_points_.at<Pose3>(key);
      auto cached_mu = traits<Pose3>::Logmap(cached_value);
      auto Sigma0 = cached_marginals_.at(key);
      auto Sigma1 = SMWMarginalApprox(cached_mu, Sigma0, mu);
      marginals[key] = Sigma1;
    }

    auto computed_marginals = Marginals::marginalCovariance(keys);

    for (const auto& key : skipped_keys) {
      marginals[key] = cached_marginals_.at(key);
    }

    //  merge the two maps
    for (const auto& [key, value] : computed_marginals) {
      marginals[key] = value;
    }

    std::cout << "number of approximated marginals: " << approx_keys.size()
              << std::endl;
    std::cout << "number of skipped marginals: " << skipped_keys.size()
              << std::endl;
    std::cout << "number of computed marginals: " << computed_marginals.size()
              << std::endl;

    return marginals;
  }

  Matrix SMWMarginalApprox(const Vector& mu0,
                           const Matrix& Sigma0,
                           const Vector& mu1) {
    Vector delta = mu1 - mu0;

    Eigen::LLT<Matrix> llt(Sigma0);
    if (llt.info() != Eigen::Success) {
      throw std::runtime_error("Sigma0 is not positive definite");
    }
    Vector u = llt.solve(delta);

    double denom = 1.0 - delta.transpose() * u;  // Now subtracting
    if (std::abs(denom) < 1e-10) {
      throw std::runtime_error(
          "Denominator too small — near singular correction");
    }

    Matrix correction = (Sigma0 * u * u.transpose() * Sigma0) / denom;

    Matrix Sigma1 = Sigma0 + correction;  // Grows uncertainty

    return Sigma1;
  }

 protected:
  Values cached_linearization_points_;
  std::map<Key, Matrix> cached_marginals_;
};