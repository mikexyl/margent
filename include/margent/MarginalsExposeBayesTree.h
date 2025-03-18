#include <gtsam/nonlinear/Marginals.h>

using namespace gtsam;

class MarginalsExposeBayesTree : public Marginals {
 public:
  MarginalsExposeBayesTree(const NonlinearFactorGraph& graph,
                           const Values& values)
      : Marginals(graph, values) {}

  const GaussianBayesTree& getBayesTree() const { return bayesTree_; }
  void setBayesTree(const GaussianBayesTree& bayesTree) {
    bayesTree_ = bayesTree;
  }
};