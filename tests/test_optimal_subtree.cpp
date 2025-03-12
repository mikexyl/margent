#include <aria_viz/visualizer_rerun.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/slam/dataset.h>

#include "margent/subtree_search.h"

using namespace gtsam;

int main() {
  auto [graph, values] = readG2o("tinyGrid3D.g2o", true);
  // add prior to first pose
  graph->add(PriorFactor<Pose3>(0, Pose3(), noiseModel::Unit::Create(6)));
  // optimize
  GaussNewtonOptimizer optimizer(*graph, *values);
  *values = optimizer.optimize();

  MarginalsExposeBayesTree marginals(*graph, *values);
  marginals.getBayesTree();

  aria::viz::VisualizerRerun visualizer({"test_optimal_subtree"});

  visualizer.drawBayesTree(
      "bayes_tree", marginals.getBayesTree(), Eigen::Vector4f(1, 0, 0, 1));
  return 0;
}