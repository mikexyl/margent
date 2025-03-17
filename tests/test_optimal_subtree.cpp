#include <aria_viz/visualizer_rerun.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/slam/dataset.h>

#include "margent/subtree_search.h"
#include "margent/MarginalsExposeBayesTree.h"

using namespace gtsam;

int main() {
  auto [graph, values] = readG2o("smallGrid3D.g2o", true);
  // add prior to first pose
  graph->add(PriorFactor<Pose3>(0, Pose3(), noiseModel::Unit::Create(6)));
  // optimize
  GaussNewtonOptimizer optimizer(*graph, *values);
  *values = optimizer.optimize();

  MarginalsExposeBayesTree marginals(*graph, *values);
  auto bayesTree = marginals.getBayesTree();

  aria::viz::VisualizerRerun visualizer({"test_optimal_subtree"});

  visualizer.drawBayesTree(
      "bayes_tree", bayesTree, Eigen::Vector4f(0, 0, 0, 1));

  // Create cost function instances
  std::shared_ptr<CostFunction> conditionCost =
      std::make_shared<ConditionNumberCost>(1e6);
  std::shared_ptr<CostFunction> frontalCost =
      std::make_shared<FrontalVariableCountCost>(-1000);

  // Define weights (adjust these to prioritize different costs)
  std::vector<std::shared_ptr<CostFunction>> costFunctions = {conditionCost,
                                                              frontalCost};
  std::vector<double> weights = {
      0.4,
      0.6};  // 70% importance on condition number, 30% on frontal variables

  // Create combined cost function
  auto combinedCost =
      std::make_shared<CombinedCostFunction>(costFunctions, weights, 10000);

  // Set total cost threshold
  double maxThreshold = 1000.0;

  // Search for the best subtree using condition number heuristic
  auto [optimalSubtree, totalCost] =
      SubtreeSearch::searchOptimalSubtree(bayesTree, combinedCost);

  // Print the results
  std::cout << "Optimal subtree contains " << optimalSubtree.size()
            << " cliques, total cost: " << totalCost << std::endl;
  std::cout
      << std::format(
             "Optimal subtree contains {} out of {} cliques, total cost: {}",
             optimalSubtree.size(),
             bayesTree.nodes().size(),
             totalCost)
      << std::endl;

  visualizer.drawBayesTreeEdges(
      "bayes_tree", optimalSubtree, {{255, 255, 100, 0.9}}, 1.0, false);
  return 0;
}