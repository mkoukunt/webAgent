import math
import random
import xml.etree.ElementTree as ET


class DOMNodeState:

  def __init__(self, elem, target_tag="target"):
    self.elem = elem
    self.target_tag = target_tag

  def get_possible_actions(self):
    # Each child element is a possible action/transition
    return list(self.elem)

  def take_action(self, child_elem):
    return DOMNodeState(child_elem, self.target_tag)

  def is_terminal(self):
    # Terminal if it matches the target or has no children
    return (
        self.elem.tag == self.target_tag
        or len(list(self.elem)) == 0
    )

  def get_reward(self):
    # Simple heuristic: high reward if tag matches target
    if self.elem.tag == self.target_tag:
      return 1.0
    # Or score based on text / attributes
    return 0.1 if "important" in self.elem.attrib.get("class", "") else 0.0


class MCTSNode:

  def __init__(self, state, parent=None, action=None):
    self.state = state
    self.parent = parent
    self.action = action
    self.children = []
    self.visits = 0
    self.value = 0.0
    self.unvisited_actions = state.get_possible_actions()

  def is_fully_expanded(self):
    return len(self.unvisited_actions) == 0

  def best_child(self, c_param=1.41):
    def ucb1(child):
      return (
          child.value / (child.visits + 1e-6)
          + c_param
          * math.sqrt(2 * math.log(self.visits + 1) / (child.visits + 1e-6))
      )

    # Return child with max UCB1
    return max(self.children, key=ucb1)

  def expand(self):
    action = self.unvisited_actions.pop()
    next_state = self.state.take_action(action)
    child_node = MCTSNode(next_state, parent=self, action=action)
    self.children.append(child_node)
    return child_node


def run_mcts_dom_search(root_elem, target_tag, iterations=100):
  root_state = DOMNodeState(root_elem, target_tag)
  root_node = MCTSNode(root_state)

  for _ in range(iterations):
    node = root_node
    state = root_state

    # 1. Selection
    while node.is_fully_expanded() and node.children:
      node = node.best_child()
      state = node.state

    # 2. Expansion
    if not node.state.is_terminal() and node.unvisited_actions:
      node = node.expand()
      state = node.state

    # 3. Simulation (Rollout)
    sim_state = state
    depth = 0
    while not sim_state.is_terminal() and depth < 10:
      actions = sim_state.get_possible_actions()
      if not actions:
        break
      sim_state = sim_state.take_action(random.choice(actions))
      depth += 1

    reward = sim_state.get_reward()

    # 4. Backpropagation
    while node is not None:
      node.visits += 1
      node.value += reward
      node = node.parent

  return root_node
