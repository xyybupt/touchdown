from base_navigator import BaseNavigator
import os
import random
import numpy as np
from PIL import Image

IMAGE_ROOT = "/media/data3/xuyuanyuan/jpegs_manhattan_touchdown_2022/jpegs_manhattan_touchdown_2022"
POLICY_MODE = "shortest_path"
GOAL_PANOID = "5DrpCWbrHxKV4uH31RrP7A"


class Navigator(BaseNavigator):
    def __init__(self):
        super(Navigator, self).__init__()

    def navigate(self, start_graph_state, show_info, goal_panoid=None):
        self.graph_state = start_graph_state
        
        while True:
            image_feature = self.get_image_feature(self.graph_state)
            if POLICY_MODE == "shortest_path" and goal_panoid is not None:
                move = self.shortest_path_policy(self.graph_state, goal_panoid)
            else:
                move = self.random_policy(image_feature)
            if move == 'stop':
                print('Action `stop` is chosen.')
                break
            self.step(move)

            if show_info:
                self.show_state_info(self.graph_state)

    def policy(self, state):
        raise NotImplementedError

    def get_image_feature(self, graph_state):
        panoid, heading = graph_state

        image_path = os.path.join(IMAGE_ROOT, f"{panoid}.jpg")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found for panoid={panoid}: {image_path}")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image_feature = np.array(img)

        shift_angle = 157.5 + self.graph.nodes[panoid].pano_yaw_angle - heading
        width = image_feature.shape[1]
        shift = int(width * shift_angle / 360)
        image_feature = np.roll(image_feature, shift, axis=1)

        return image_feature

    def random_policy(self, state):
        return random.choice(['forward', 'left', 'right', 'stop'])

    def shortest_path_policy(self, graph_state, goal_panoid):
        curr_panoid, _ = graph_state
        if curr_panoid == goal_panoid:
            return "stop"

        path = self._find_shortest_path(curr_panoid, goal_panoid)
        if path is None or len(path) < 2:
            return "stop"

        next_panoid = path[1]
        available_actions, next_graph_states = self.get_available_next_moves(graph_state)
        for action, next_state in zip(available_actions, next_graph_states):
            if next_state[0] == next_panoid:
                return action
        return "forward"

    def _find_shortest_path(self, start_panoid, goal_panoid):
        queue = [(start_panoid, [start_panoid])]
        visited = {start_panoid}

        while queue:
            current, path = queue.pop(0)
            if current == goal_panoid:
                return path
            neighbors = self.graph.nodes[current].neighbors.values()
            for neighbor in neighbors:
                panoid = neighbor.panoid
                if panoid in visited:
                    continue
                visited.add(panoid)
                queue.append((panoid, path + [panoid]))
        return None


if __name__ == '__main__':
    navigator = Navigator()
    navigator.navigate(
        start_graph_state=('sbtZW9Akt4izrxdQRDPwMQ', 209), 
        show_info=True,
        goal_panoid=GOAL_PANOID
    )
