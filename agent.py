
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from playwright.sync_api import sync_playwright

class WebDOMEnv(gym.Env):
    def __init__(self, target_url):
        super().__init__()
        self.target_url = target_url

        # Setup Playwright
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.page = self.browser.new_page()

        # Limit max interactive elements we keep track of per page
        self.max_elements = 50

        # Action Space: Discrete index corresponding to the element to click
        self.action_space = spaces.Discrete(self.max_elements)

        # Observation Space: A simplified array/dictionary representing DOM element attributes
        # (For deep RL, you would encode text embeddings; here we use an empty placeholder)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.max_elements, 1), dtype=np.float32)
        self.interactive_elements = []

    def _parse_dom(self):
        """Extracts clickable elements from the current page state."""
        # Query typical clickable elements
        elements = self.page.query_selector_all("button, a, input[type='submit'], [role='button']")
        self.interactive_elements = elements[:self.max_elements]

        # Build an observation matrix (e.g., element existence flags)
        obs = np.zeros((self.max_elements, 1), dtype=np.float32)
        for i in range(len(self.interactive_elements)):
            obs[i] = 1.0
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.page.goto(self.target_url)
        self.page.wait_for_load_state("networkidle")

        obs = self._parse_dom()
        info = {}
        return obs, info

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # Execute click if the chosen action index corresponds to a real element
        if action < len(self.interactive_elements):
            try:
                target_element = self.interactive_elements[action]
                target_element.click(timeout=2000)
                self.page.wait_for_load_state("domcontentloaded")
            except Exception:
                reward -= 0.1  # Penalty for misclicking or trying to click an invisible item

        # Parse the new state of the DOM after the click
        obs = self._parse_dom()

        # Define your Success Criteria / Verifier
        if "success" in self.page.url:
            reward += 1.0
            terminated = True

        info = {}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.browser.close()
        self.playwright.stop()
