from google import genai
import json
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from typing import List, Dict, Any, Optional

# --- Configuration Area ---
GEMINI_API_KEY = "APIKEY"
VISION_MODEL_NAME = "gemini-2.0-flash"
TEXT_MODEL_NAME = "gemini-2.0-flash"
TARGET_URL = "https://movie.douban.com/tv/"

# Configure the Gemini client globally
try:
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring Gemini API: {e}. Please ensure GEMINI_API_KEY is set correctly.")
    exit()

# --- 1. Perception Module ---
class PerceptionModule:
    """
    Responsible for perceiving the web environment, processing screenshots
    and optional HTML information using the Gemini Vision API.
    Inspired by WebVoyager, combining visual and textual information.
    """
    def __init__(self):
        # Initialize the specific model
        self.model = GEMINI_CLIENT.models

    def perceive(self, screenshot_path: str, simplified_html: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes the screenshot and optional HTML snippet using the Gemini Vision API
        to extract key information about the current page state.

        Args:
            screenshot_path: Path to the current webpage screenshot file.
            simplified_html: Optional simplified HTML snippet containing interactive element info.

        Returns:
            A dictionary containing a structured understanding of the page state, e.g.:
            {
                "summary": "Page summary...",
                "interactive_elements": [
                    {"id": "element_1", "type": "button", "text": "More", "visual_description": "Blue button mid-page"},
                    # ... other elements
                ],
                "potential_actions": ["Scroll down", "Click 'More'"]
            }
            Returns an error dictionary if the API call fails or parsing fails.
        """
        print(f"Perception Module: Analyzing screenshot {screenshot_path}...")
        if not Path(screenshot_path).is_file():
             return {"error": f"Screenshot file not found: {screenshot_path}"}

        # Prepare the prompt for the Vision API
        prompt_text = f"""
        Analyze this webpage screenshot. Considering the optional simplified HTML info below (if provided), perform these tasks:
        1. Briefly summarize the main content and purpose of the page.
        2. Identify and list the main *interactive* elements (buttons, links, inputs, selects, etc.). For each, provide:
           - A temporary unique ID (e.g., 'element_1', 'element_2').
           - The element type (e.g., 'button', 'link', 'input', 'select').
           - The visible text or its function (e.g., 'Login', 'Next Page', 'Search movies').
           - A visual description or location (e.g., 'Red button top-right', 'Third link in list').
           #- (Optional but helpful) A precise CSS Selector or XPath for this element if possible.
        3. Identify and list key *content* elements relevant to the likely user goal (like product names, prices, article titles, list items, drama titles, ratings). For each, provide:
           - A type describing the content (e.g., 'drama_title', 'rating', 'list_item', 'header').
           - The text content of the element.
        4. Based on the page content, suggest some possible user actions.
        5. IMPORTANT: Structure your entire response as a single JSON object. Example format:
           {{
             "summary": "...",
             "interactive_elements": [
               {{"id": "...", "type": "...", "text": "...", "visual_description": "..."}}
               # Example with locator:
               # {{"id": "...", "type": "...", "text": "...", "visual_description": "...", "css_selector": "#loginButton"}}
             ],
             "content_elements": [
                {{"type": "drama_title", "text": "Example Drama 1"}},
                {{"type": "rating", "text": "8.5"}},
                ...
             ],
             "potential_actions": ["...", "..."]
           }}

        Simplified HTML (if available):
        {simplified_html if simplified_html else 'None'}
        """

        try:
            # Upload the file to the Gemini API (required for gemini-1.5 models)
            # Note: For gemini-pro-vision, you might use inline data instead. Check SDK docs.
            print("Perception Module: Uploading screenshot...")
            uploaded_file = GEMINI_CLIENT.files.upload(file=screenshot_path)
            print(f"Perception Module: Screenshot uploaded successfully (URI: {uploaded_file.uri}).")

            # Generate content using the vision model
            print("Perception Module: Calling Gemini Vision API...")
            # Ensure contents list format matches SDK requirements
            response = self.model.generate_content(
                model=VISION_MODEL_NAME,
                contents=[uploaded_file, prompt_text]
            )
            print("Perception Module: Received response from Gemini Vision API.")

            # Clean potential markdown code block fences
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            # Parse the JSON response
            structured_state = json.loads(response_text)
            print("Perception Module: State parsed successfully.")
            # Basic validation
            if not isinstance(structured_state, dict) or "interactive_elements" not in structured_state:
                 raise ValueError("Parsed JSON does not have the expected structure.")

            # Clean up the uploaded file after use
            print(f"Perception Module: Deleting uploaded file {uploaded_file.name}...")
            GEMINI_CLIENT.files.delete(name=uploaded_file.name)
            print("Perception Module: Uploaded file deleted.")
            # print("structured_state:", structured_state)
            return structured_state

        except Exception as e:
            print(f"Perception Module: Error during perception: {e}")
            # Attempt to delete file even if generation failed, if it was uploaded
            if 'uploaded_file' in locals() and uploaded_file:
                try:
                    print(f"Perception Module: Attempting to delete file {uploaded_file.name} after error...")
                    GEMINI_CLIENT.files.delete(name=uploaded_file.name)
                    print("Perception Module: File deleted after error.")
                except Exception as delete_error:
                    print(f"Perception Module: Could not delete file after error: {delete_error}")
            return {"error": str(e)}


# --- 2. Planning & Reasoning Module ---
class PlanningReasoningModule:
    """
    Decides the next action based on the user's goal, current state, and history.
    Uses a text-based LLM (e.g., Gemini Text API) for reasoning.
    References the WebAgent paper's idea of potentially decomposing complex tasks.
    """
    def __init__(self):
        # Initialize the specific text model
        self.model = GEMINI_CLIENT.models

    def plan_action(self, goal: str, current_state: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates the next action based on the goal, perceived state, and action history.

        Args:
            goal: The user's ultimate objective.
            current_state: The structured state returned by PerceptionModule.
            history: List of previously executed actions.

        Returns:
            A dictionary describing the action to execute, e.g.:
            {'action_type': 'click', 'element_id': 'element_1_parsed'}
            {'action_type': 'type', 'element_id': 'element_2_parsed', 'text': 'Sci-Fi'}
            {'action_type': 'scroll', 'direction': 'down'}
            {'action_type': 'stop', 'reason': 'Task completed'}
            {'action_type': 'error', 'message': 'Cannot plan action'}
        """
        print("Planning Module: Planning next action...")
        if "error" in current_state:
            return {'action_type': 'error', 'message': f'Perception module failed: {current_state["error"]}'}
        if not current_state.get("interactive_elements"):
            print("Warning: No interactive elements identified by perception module.")
            # Can still attempt to scroll or stop

        # Construct the prompt for the Gemini Text API
        prompt = f"""
        You are a web assistant. Decide the next best action based on the user's goal, current web page state, and action history.

        User Goal: {goal}

        Current Web Page State Summary: {current_state.get('summary', 'No summary available')}

        Interactive Elements on Page:
        {self._format_elements(current_state.get('interactive_elements', []))}

        Relevant Content Elements on Page:
        {self._format_content(current_state.get('content_elements', []))}

        Action History (Last {len(history)} steps):
        {self._format_history(history)}

        Based on the goal and the current page content/elements, decide the next step.
        * If the information needed to answer the user's goal ('{goal}') is present in the 'Relevant Content Elements', generate an ANSWER action.
        * Otherwise, determine the best interaction (click, type, select, scroll) to get closer to the goal.
        * If the goal seems impossible or already achieved in a previous step (check history), generate a STOP action.

        Your response MUST be one of the following JSON formats:
        1. Click element: {{"action_type": "click", "element_id": "element_id_to_click"}}
        2. Type text into input: {{"action_type": "type", "element_id": "input_element_id", "text": "text_to_type"}}
        3. Select option from dropdown: {{"action_type": "select", "element_id": "select_element_id", "option_text": "text_of_option_to_select"}}
        4. Scroll page down: {{"action_type": "scroll", "direction": "down"}}
        5. Scroll page up: {{"action_type": "scroll", "direction": "up"}}
        6. Answer the user's goal: {{"action_type": "ANSWER", "content": "The answer based on current content elements"}}
        7. Task finished or cannot proceed: {{"action_type": "stop", "reason": "Explain why"}}
        8. Navigate to a new URL (use only if necessary): {{"action_type": "navigate", "url": "target_url"}}

        Output ONLY the JSON for the next action.
        """

        try:
            # Generate content using the text model
            print("Planning Module: Calling Gemini Text API...")
            response = self.model.generate_content(
                model=TEXT_MODEL_NAME,
                contents = prompt,
            )
            print("Planning Module: Received response from Gemini Text API.")

            # Clean potential markdown code block fences
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            print(f"Planning Module: Gemini proposed action text: {response_text}")

            # Parse the JSON response
            action = json.loads(response_text)

            # Validate action format
            if 'action_type' not in action:
                raise ValueError("Response missing 'action_type'")

            print(f"Planning Module: Planned action: {action}")
            return action

        except json.JSONDecodeError as e:
            print(f"Planning Module: Failed to parse Gemini JSON response: {e}")
            print(f"Original text: {response_text}")
            return {'action_type': 'error', 'message': f'Failed to parse planning response: {e}'}
        except Exception as e:
            print(f"Planning Module: Error during planning: {e}")
            return {'action_type': 'error', 'message': f'Unknown planning error: {e}'}

    def _format_elements(self, elements: List[Dict[str, Any]]) -> str:
        if not elements:
            return "None"
        formatted = []
        for el in elements:
            text_or_desc = el.get('text') or el.get('visual_description') or 'No text/desc'
            formatted.append(f"- ID: {el.get('id', 'Unknown')}, Type: {el.get('type', 'Unknown')}, Text/Desc: {text_or_desc}")
        return "\n".join(formatted)

    def _format_content(self, elements: List[Dict[str, Any]]) -> str:
        if not elements:
            return "None"
        formatted = []
        for el in elements:
            formatted.append(f"- Type: {el.get('type', 'Unknown')}, Content: {el.get('text', 'N/A')}")
        return "\n".join(formatted)

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        if not history:
            return "None"
        formatted = [f"Step {i+1}: {action}" for i, action in enumerate(history)]
        return "\n".join(formatted)


# --- 3. Execution Module ---
class ExecutionModule:
    """
    Translates planned abstract actions into concrete browser operations (using Selenium).
    """
    def __init__(self, driver: webdriver.Remote):
        self.driver = driver

    def execute(self, action: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Executes the given action.

        Args:
            action: The action command from PlanningReasoningModule.
            state: The current state from PerceptionModule, used for locating elements.

        Returns:
            True if execution was successful, False otherwise.
        """
        action_type = action.get('action_type')
        print(f"Execution Module: Attempting action - {action_type}")

        try:
            if action_type == "click":
                element = self._find_element_for_action(action.get('element_id'), state)
                if element:
                    print(f"  Clicking element: {action.get('element_id')}")
                    element.click()
                    return True
                else:
                    print(f"  Error: Could not find element to click with ID: {action.get('element_id')}")
                    return False
            elif action_type == "type":
                element = self._find_element_for_action(action.get('element_id'), state)
                text_to_type = action.get('text', '')
                if element:
                    print(f"  Typing '{text_to_type}' into element {action.get('element_id')}")
                    element.clear() # Clear first
                    element.send_keys(text_to_type)
                    return True
                else:
                    print(f"  Error: Could not find element to type into with ID: {action.get('element_id')}")
                    return False
            elif action_type == "select":
                element_id = action.get('element_id')
                option_text = action.get('option_text')
                print(f"  Selecting: Element {element_id}, Option '{option_text}'")
                element = self._find_element_for_action(element_id, state)
                if element and element.tag_name.lower() == 'select':
                    from selenium.webdriver.support.ui import Select
                    select_obj = Select(element)
                    try:
                        select_obj.select_by_visible_text(option_text)
                        print(f"    Successfully selected option '{option_text}'.")
                        return True
                    except NoSuchElementException:
                        print(f"  Error: Option '{option_text}' not found in select element {element_id}")
                         # Try selecting by value as a fallback if needed
                         # try:
                         #     select_obj.select_by_value(option_text)
                         #     print(f"    Successfully selected option by value '{option_text}'.")
                         #     return True
                         # except NoSuchElementException:
                         #     print(f"  Error: Option '{option_text}' not found by text or value.")
                         #     return False
                        return False
                else:
                    print(f"  Error: Element {element_id} not found or is not a select element.")
                    return False
            elif action_type == "scroll":
                direction = action.get('direction', 'down')
                print(f"  Scrolling page: {direction}")
                scroll_amount = "window.innerHeight" if direction == 'down' else "-window.innerHeight"
                self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
                return True
            elif action_type == "navigate":
                url = action.get('url')
                if url:
                    print(f"  Navigating to new URL: {url}")
                    self.driver.get(url)
                    return True
                else:
                    print("  Error: Navigate action missing URL.")
                    return False
            elif action_type == "ANSWER":
                 # This action is handled by the main loop, not executed on the browser
                 print(f"  ANSWER action generated: {action.get('content')}")
                 return True # Considered successful execution            
            elif action_type == "stop":
                print(f"  Execution stopped as planned: {action.get('reason', 'No reason specified')}")
                return True # Stop is considered a successful execution outcome
            elif action_type == "error":
                print(f"  Execution failed (due to planning error): {action.get('message', 'Unknown error')}")
                return False
            else:
                print(f"  Error: Unknown action type '{action_type}'")
                return False
        except WebDriverException as e:
            print(f"Execution Module: Selenium error during execution: {e}")
            return False
        except Exception as e:
            print(f"Execution Module: Unexpected error during execution: {e}")
            return False

    def _find_element_for_action(self, element_id: Optional[str], state: Dict[str, Any]) -> Optional[WebElement]:
        """
        Attempts to locate the actual WebElement on the page based on the element_id
        provided by the PerceptionModule. This is a simplified version.
        Robust implementation requires better locators (XPath, CSS) from perception.
        """
        if not element_id:
            return None

        target_element_info = next((el for el in state.get('interactive_elements', []) if el.get('id') == element_id), None)

        if not target_element_info:
            print(f"  Warning: Element info for ID '{element_id}' not found in perceived state.")
            return None

        # --- Ideal Scenario: Use Locator from Perception ---
        # Check if perception module provided a precise locator
        css_selector = target_element_info.get('css_selector')
        xpath = target_element_info.get('xpath')

        if css_selector:
            try:
                element = self.driver.find_element(By.CSS_SELECTOR, css_selector)
                if element.is_displayed() and element.is_enabled():
                    print(f"    Located interactable element for ID '{element_id}' using provided CSS Selector: {css_selector}")
                    return element
            except NoSuchElementException:
                print(f"    Provided CSS Selector '{css_selector}' not found.")
            except Exception as e:
                print(f"    Error using CSS Selector '{css_selector}': {e}")

        if xpath:
            try:
                element = self.driver.find_element(By.XPATH, xpath)
                if element.is_displayed() and element.is_enabled():
                    print(f"    Located interactable element for ID '{element_id}' using provided XPath: {xpath}")
                    return element
            except NoSuchElementException:
                print(f"    Provided XPath '{xpath}' not found.")
            except Exception as e:
                print(f"    Error using XPath '{xpath}': {e}")

        # --- Fallback: Heuristic Locators Based on Text/Type ---
        print(f"  Info: No precise locator provided for ID '{element_id}'. Falling back to heuristic text search.")
        element_text = target_element_info.get('text')
        element_type = target_element_info.get('type', '').lower()

        locators = []
        if element_text:
            # 1. Prioritize exact matches on specific tags
            if element_type == 'link':
                locators.append((By.XPATH, f".//a[normalize-space(.)='{element_text}']")) # Exact match specific tag
                locators.append((By.LINK_TEXT, element_text))
            elif element_type == 'button':
                locators.append((By.XPATH, f".//button[normalize-space(.)='{element_text}']"))
                locators.append((By.XPATH, f".//input[@type='button' and @value='{element_text}']"))
                locators.append((By.XPATH, f".//input[@type='submit' and @value='{element_text}']"))
            elif element_type == 'input':
                 locators.append((By.XPATH, f".//input[@placeholder='{element_text}']"))
                 locators.append((By.XPATH, f".//textarea[@placeholder='{element_text}']"))
                 locators.append((By.XPATH, f".//input[@aria-label='{element_text}']"))
                 # Find input associated with a label containing the text
                 locators.append((By.XPATH, f"//label[contains(normalize-space(), '{element_text}')]/preceding-sibling::input[1]"))
                 locators.append((By.XPATH, f"//label[contains(normalize-space(), '{element_text}')]/following-sibling::input[1]"))
                 locators.append((By.XPATH, f".//input[@id=(//label[contains(normalize-space(), '{element_text}')]/@for)]")) # Match label's 'for' attribute
            elif element_type == 'select':
                 locators.append((By.XPATH, f".//select[@aria-label='{element_text}']"))
                 locators.append((By.XPATH, f"//label[contains(normalize-space(), '{element_text}')]/following-sibling::select[1]"))
                 locators.append((By.XPATH, f"//label[contains(normalize-space(), '{element_text}')]/preceding-sibling::select[1]"))
                 locators.append((By.XPATH, f".//select[@id=(//label[contains(normalize-space(), '{element_text}')]/@for)]"))

            # 2. Try exact match on any tag (less specific)
            locators.append((By.XPATH, f"//*[normalize-space(.)='{element_text}']"))

            # 3. Try contains match on specific tags (fallback)
            if element_type == 'link':
                locators.append((By.PARTIAL_LINK_TEXT, element_text))
                locators.append((By.XPATH, f".//a[contains(normalize-space(), '{element_text}')]"))
            elif element_type == 'button':
                locators.append((By.XPATH, f".//button[contains(normalize-space(), '{element_text}')]"))
                locators.append((By.XPATH, f".//input[@type='button' and contains(@value, '{element_text}')]"))
                locators.append((By.XPATH, f".//input[@type='submit' and contains(@value, '{element_text}')]"))

            # 4. Generic contains match (last resort)
            locators.append((By.XPATH, f"//*[contains(normalize-space(), '{element_text}')]"))

        # Remove duplicate locators while preserving order
        seen_locators = set()
        unique_locators = []
        for loc in locators:
            loc_tuple = (loc[0], loc[1]) # Make hashable
            if loc_tuple not in seen_locators:
                unique_locators.append(loc)
                seen_locators.add(loc_tuple)

        # Try locators in order
        for by, value in unique_locators:
            try:
                elements = self.driver.find_elements(by, value)
                # Filter for the most specific, interactable element
                best_element = None
                min_depth = float('inf')

                for element in elements:
                    if element.is_displayed() and element.is_enabled():
                         # Heuristic: Prefer elements that directly contain the text, not just descendants
                         # And prefer elements deeper in the DOM tree (more specific)
                         element_own_text = self.driver.execute_script("return arguments[0].textContent", element).strip()
                         # Check if the element's direct text matches or contains the target text
                         # This helps avoid picking large containers
                         is_direct_match = element_text in element_own_text

                         # Calculate depth (simple heuristic)
                         depth = len(element.find_elements(By.XPATH, "./ancestor::*"))

                         # Prioritize direct matches and deeper elements
                         if is_direct_match: # Strong preference for direct match
                              if depth < min_depth : # If multiple direct matches, pick deepest
                                   min_depth = depth
                                   best_element = element
                         elif best_element is None: # If no direct match found yet, take the first interactable one
                              best_element = element
                              min_depth = depth
                         # Else: Already have a candidate, and this one isn't a direct match, so ignore unless it's the only option

                if best_element:
                     print(f"    Located interactable element for ID '{element_id}' using heuristic {by}: {value} (Selected best candidate)")
                     return best_element

            except NoSuchElementException:
                continue # Try next locator
            except Exception as e:
                 print(f"    Error checking locator {by}={value}: {e}") # Log other errors

        print(f"  Warning: Execution module could not reliably locate element for ID '{element_id}' with text '{element_text}'.")
        return None


# --- Main Agent Class ---
class SimpleWebAgent:
    def __init__(self, goal: str, start_url: str):
        self.goal = goal
        self.start_url = start_url

        options = webdriver.ChromeOptions()
        # options.add_argument('--headless') # Uncomment for headless mode
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        try:
            self.driver = webdriver.Chrome(options=options)
        except WebDriverException as e:
            print(f"Error initializing WebDriver: {e}")
            print("Please ensure ChromeDriver is installed and accessible in your PATH.")
            exit()

        self.driver.set_window_size(1280, 900) # Set a reasonable window size

        # Initialize modules
        self.perception_module = PerceptionModule()
        self.planning_module = PlanningReasoningModule()
        self.execution_module = ExecutionModule(self.driver)
        self.history = []
        self.max_steps = 15 # Limit steps to prevent infinite loops
        self.final_answer = None

    def run(self):
        """Executes the main loop of the WebAgent."""
        print(f"WebAgent starting. Goal: {self.goal}")
        try:
            self.driver.get(self.start_url)
            print(f"Navigated to: {self.start_url}")
            time.sleep(5) # Initial wait for page load

            for step in range(self.max_steps):
                print(f"\n--- Step {step + 1}/{self.max_steps} ---")

                # 1. Perceive
                screenshot_file = f"screenshot_step_{step + 1}.png"
                try:
                    if not self.driver.save_screenshot(screenshot_file):
                        print("Error: Failed to save screenshot.")
                        break
                except WebDriverException as e:
                    print(f"Error saving screenshot: {e}")
                    break

                # Optionally extract simplified HTML here if needed
                current_state = self.perception_module.perceive(screenshot_file)

                if "error" in current_state:
                    print(f"Perception failed, stopping agent. Error: {current_state['error']}")
                    break
                
                # 2. Plan
                action = self.planning_module.plan_action(self.goal, current_state, self.history)

                if action.get('action_type') == 'ANSWER':
                    self.final_answer = action.get('content')
                    print(f"Goal achieved! Final Answer: {self.final_answer}")
                    break # Exit loop after getting the answer

                # 3. Execute
                if action.get('action_type') == 'stop':
                    print(f"Planning module decided to stop: {action.get('reason', 'No reason specified')}")
                    break
                if action.get('action_type') == 'error':
                    print(f"Planning failed, stopping agent. Error: {action.get('message', 'Unknown error')}")
                    break

                success = self.execution_module.execute(action, current_state)
                self.history.append(action) # Record the attempted action

                if not success:
                    print("Execution module reported failure. Stopping agent.")
                    break

                # Wait for potential page changes or loading
                time.sleep(4)

            else: # for loop finished without break (max_steps reached)
                print(f"\nMaximum steps ({self.max_steps}) reached. Stopping agent.")

            if self.final_answer:
                print(f"\n--- Final Result ---")
                print(f"Agent goal: {self.goal}")
                print(f"Agent answer: {self.final_answer}")
            else:
                print("\nAgent stopped without providing a final answer.")

        except WebDriverException as e:
            print(f"WebAgent encountered a WebDriver error: {e}")
        except Exception as e:
            print(f"WebAgent encountered an unexpected error: {e}")
        finally:
            print("WebAgent finished.")
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
            # Clean up screenshots
            for i in range(self.max_steps + 1):
                sf = Path(f"screenshot_step_{i}.png")
                if sf.exists():
                    try:
                        sf.unlink()
                    except OSError as e:
                        print(f"Error deleting screenshot {sf}: {e}")


if __name__ == "__main__":
    user_goal = "在豆瓣电视剧页面找到近期热门的电视剧列表，并告诉我前三个的名字"

    agent = SimpleWebAgent(goal=user_goal, start_url=TARGET_URL)
    agent.run()
